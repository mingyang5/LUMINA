from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import os
import random
import torch
from image_retrieval import run_dino_image_retrieval, get_image_files, text_image_retrieval

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from agent_models.utils import *

# Same dataset folders as original structure
DATASET_FOLDERS = [
    "./datasets/DiffusionDB",
    "./datasets/generated_images",
    "./datasets/GAIC",
    "./datasets/open_vid"
]

VIDEO_DATASET_FOLDERS = [
    "./datasets/generated_images",
]

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_all_images():
    all_images = []
    for folder in DATASET_FOLDERS:
        if os.path.isdir(folder):
            all_images.extend([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", "webp"))
            ])
    return all_images

def get_all_videos():
    all_videos = []
    for folder in VIDEO_DATASET_FOLDERS:
        if os.path.isdir(folder):
            all_videos.extend([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".mp4", ".webm", ".mov"))
            ])
    return all_videos

@app.route('/api/text_image_retrieval', methods=['POST'])
def text_image_retrieval_api():
    data = request.get_json()
    if not data or 'query_text' not in data:
        return jsonify({'error': 'Missing query_text'}), 400

    text_query = data['query_text']
    print(f"[INFO] Received text query: {text_query}")

    image_list = get_all_images()
    print(f"[INFO] Found {len(image_list)} images in dataset")

    if len(image_list) > 100:
        image_list = random.sample(image_list, 100)

    results_raw, results = text_image_retrieval(image_list, text_query, device)

    results = [
        {
            'path': f"/image_proxy?path={p}",
            'distance': float(dist),
            'model': model_name
        } for p, (dist, model_name) in results_raw
    ]

    print(f"[INFO] Top results: {results[:3]}")
    return jsonify({'results': results})


@app.route('/api/image_image_retrieval', methods=['POST'])
def image_image_retrieval_api():
    if 'query_image' not in request.files:
        return jsonify({'error': 'Missing file'}), 400

    uploaded_file = request.files['query_image']
    filename = secure_filename(uploaded_file.filename)
    query_image_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(query_image_path)

    print(f"[INFO] Received query image: {query_image_path}")

    image_list = get_all_images()
    print(f"[INFO] Total images in dataset: {len(image_list)}")

    if len(image_list) > 100:
        image_list = random.sample(image_list, 100)

    paths, dists = run_dino_image_retrieval(image_list, query_image_path, device, topk=8)

    results = [
        {
            'path': f"/image_proxy?path={p}",
            'distance': float(d)
        } for p, d in zip(paths, dists)
    ]

    print(f"[INFO] Top retrieval results: {[os.path.basename(r['path']) for r in results[:5]]}")

    return jsonify({'results': results})


def sample_video_frames(video_path, num_frames=5):
    """Sample `num_frames` evenly from the video and return paths to the extracted frames."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < num_frames:
        frame_ids = list(range(frame_count))
    else:
        interval = frame_count // num_frames
        frame_ids = [i * interval for i in range(num_frames)]

    frame_dir = os.path.join(UPLOAD_FOLDER, "video_frames", os.path.basename(video_path))
    os.makedirs(frame_dir, exist_ok=True)
    frame_paths = []

    for idx, fid in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_path = os.path.join(frame_dir, f"frame_{idx}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

    cap.release()
    return frame_paths


@app.route('/api/image_video_retrieval', methods=['POST'])
def image_video_retrieval_api():
    if 'query_image' not in request.files:
        return jsonify({'error': 'Missing file'}), 400

    uploaded_file = request.files['query_image']
    filename = secure_filename(uploaded_file.filename)
    query_image_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(query_image_path)

    print(f"[INFO] Received query image: {query_image_path}")
    
    video_list = get_all_videos()
    print(f"[INFO] Found {len(video_list)} videos in dataset")

    # if len(video_list) > 20:
    #     video_list = random.sample(video_list, 20)

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    query_vec = dino_image_query(query_image_path, model=model, device=device)

    scored_videos = []

    for video_path in video_list:
        try:
            frame_paths = sample_video_frames(video_path, num_frames=5)
            if len(frame_paths) == 0:
                continue

            index, _ = image_folder_embed(frame_paths, model=model, device=device)
            distances, _ = index.search(query_vec, len(frame_paths))
            avg_dist = float(distances.mean())
            scored_videos.append((video_path, avg_dist))

        except Exception as e:
            print(f"[ERROR] processing {video_path}: {e}")
            continue

    scored_videos = sorted(scored_videos, key=lambda x: x[1])
    top_videos = scored_videos[:5]

    results = [
        {
            'video_path': f"/video_proxy?path={video}",
            'distance': dist
        }
        for video, dist in top_videos
    ]

    print(f"[INFO] Top video results: {[os.path.basename(r['video_path']) for r in results]}")
    return jsonify({'results': results})


@app.route('/image_proxy')
def image_proxy():
    path = request.args.get("path")
    if path and os.path.isfile(path):
        return send_file(path, mimetype='image/png')
    return "Not Found", 404

@app.route('/video_proxy')
def video_proxy():
    path = request.args.get("path")
    if path and os.path.isfile(path):
        return send_file(path, mimetype='video/mp4')
    return "Video Not Found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

