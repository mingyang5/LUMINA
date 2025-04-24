from flask import Flask, jsonify, send_file, request
import os
import random
from flask_cors import CORS
import torch
from image_retrieval import run_dino_image_retrieval, get_image_files, text_image_retrieval

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_FOLDERS = [
    "./datasets/DiffusionDB",
    "./datasets/generated_images",
    "./datasets/GAIC"
    "./datasets/open_vid"
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
    for folder in DATASET_FOLDERS:
        if os.path.isdir(folder):
            all_videos.extend([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".mp4", ".webm", ".mov"))
            ])
    return all_videos

@app.route("/api/sample_images")
def sample_images():
    all_images = get_all_images()
    sampled = random.sample(all_images, min(500, len(all_images)))
    return jsonify(images=sampled)

@app.route("/api/sample_videos")
def sample_videos():
    all_videos = get_all_videos()
    sampled = random.sample(all_videos, min(10, len(all_videos)))
    return jsonify(videos=sampled)

@app.route("/image_proxy")
def image_proxy():
    path = request.args.get("path")
    if path and os.path.isfile(path):
        return send_file(path, mimetype='image/png')
    return "Not Found", 404

@app.route("/api/related_images")
def related_images():
    query_path = request.args.get("path")
    if not query_path or not os.path.isfile(query_path):
        return jsonify(related=[])

    print("[INFO] Performing image-image retrieval for:", query_path)
    base_dir = os.path.dirname(query_path)

    image_list = get_image_files(base_dir)
    if query_path not in image_list:
        image_list.append(query_path)

    if len(image_list) > 50:
        sampled_images = random.sample([img for img in image_list if img != query_path], 49)
        sampled_images.append(query_path)
    else:
        sampled_images = image_list

    print(f"[INFO] Using {len(sampled_images)} images for retrieval")
    paths, _ = run_dino_image_retrieval(sampled_images, query_path, device, topk=12)

    related = [p for p in paths if p != query_path][:12]
    return jsonify(related=related)

@app.route("/api/text_image_retrieval", methods=["POST"])
def text_image_retrieval_api():
    data = request.get_json()
    if not data or 'query_text' not in data:
        return jsonify({'error': 'Missing query_text'}), 400

    text_query = data['query_text']
    print(f"[INFO] Received text query: {text_query}")

    image_list = get_all_images()
    if len(image_list) > 50:
        image_list = random.sample(image_list, 50)

    results_raw, results = text_image_retrieval(image_list, text_query, device)

    results = [
        {
            'path': f"/image_proxy?path={p}",
            'distance': float(dist),
            'model': model_name
        } for p, (dist, model_name) in results_raw
    ]
    
    print("Top results:", results[:3])

    return jsonify({'results': results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)