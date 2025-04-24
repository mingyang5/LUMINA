# Modified version of the given code to align with the structure from the first code block

from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import os
import random
import torch
from image_retrieval import run_dino_image_retrieval, get_image_files, text_image_retrieval

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same dataset folders as original structure
DATASET_FOLDERS = [
    "./datasets/DiffusionDB",
    "./datasets/generated_images",
    "./datasets/GAIC",
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


@app.route('/image_proxy')
def image_proxy():
    path = request.args.get("path")
    if path and os.path.isfile(path):
        return send_file(path, mimetype='image/png')
    return "Not Found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

