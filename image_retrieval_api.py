from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import torch
from image_retrieval import run_dino_image_retrieval, get_image_files

from flask_cors import CORS

remote_server_ip = ""       # 10.xx.xx.xx

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "/mnt/data/mingyang/courses/csce670/CSCE-670-Agentic-Retrieval/uploads"
DATASET_FOLDER = "/mnt/data/mingyang/courses/csce670/CSCE-670-Agentic-Retrieval/datasets/stable_diffusion"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.route('/api/image_image_retrieval', methods=['POST'])    #  POST 类型的 API 接口，http://<服务器IP>:<端口>/api/image_retrieval -> 
def image_image_retrieval():
    if 'query_image' not in request.files:
        return jsonify({'error': 'Missing file'}), 400

    uploaded_file = request.files['query_image']
    filename = secure_filename(uploaded_file.filename)
    query_image_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(query_image_path)

    print(f"[INFO] Received query image: {query_image_path}")

    image_list = get_image_files(DATASET_FOLDER)
    paths, dists = run_dino_image_retrieval(image_list, query_image_path, device)

    print(f"[INFO] Retrieval results: {[os.path.basename(p) for p in paths]}")
    
    results = [
        {
            'path': f"http://{remote_server_ip}:5001/static_dataset/{os.path.basename(p)}",
            'distance': float(d)
        } for p, d in zip(paths, dists)
    ]
    
    print(results)
    
    return jsonify({'results': results})


@app.route('/static_dataset/<path:filename>')               # 访问静态图像的接口 -> 取出文件
def serve_static_dataset(filename):
    full_path = os.path.join(DATASET_FOLDER, filename)
    if not os.path.exists(full_path):
        print(f"[ERROR] File not found: {full_path}")
    else:
        print(f"[INFO] Serving file: {full_path}")
    return send_from_directory(DATASET_FOLDER, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
