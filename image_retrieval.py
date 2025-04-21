import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import argparse
from PIL import Image

import torch
from transformers import (
    CLIPModel, CLIPProcessor,
    ChineseCLIPModel, ChineseCLIPProcessor,
    CLIPSegModel, AutoProcessor
)

from agent_models.utils import *
from vlm_verifier import *

def run_clip_retrieval(image_folder, query, device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    index, _ = image_folder_embed(image_folder, model, device, is_clip=True, processor=processor)
    query_vec = clip_text_query(query, model, device, processor)
    return search_and_visualize(index, query_vec, image_folder, topk=5, visualize=False)

def run_chinese_clip_retrieval(image_folder, query, device):
    model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(device).eval()
    processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    index, _ = image_folder_embed(image_folder, model, device, is_clip=True, processor=processor)
    query_vec = clip_text_query(query, model, device, processor)
    return search_and_visualize(index, query_vec, image_folder, topk=5, visualize=False)

def run_clipseg_retrieval(image_folder, query, device):
    model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    index, _ = image_folder_embed(image_folder, model, device, is_clip=True, processor=processor)
    query_vec = clip_text_query(query, model, device, processor)
    return search_and_visualize(index, query_vec, image_folder, topk=5, visualize=False)

def run_dino_image_retrieval(image_folder, query_image, device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    index, _ = image_folder_embed(image_folder, model=model, device=device)
    query_vec = dino_image_query(query_image, model=model, device=device)
    return search_and_visualize(index, query_vec, image_folder, topk=5, visualize=False)

def text_image_retrieval(image_folder, text_query, device):
    print("\n[CLIP Retrieval]")
    results["clip"] = run_clip_retrieval(image_folder, text_query, device)

    print("\n[Chinese-CLIP Retrieval]")
    results["chinese_clip"] = run_chinese_clip_retrieval(image_folder, text_query, device)

    print("\n[CLIPSeg Retrieval]")
    results["clipseg"] = run_clipseg_retrieval(image_folder, text_query, device)
    
    all_results = []
    for model_name, (paths, dists) in results.items():
        # print(f"\n{model_name.upper()} Top-5:")
        for path, dist in zip(paths, dists):
            # print(f" - {path} | Distance: {dist:.4f}")
            all_results.append((path, dist, model_name))
            
    unique_results = {}
    for path, dist, model_name in all_results:
        if path not in unique_results or dist < unique_results[path][0]:
            unique_results[path] = (dist, model_name)
            
    sorted_results = sorted(unique_results.items(), key=lambda x: x[1][0])
    for i, (path, (dist, model_name)) in enumerate(sorted_results):
        print(f"{i+1}. [{model_name.upper()}] {path} | Distance: {dist:.4f}")
    
    return sorted_results

def image_image_retrieval(image_folder, image_query_path, device):
    print("\n[DINOv2 Retrieval]")
    results["dino"] = run_dino_image_retrieval(image_folder, image_query_path, device)

    all_results = []
    for model_name, (paths, dists) in results.items():
        # print(f"\n{model_name.upper()} Top-5:")
        for path, dist in zip(paths, dists):
            # print(f" - {path} | Distance: {dist:.4f}")
            all_results.append((path, dist, model_name))
            
    unique_results = {}
    for path, dist, model_name in all_results:
        if path not in unique_results or dist < unique_results[path][0]:
            unique_results[path] = (dist, model_name)
            
    sorted_results = sorted(unique_results.items(), key=lambda x: x[1][0])
    for i, (path, (dist, model_name)) in enumerate(sorted_results):
        print(f"{i+1}. [{model_name.upper()}] {path} | Distance: {dist:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder_path", type=str, default="./datasets/stable_diffusion")
    parser.add_argument("--image_query_path", type=str, default="./datasets/example/00001.png")
    parser.add_argument("--text_query", type=str, default="The guy with a camera")
    
    parser.add_argument("--text_image_retrieval", action="store_true")
    parser.add_argument("--image_image_retrieval", action="store_true")
    
    # vision-language model
    parser.add_argument("--vlm_model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")  # model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_folder = get_image_files(args.image_folder_path)

    results = {}
    args.text_image_retrieval = True    # if debug
    if args.text_image_retrieval:
        sorted_results = text_image_retrieval(image_folder, args.text_query, device)

    if args.image_image_retrieval and args.image_query_path:
        sorted_results = image_image_retrieval(image_folder, args.image_query_path, device)

    image_list = [path for path, (dist, model_name) in sorted_results]
    