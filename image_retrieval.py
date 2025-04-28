import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'

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


def run_clip_retrieval(image_folder, query, device, topk=5):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    index, _ = image_folder_embed(image_folder, model, device, is_clip=True, processor=processor)
    query_vec = clip_text_query(query, model, device, processor)
    return search_and_visualize(index, query_vec, image_folder, topk=topk, visualize=False)

# finetuning
def run_finetuned_clip_retrieval(image_folder, query, model_path, device, topk=5):
    import clip # open clip
    
    model = torch.load(model_path, map_location=device).eval()
    preprocess = clip.load("ViT-L/14", device=device)[1]
    index, _ = image_folder_embed_ft(image_folder, model, device, preprocess)
    query_vec = clip_text_query_ft(query, model, device)
    return search_and_visualize(index, query_vec, image_folder, topk=topk, visualize=False)


def run_chinese_clip_retrieval(image_folder, query, device, topk=5):
    model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(device).eval()
    processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    index, _ = image_folder_embed(image_folder, model, device, is_clip=True, processor=processor)
    query_vec = clip_text_query(query, model, device, processor)
    return search_and_visualize(index, query_vec, image_folder, topk=topk, visualize=False)


def run_clipseg_retrieval(image_folder, query, device, topk=5):
    model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    index, _ = image_folder_embed(image_folder, model, device, is_clip=True, processor=processor)
    query_vec = clip_text_query(query, model, device, processor)
    return search_and_visualize(index, query_vec, image_folder, topk=topk, visualize=False)


def run_dino_image_retrieval(image_folder, query_image, device, topk=5):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    index, _ = image_folder_embed(image_folder, model=model, device=device)
    query_vec = dino_image_query(query_image, model=model, device=device)
    return search_and_visualize(index, query_vec, image_folder, topk=topk, visualize=False)


def text_image_retrieval(image_folder, text_query, device):
    results = {}
    print("\n[CLIP Retrieval]")
    results["clip"] = run_clip_retrieval(image_folder, text_query, device, topk=8)

    # print("\n[Chinese-CLIP Retrieval]")
    # results["chinese_clip"] = run_chinese_clip_retrieval(image_folder, text_query, device)

    print("\n[CLIPSeg Retrieval]")
    results["clipseg"] = run_clipseg_retrieval(image_folder, text_query, device, topk=8)
    
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
    
    return sorted_results, results


def image_image_retrieval(image_folder, image_query_path, device):
    results = {}
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
    
    return sorted_results, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder_path", type=str, default="./datasets/DiffusionDB")
    parser.add_argument("--image_query_path", type=str, default="./datasets/example/00001.png")
    parser.add_argument("--text_query", type=str, default="The guy with yellow cloth")
    parser.add_argument("--text_image_retrieval", action="store_true")
    parser.add_argument("--image_image_retrieval", action="store_true")
    # vision-language model
    parser.add_argument("--vlm_verifer", action="store_true")
    parser.add_argument("--vlm_model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")  # model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_folder = get_image_files(args.image_folder_path)

    args.text_image_retrieval = True    # if debug
    if args.text_image_retrieval:
        # finetuning model
        # result_paths, distances = run_finetuned_clip_retrieval(image_folder, args.text_query, model_path="./datasets/finetuning_results/ft-checkpoints/clip_ft_epoch_5.pt")
        # print(result_paths)
        
        # The text_image_retrieval function currently does not integrate our fine-tuned models. 
        # However, you can refer to the implementation above to incorporate them. Our trained models are available here for direct download:
        # https://drive.google.com/drive/folders/1gzbKjAjS8ED1GMFFlfEiSXeGuzlCycl_?usp=drive_link
        sorted_results, results = text_image_retrieval(image_folder, args.text_query, device)

    if args.image_image_retrieval and args.image_query_path:
        sorted_results, results = image_image_retrieval(image_folder, args.image_query_path, device)
    
    # vlm verifier
    image_list = [path for path, (dist, model_name) in sorted_results]
    if args.vlm_verifer:
        image_list = [path for path, (dist, model_name) in sorted_results]
        system_prompt = (
            "You are a helpful assistant that can understand visual content. "
            "You will be shown several image frames. Your task is to identify which image best matches the given text query. "
            "Only output the frame ID of the best matching image. Do not explain your reasoning."
        )

        user_prompt = (
            "Please examine all the provided images and select the one that best corresponds to the following text.\n\n"
            "Example:\n"
            "Images: <frame 0>, <frame 1>, <frame 2>\n"
            "Text query: 'A cat sitting on a sofa'\n"
            "Answer: <frame 1>\n\n"
            "Now for the following input:"
        )
        
        prefer_frame = qwen_vlm_verifier(system_prompt, user_prompt, args.text_query, image_list, model_id=args.vlm_model_id)
        
        # user_preference_image_path
        print(prefer_frame)