import os
import argparse
import cv2
import torch
from agent_models.utils import *
from transformers import (
    CLIPModel, CLIPProcessor,
    ChineseCLIPModel, ChineseCLIPProcessor,
    CLIPSegModel, AutoProcessor
)
from glob import glob
from PIL import Image


def extract_video_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamps = []
    frame_paths = []

    os.makedirs(output_folder, exist_ok=True)
    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        timestamps.append(frame_idx / fps)
        frame_paths.append(frame_path)
        frame_idx += 1
    cap.release()
    return frame_paths, timestamps

def run_dino_image_retrieval_with_timestamp(image_folder_path, query_image, device, timestamps):
    image_folder = sorted(glob(os.path.join(image_folder_path, "*")))
    
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    index, _ = image_folder_embed(image_folder, model=model, device=device)
    
    query_vec = dino_image_query(query_image, model=model, device=device)
    
    top_paths = search_and_visualize(index, query_vec, image_folder, topk=1, visualize=False)
    top_path = top_paths[0]
    
    frame_idx = int(os.path.basename(top_path).split("_")[1].split(".")[0])
    time_in_seconds = timestamps[frame_idx]
    return top_path, time_in_seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="./datasets/example/00001.mp4")
    parser.add_argument("--image_query_path", type=str, default="./datasets/example/00001_video_frame.png")
    parser.add_argument("--text_query", type=str, default="")
    
    parser.add_argument("--text_image_retrieval", action="store_true")
    parser.add_argument("--image_image_retrieval", action="store_true")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.image_image_retrieval:
        temp_frame_dir = "./temp_frames"
        os.makedirs(temp_frame_dir, exist_ok=True)
        frame_paths, timestamps = extract_video_frames(args.video_path, temp_frame_dir)
        
        best_frame_path, time_sec = run_dino_image_retrieval_with_timestamp(
            temp_frame_dir, args.image_query_path, device, timestamps
        )
        
        print(f"\nBest matching frame: {best_frame_path}")
        print(f"Located at {time_sec:.2f} seconds in video.")

