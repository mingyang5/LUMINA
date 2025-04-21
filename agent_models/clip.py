import os
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import supervision as sv

import torch
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel

from utils import *


if __name__ == "__main__":
    # clip text-image retrieval
    image_folder_path = "/mnt/data/mingyang/courses/csce670/CSCE-670-Agentic-Retrieval/datasets/stable_diffusion"
    # image_query_path = "/mnt/data/mingyang/courses/csce670/CSCE-670-Agentic-Retrieval/datasets/example/000_00001.png"
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_folder = get_image_files(image_folder_path)
    index, image_folder_feature_dict = image_folder_embed(image_folder, clip_model, device, is_clip=True, processor=clip_processor)
    
    text_query = "The guy with a camera"
    query_vec = clip_text_query(text_query, clip_model, device, clip_processor)
    
    retrieval_paths, _ = search_and_visualize(index, query_vec, image_folder, topk=5)