import os
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import supervision as sv

import torch
import torchvision.transforms as T
from transformers import AutoProcessor, CLIPSegModel

from utils import *


if __name__ == "__main__":
    # clip text-image retrieval
    image_folder_path = "/mnt/data/mingyang/courses/csce670/CSCE-670-Agentic-Retrieval/datasets/stable_diffusion"
    # image_query_path = "/mnt/data/mingyang/courses/csce670/CSCE-670-Agentic-Retrieval/datasets/example/000_00001.png"
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    chinese_clip_model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()
    chinese_clip_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    image_folder = get_image_files(image_folder_path)
    index, image_folder_feature_dict = image_folder_embed(image_folder, chinese_clip_model, device, is_clip=True, processor=chinese_clip_processor)
    
    text_query = "The guy with a camera"
    query_vec = clip_text_query(text_query, chinese_clip_model, device, chinese_clip_processor)
    
    retrieval_paths, _ = search_and_visualize(index, query_vec, image_folder, topk=5)
    print(retrieval_paths)