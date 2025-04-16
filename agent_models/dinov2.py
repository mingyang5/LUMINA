import torch
import torchvision.transforms as T

from utils import *


if __name__ == "__main__":
    # Dinov2 image-image retrieval
    image_folder_path = "/mnt/data/mingyang/courses/csce670/CSCE-670-Agentic-Retrieval/datasets/stable_diffusion"
    image_query_path = "/mnt/data/mingyang/courses/csce670/CSCE-670-Agentic-Retrieval/datasets/example/000_00001.png"
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    
    image_folder = get_image_files(image_folder_path)
    index, image_folder_feature_dict = image_folder_embed(image_folder, model=dinov2, device=device)

    query_vec = dino_image_query(image_query_path, model=dinov2, device=device)
    
    retrieval_paths, _ = search_and_visualize(index, query_vec, image_folder, topk=5)
    print(retrieval_paths)
    
