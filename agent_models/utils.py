import os
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import supervision as sv

import torch
import torchvision.transforms as T


def get_image_files(folder: str, exts={".jpg", ".jpeg", ".png", ".bmp", ".webp"}) -> list:
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[-1].lower() in exts
    ]


def get_transform(size=224):
    return torch.nn.Sequential(
        torch.nn.Upsample(size=(size, size), mode='bilinear', align_corners=False),
        T.Normalize([0.5], [0.5])
    )


def load_image_tensor(img_path: str, transform, device) -> torch.Tensor:
    # DinoV2 suggest 224x224 image input
    image = Image.open(img_path).convert("RGB").resize((224, 224))
    image_tensor = T.ToTensor()(image).unsqueeze(0).to(device)
    return transform(image_tensor)


def image_folder_embed(files: list, model, device, is_clip=False, is_blip=False, processor=None):
    if is_blip:
        dim = 1408
    elif is_clip:
        dim = 512
    else:
        dim = 384   # Dinov2
    
    index = faiss.IndexFlatL2(dim)
    image_folder_feature_dict = {}
    transform = get_transform()

    with torch.no_grad():
        for file in tqdm(files, desc="Extracting embeddings"):
            try:
                image = Image.open(file).convert("RGB")

                if is_blip:
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    outputs = model.vision_model(**inputs)
                    features = outputs.pooler_output
                
                elif is_clip:
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    features = model.get_image_features(**inputs)
                
                else:
                    img_tensor = load_image_tensor(file, transform, device)
                    features = model(img_tensor)

                features = features / features.norm(dim=-1, keepdim=True)
                features_np = features.cpu().numpy().astype("float32").reshape(1, -1)
                index.add(features_np)

                file_basename = os.path.basename(file)
                image_folder_feature_dict[file_basename] = features_np.tolist()

            except Exception as e:
                print(f"Error processing {file}: {e}")

    return index, image_folder_feature_dict


# query
def dino_image_query(query_path, model, device, is_clip=False, processor=None):
    transform = get_transform()
    with torch.no_grad():
        if is_clip:
            inputs = processor(images=Image.open(query_path).convert("RGB"), return_tensors="pt").to(device)
            features = model.get_image_features(**inputs)
        else:
            img_tensor = load_image_tensor(query_path, transform, device)
            features = model(img_tensor)

        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32").reshape(1, -1)


def clip_text_query(text, model, device, processor):
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt").to(device)
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32").reshape(1, -1)


# def search_and_visualize(index, query_vector, files, topk=5, visualize=False, title="Query"):
#     D, I = index.search(query_vector, topk)
#     result_paths = [files[i] for i in I[0]]

#     if visualize:
#         print(f"{title}:")
#         print("=" * 40)
#         for i, path in enumerate(result_paths):
#             print(f"Top {i+1} Match: {os.path.basename(path)}")
#             img = cv2.resize(cv2.imread(path), (416, 416))
#             sv.plot_image(img, size=(6, 6))

#     return result_paths


def search_and_visualize(index, query_vector, files, topk=5, visualize=False, title="Query"):
    D, I = index.search(query_vector, topk)
    result_paths = [files[i] for i in I[0]]
    distances = D[0]

    if visualize:
        print(f"{title}:")
        print("=" * 40)
        for i, (path, dist) in enumerate(zip(result_paths, distances)):
            print(f"Top {i+1} Match: {os.path.basename(path)} | Distance: {dist:.4f}")
            img = cv2.resize(cv2.imread(path), (416, 416))
            sv.plot_image(img, size=(6, 6))

    return result_paths, distances


# clip finetuning
def image_folder_embed_ft(files: list, model, device, preprocess):
    dim = 768  # if ViT-L/14
    # dim = 512 # if ViT-B/32
    index = faiss.IndexFlatL2(dim)
    image_folder_feature_dict = {}

    with torch.no_grad():
        for file in tqdm(files, desc="Extracting embeddings (Fine-tuned CLIP)"):
            try:
                image = Image.open(file).convert("RGB")
                image = preprocess(image).unsqueeze(0).to(device)
                features = model.encode_image(image)
                features = features / features.norm(dim=-1, keepdim=True)
                features_np = features.cpu().numpy().astype("float32").reshape(1, -1)
                index.add(features_np)

                file_basename = os.path.basename(file)
                image_folder_feature_dict[file_basename] = features_np.tolist()

            except Exception as e:
                print(f"Error processing {file}: {e}")

    return index, image_folder_feature_dict

def clip_text_query_ft(text, model, device):
    import clip
    
    with torch.no_grad():
        text_tokens = clip.tokenize([text]).to(device)
        features = model.encode_text(text_tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32").reshape(1, -1)