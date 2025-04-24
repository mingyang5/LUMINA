import os
# os.environ['CUDA_VISIBLE_DEVICES']='4'

import json
from tqdm import tqdm

import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# GenAI-Bench
json_path = "./generation_prompts.json"
save_dir = "./generated_images" 

os.makedirs(save_dir, exist_ok=True)

with open(json_path, "r") as f:
    prompts_data = json.load(f)

for idx, item in enumerate(tqdm(prompts_data)):
    prompt = item["prompt"]
    prompt_id = item["id"]
    
    filename = f"{idx:03d}_{prompt_id}_sd3.png"
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path):
        continue
    
    print(prompt)
    image = pipe(
        prompt,
        height=512,
        width=512,
        guidance_scale=5,
        num_inference_steps=20,
    ).images[0]

    image.save(save_path)
