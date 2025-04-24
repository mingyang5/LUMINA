# run `pip install git+https://github.com/huggingface/diffusers` before using Sana with diffusers
import torch
from diffusers import SanaPipeline

import os
import json
from tqdm import tqdm

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    variant="bf16",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)

if pipe.transformer.config.sample_size == 128:
    pipe.vae.enable_tiling(
        tile_sample_min_height=1024,
        tile_sample_min_width=1024,
        tile_sample_stride_height=896,
        tile_sample_stride_width=896,
    )

# GenAI-Bench
json_path = "./generation_prompts.json"
save_dir = "./generated_images" 
os.makedirs(save_dir, exist_ok=True)

with open(json_path, "r") as f:
    prompts_data = json.load(f)

for idx, item in enumerate(tqdm(prompts_data)):
    prompt = item["prompt"]
    prompt_id = item["id"]
    
    print(prompt)
    
    image = pipe( 
        prompt=prompt,
        height=512,
        width=512,
        guidance_scale=5,
        num_inference_steps=20,
        generator=torch.Generator(device="cuda").manual_seed(1024),
    )[0]

    words = prompt.split()
    short_prompt = "_".join(words[:4]) if len(words) >= 4 else "_".join(words)
    short_prompt = short_prompt.replace('/', '_').replace('\\', '_')
    
    filename = f"{idx:03d}_{prompt_id}_sana.png"
    save_path = os.path.join(save_dir, filename)

    image[0].save(save_path)
