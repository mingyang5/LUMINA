import os
import json
from tqdm import tqdm

import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = scheduler
pipe.to("cuda")

json_path = "./generation_prompts.json"
save_dir = "./generated_images" 
os.makedirs(save_dir, exist_ok=True)

with open(json_path, "r") as f:
    prompts_data = json.load(f)
        
for idx, item in enumerate(tqdm(prompts_data)):
    prompt = item["prompt"]
    prompt_id = item["id"]
    
    print(prompt)
    
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0,
        num_inference_steps=20
        ).frames[0]
    
    filename = f"{idx:03d}_{prompt_id}_wan.mp4"
    save_path = os.path.join(save_dir, filename)
    
    export_to_video(output, save_path, fps=16)