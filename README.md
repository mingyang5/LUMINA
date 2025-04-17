# Incorporating Multi-Modality and Latent Representations for Image Retrieval

**CSCE 670 — Information Storage & Retrieval**  
*Course Project*

**Team Members:**  
- Yufeng Yang  
- Mingyang Wu  
- Yu-Hsuan Ho  
- Yiming Xiao



## Environment

```bash
conda create -n retrieval python=3.10 -y
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c pytorch -c nvidia faiss-gpu

pip install opencv-python supervision xformers

# VLM inference
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]
pip install -U flash-attn --no-build-isolation
```

## Retrieval Models

### Supported Text-Image Retrieval Models

- [x] CLIP
- [x] ChineseCLIP
- [x] CLIPSeg

### Supported Image-Image Retrieval Models

- [x] DinoV2


## Vision-Language Model verifers

- [x] QWen2.5-VL


## AIGC image Data Preparation

- [x] Qwen2.5 for word_tag


## Implementation

```bash
# text-image retrieval
CUDA_VISIBLE_DEVICES=1 python image_retrieval.py --text_image_retrieval
```