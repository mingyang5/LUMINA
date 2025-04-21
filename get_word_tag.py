import os
import base64
from glob import glob

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def create_message(system_content, user_content, text_query, image_files):
    message = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": []}
    ]
    
    for img in image_files:
        message[1]["content"].append({"type": "image", "image": img})
    
    user_text = f"{user_content}\nImages: " + ", ".join([f"<frame {i}>" for i in range(len(image_files))]) + f"\nText query: '{text_query}'"
    message[1]["content"].append({"type": "text", "text": user_text})
    
    return message


def qwen_word_tag(system_prompt, user_prompt, text_query, image_files, model_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda"):
    # recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",    
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    messages = create_message(system_prompt, user_prompt, text_query, image_files)
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text


if __name__ == "__main__":
    print("result")