import os
# os.environ['CUDA_VISIBLE_DEVICES']='7'

import json
from glob import glob

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_message(system_content, user_content, image_caption):
    user_text = f"{user_content}\n" + f"\n### Caption: {image_caption}\n### Output:" 
    
    message = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_text}
    ]
    
    return message


def qwen_word_tag(system_prompt, user_prompt, image_captions, model_id="Qwen/Qwen2.5-3B-Instruct", device="cuda"):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    image_caption_word_tag = {}
    for image_id, image_caption in  tqdm(image_captions.items()):
        # print(image_id, image_caption)
        
        messages = create_message(system_prompt, user_prompt, image_caption)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response_word_tag = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response_word_tag)
        image_caption_word_tag[image_id] = response_word_tag
    
    return image_caption_word_tag


if __name__ == "__main__":
    # print("result")
    aigc_image_caption_json = "datasets/diffusiondb/part-000001.json"
    with open(aigc_image_caption_json, "r") as f:
        data = json.load(f)
        
    image_captions = {}
    for image_id, item in data.items():
        image_captions[image_id] = item['p']
    
    system_prompt = """
        You extract entity words from image captions. For each caption, identify:\n
        - 'background': static spaces appearing in image (e.g., room, wall, street).\n
        - 'subject': humans or animals that appear in the image.\n
        - 'object': concrete, visually separable items (e.g., tools, clothing, accessories) that can be segmented individually using vision models such as SAM. 
        Do not include abstract concepts, emotions, facial expressions, or actions (e.g., exclude 'smile', 'happy', 'waving hands', 'conversation').\n\n
        Rules:\n
        1. Entity words must be nouns without quantifiers.\n
        2. Each entity word must be an exact substring of the caption.\n\n
        Output format:\n{'background': [...], 'subject': [...], 'object': [...]}.\nUse empty strings ('') in lists if no entity is found.\nDo not explain or add extra text.
    """
    
    user_prompt = """
        Given a image caption, please retrieve the entity words that indicate background, subject, and visually separable objects.\n\n
        [Definition of background] static background elements that appear in most of the image and are not meant to be segmented individually. These typically include locations, environments, or large surfaces (e.g., 'room', 'wall', 'street').\n\n[Definition of subject] human or animal subjects that appear in the image.\n\n
        [Definition of object] visually separable, concrete items (e.g., tools, accessories, clothing, carried objects) that can be individually segmented in the image. Do not include abstract concepts, facial expressions, emotions, or actions (e.g., exclude 'conversation', 'smile', 'serious expression').\n\n
        All entity words must follow two strict rules:\n
            1) The entity word is a noun without any quantifier.\n
            2) The entity word is an exact subset of the caption. Do not modify any characters, words, or symbols.\n\n
        Here are some examples, follow this format to output the results:\n\n
        ### Caption: A woman in a mask and coat, with long brown hair, shows a small green-capped bottle to the camera.\n
        ### Output: {'background': [''], 'subject': ['woman'], 'object': ['mask', 'coat', 'long brown hair', 'green-capped bottle']}
    """
    
    image_caption_word_tag = qwen_word_tag(system_prompt, user_prompt, image_captions)
    save_word_tag_json = "./datasets/diffusiondb_word_tag.json"
    with open(save_word_tag_json, "w") as f:
        json.dump(image_caption_word_tag, f, indent=2)
    print(f"Saved word tags to {save_word_tag_json}")