import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import argparse
import json
from glob import glob

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_message(system_content, user_content, image_caption):
    user_text = f"{user_content}\n\n### Caption: {image_caption}\n### Output:"
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
    for image_id, image_caption in tqdm(image_captions.items()):
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

        # word_tag post-process/filter rules
        try:
            word_tag_dict = eval(response_word_tag.strip())
            assert isinstance(word_tag_dict, dict)
        except Exception:
            continue

        # Check that all words are substrings of the caption
        all_words = []
        for key in ['background', 'subject', 'object']:
            if key in word_tag_dict and isinstance(word_tag_dict[key], list):
                all_words.extend(word_tag_dict[key])

        if any(word and word not in image_caption for word in all_words):
            continue

        # Reclassify background entities: e.g., "cloud" should move to background
        background_keywords = ['cloud', 'fog', 'mist', 'sky', 'horizon', 'field', 'ocean', 'sea', 'grass', 'ground', 'snow', 'sand', 'desert']
        new_background = set(word_tag_dict.get('background', []))
        new_subject = set(word_tag_dict.get('subject', []))
        new_object = set(word_tag_dict.get('object', []))

        for obj in list(new_object):
            if any(bg_word in obj for bg_word in background_keywords):
                new_object.discard(obj)
                new_background.add(obj)

        word_tag_dict['background'] = list(new_background)
        word_tag_dict['subject'] = list(new_subject)
        word_tag_dict['object'] = list(new_object)

        # Remove samples without subject
        # if not word_tag_dict['subject'] or all(word == '' for word in word_tag_dict['subject']):
        #     continue
        
        # Remove samples with plural subjects
        plural_subject = any(subj.endswith('s') and not subj.endswith('ss') for subj in word_tag_dict['subject'])
        if plural_subject:
            continue
        
        image_caption_word_tag[image_id] = word_tag_dict

    return image_caption_word_tag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aigc_image_caption_json", type=str, default="./datasets/example/diffusiondb_part-000001.json")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--save_word_tag_json", type=str, default="./datasets/example/diffusiondb_word_tag.json")
    args = parser.parse_args()

    system_prompt = """
        You extract entity words from image captions. For each caption, identify:
        - 'background': static spaces appearing in image (e.g., room, wall, street).
        - 'subject': humans or animals that appear in the image.
        - 'object': concrete, visually separable items (e.g., tools, clothing, accessories) that can be segmented individually using vision models such as SAM. 
        Do not include abstract concepts, emotions, facial expressions, or actions (e.g., exclude 'smile', 'happy', 'waving hands', 'conversation').

        Rules:
        1. Entity words must be nouns without quantifiers.
        2. Each entity word must be an exact substring of the caption.

        Output format:
        {'background': [...], 'subject': [...], 'object': [...]}.
        Use empty strings ('') in lists if no entity is found.
        Do not explain or add extra text.
    """
    user_prompt = """
        Given a image caption, please retrieve the entity words that indicate background, subject, and visually separable objects.

        [Definition of background] static background elements that appear in most of the image and are not meant to be segmented individually. These typically include locations, environments, or large surfaces (e.g., 'room', 'wall', 'street').
        [Definition of subject] human or animal subjects that appear in the image.
        [Definition of object] visually separable, concrete items (e.g., tools, accessories, clothing, carried objects) that can be individually segmented in the image. Do not include abstract concepts, facial expressions, emotions, or actions (e.g., exclude 'conversation', 'smile', 'serious expression').

        All entity words must follow two strict rules:
        1) The entity word is a noun without any quantifier.
        2) The entity word is an exact subset of the caption. Do not modify any characters, words, or symbols.

        Here are some examples, follow this format to output the results:

        ### Caption: A woman in a mask and coat, with long brown hair, shows a small green-capped bottle to the camera.
        ### Output: {'background': [''], 'subject': ['woman'], 'object': ['mask', 'coat', 'long brown hair', 'green-capped bottle']}
    """
    
    # with open(args.aigc_image_caption_json, "r") as f:
    #     data = json.load(f)
    # image_captions = {image_id: item['p'] for image_id, item in data.items()}
    
    image_captions = {}
    for aigc_image_caption_json in sorted(glob(os.path.join("datasets/example", "*.json"))):
        with open(aigc_image_caption_json, "r") as f:
            data = json.load(f)
        image_captions.update({image_id: item['p'] for image_id, item in data.items()})

    image_caption_word_tag = qwen_word_tag(system_prompt, user_prompt, image_captions, model_id=args.model_id)
    
    final_word_tag = {}
    for image_id, tags in image_caption_word_tag.items():
        final_word_tag[image_id] = {
            'subject': tags.get('subject', []),
            'object': tags.get('object', [])
        }

    with open(args.save_word_tag_json, "w") as f:
        json.dump(final_word_tag, f, indent=2)
    print(f"Saved word tags (subject+object only) to {args.save_word_tag_json}")
