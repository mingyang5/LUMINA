import os
import json
from glob import glob

def process_json_files(json_files, save_path):
    annotations = {}

    for json_file in sorted(json_files):
        print(f"Processing {json_file}...")
        with open(json_file, "r") as f:
            data = json.load(f)

        for image_name, info in data.items():
            prompt = info.get('p', '').strip()
            if prompt:
                annotations[image_name] = [prompt]

    with open(save_path, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Saved processed annotations to {save_path}")


if __name__ == "__main__":
    input_json_files = [
        "/mnt/data/mingyang/courses/csce670/CSCE-670-Agentic-Retrieval/datasets/example/diffusiondb_part-000001.json",
        "/mnt/data/mingyang/courses/csce670/CSCE-670-Agentic-Retrieval/datasets/example/diffusiondb_part-000002.json"
    ]

    output_annotations_path = "./aigc_diffusiondb_labels.json"

    process_json_files(input_json_files, output_annotations_path)
