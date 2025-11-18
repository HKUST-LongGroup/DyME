import shutil
from datasets import load_dataset
import json
import os
import re
from PIL import Image
from tqdm import tqdm


def parse_gpt_response(response_text):
    """
    Parse answer and hint from GPT response text.
    """
    hint = response_text
    match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", response_text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        answer = ""
    return answer, hint


def process_and_save_llava_cot(
        source_images_root,
        base_output_dir
):
    """
    Load the Xkev/LLaVA-CoT-100k dataset, read images from local disk,
    filter the ChartQA portion, and convert it into the desired format
    while correctly saving images and metadata.
    """
    # 1. Define output directories
    image_output_dir = os.path.join(base_output_dir, "images")
    json_output_dir = os.path.join(base_output_dir, "json")

    # 2. Create directories
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    print(f"Source image root directory: {os.path.abspath(source_images_root)}")
    print(f"Processed images will be saved to: {image_output_dir}")
    print(f"Processed JSON will be saved to: {json_output_dir}")

    # 3. Load dataset metadata
    print("Loading Xkev/LLaVA-CoT-100k dataset metadata...")
    try:
        dataset = load_dataset("Xkev/LLaVA-CoT-100k", split='train')
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 4. --- Key fix #1 ---
    #    Directly search within example['image'] (a string)
    print("Filtering samples that contain 'chartqa/train/'...")
    chartqa_dataset = dataset.filter(lambda example: 'chartqa/train/' in example['image'])
    print(f"Number of samples after filtering: {len(chartqa_dataset)}")

    # 5. Iterate over filtered dataset
    metadata_list = []
    for example in tqdm(chartqa_dataset, desc="Processing chartqa samples"):

        # --- Key fix #2 ---
        #    example['image'] is already the relative path string we need
        relative_path = example['image']
        source_image_path = os.path.join(source_images_root, relative_path)

        if not os.path.exists(source_image_path):
            print(f"\nWarning: source image not found, skipped: {source_image_path}")
            continue

        conversations = example["conversations"]
        conv_iter = iter(conversations)

        for human_conv in conv_iter:
            try:
                gpt_conv = next(conv_iter)
            except StopIteration:
                continue

            if human_conv.get("from") != "human" or gpt_conv.get("from") != "gpt":
                continue

            question = human_conv["value"]
            if ' Answer the question using a single word or phrase.' in question:
                question = question.replace(' Answer the question using a single word or phrase.', '')

            answer, hint = parse_gpt_response(gpt_conv["value"])

            if not answer:
                continue

            destination_image_path = os.path.join(image_output_dir, relative_path)
            destination_dir = os.path.dirname(destination_image_path)
            os.makedirs(destination_dir, exist_ok=True)

            if not os.path.exists(destination_image_path):
                try:
                    shutil.copy(source_image_path, destination_image_path)
                except Exception as e:
                    print(f"\nFailed to save image {destination_image_path}: {e}")
                    continue

            metadata_list.append({
                "question": question,
                "question_wo_prompt": question,
                "answer": answer,
                "hint": hint,
                "image": destination_image_path,
            })

    # 8. Write all metadata to a JSON file
    json_filename = os.path.join(json_output_dir, "chartqa_train_processed.json")
    print(f"\nSaving {len(metadata_list)} metadata entries to {json_filename}...")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=4, ensure_ascii=False)

    print(f"\n--- Processing completed! ---")
    print(f"All image files have been saved in: '{image_output_dir}'")
    print(f"All JSON files have been saved in: '{json_output_dir}'")


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None

    # --- How to run ---
    # 1. Set the path to the folder where you extracted the images in the first step
    SOURCE_IMAGES_ROOT_DIR = "/path/to/chartqa_output/llavacot/LLaVA-CoT-100k/unzipped_images"

    # 2. Set the output directory where you want to save the processed data
    OUTPUT_DIR = "/path/to/data/chartqa_output/llavacot"

    # 3. Call the main function
    process_and_save_llava_cot(
        source_images_root=SOURCE_IMAGES_ROOT_DIR,
        base_output_dir=OUTPUT_DIR
    )
