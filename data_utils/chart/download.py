from datasets import load_dataset
import json
import os
from PIL import Image


def save_chartqa_with_absolute_paths(base_output_dir="/path/to/chartqa_output"):
    """
    Load the HuggingFaceM4/ChartQA dataset.
    For each PIL image, generate a unique filename, save the image locally,
    and create a JSON metadata file that contains *absolute image paths*.
    """

    # 1. Define output directories
    # Use os.path.abspath() to get the full absolute path
    base_dir_abs = os.path.abspath(base_output_dir)
    image_output_dir = os.path.join(base_dir_abs, "images")
    json_output_dir = os.path.join(base_dir_abs, "json")

    # 2. Create directories
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    print(f"Images will be saved to (absolute path): {image_output_dir}")
    print(f"JSON will be saved to (absolute path): {json_output_dir}")

    # 3. Load dataset
    print("Loading HuggingFaceM4/ChartQA dataset (this may take some time)...")
    try:
        # trust_remote_code=True is required for ChartQA
        dataset = load_dataset("HuggingFaceM4/ChartQA")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Dataset splits: {list(dataset.keys())}")

    # 4. Iterate over each split (e.g., 'train', 'test')
    for split in dataset.keys():
        print(f"\n--- Processing split {split} ---")

        metadata_list = []
        total_count = len(dataset[split])

        # 5. Use enumerate to get index (i) for generating unique filenames
        for i, example in enumerate(dataset[split]):

            # Extract metadata
            question = example["query"]
            answer = example["label"][0]

            # Extract PIL image object
            pil_image = example["image"]

            # --- Key step ---
            # (A) Generate a unique filename
            #     Use 6-digit zero padding (e.g., 000001) to keep files sorted
            generated_filename = f"{split}_{i:06d}.png"

            # (B) Build the *absolute* save path
            image_save_path = os.path.join(image_output_dir, generated_filename)

            # 6. Save image to this absolute path
            # Check if the image already exists to avoid duplicate saving
            if not os.path.exists(image_save_path):
                try:
                    # Save the PIL image locally
                    pil_image.save(image_save_path)
                except Exception as e:
                    print(f"Failed to save image {image_save_path}: {e}")
                    continue

                    # 7. Add metadata with the *absolute path* to the list
            #    We use the key "image_path" to make it clear this is a path
            metadata_list.append({
                "question": question,
                "question_wo_prompt": question,
                "answer": answer,
                "image": image_save_path,
                "human_or_machine": example.get("human_or_machine", 0)
            })

            # Print progress
            if (i + 1) % 500 == 0 or (i + 1) == total_count:
                print(f"  Processed {split} split: {i + 1} / {total_count}")

        # 8. Write all metadata for this split to a JSON file
        json_filename = os.path.join(json_output_dir, f"{split}.json")
        print(f"Saving {len(metadata_list)} metadata entries to {json_filename}...")

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)

    print(f"\n--- Processing completed! ---")
    print(f"All image files have been saved in: '{image_output_dir}'")
    print(f"All JSON files have been saved in: '{json_output_dir}'")


if __name__ == "__main__":
    # Ensure PIL can handle potentially large images
    Image.MAX_IMAGE_PIXELS = None

    save_chartqa_with_absolute_paths()
