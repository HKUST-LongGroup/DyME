from datasets import load_dataset
import json
import os


def save_gsm8k_as_json(base_output_dir="/gsm8k_output"):
    """
    Load the openai/gsm8k-style dataset.
    Extract the 'question' and 'answer' fields,
    and generate a JSON metadata file for each split (train, test).
    """

    # 1. Define output directory
    # gsm8k is plain text, so we only need a json directory
    base_dir_abs = os.path.abspath(base_output_dir)
    json_output_dir = '/path/to/data/lm_math/json_new'

    # 2. Create directory
    os.makedirs(json_output_dir, exist_ok=True)

    print(f"JSON will be saved to (absolute path): {json_output_dir}")

    # 3. Load dataset
    print("Loading ankner/gsm8k-CoT dataset (main config)...")
    try:
        # 'main' config is the standard configuration for gsm8k
        dataset = load_dataset("ankner/gsm8k-CoT")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Dataset splits: {list(dataset.keys())}")

    # 4. Iterate over each split (e.g., 'train', 'test')
    for split in dataset.keys():
        print(f"\n--- Processing split {split} ---")

        metadata_list = []
        total_count = len(dataset[split])

        # 5. Iterate over each sample in the dataset
        for i, example in enumerate(dataset[split]):

            # Extract metadata
            question = example["question"]
            hint = example["response"]
            pure_answer = example["answer"]

            # 6. (No image step)
            # gsm8k does not contain images, so we do not need to save images or generate paths.
            # We directly add the text data to the list.

            # 7. Add metadata to the list
            metadata_list.append({
                "question": question,
                "answer": pure_answer,
                "hint": hint,
            })

            # Print progress
            if (i + 1) % 1000 == 0 or (i + 1) == total_count:
                print(f"  Processed {split} split: {i + 1} / {total_count}")

        # 8. Write all metadata for this split to a JSON file
        json_filename = os.path.join(json_output_dir, f"{split}.json")
        print(f"Saving {len(metadata_list)} metadata entries to {json_filename}...")

        with open(json_filename, 'w', encoding='utf-8') as f:
            # indent=2 (or 4) makes the JSON file more readable
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"\n--- Processing completed! ---")
    print(f"All JSON files have been saved in: '{json_output_dir}'")


if __name__ == "__main__":
    # gsm8k is plain text, no need for PIL.Image.MAX_IMAGE_PIXELS
    save_gsm8k_as_json()
