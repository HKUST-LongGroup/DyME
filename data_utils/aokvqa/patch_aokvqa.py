import os
import json
from datasets import load_dataset
from tqdm import tqdm
import shutil


def patch_json_with_direct_answers(base_output_dir):
    """
    Load existing JSON files and the original A-OKVQA dataset,
    and add the 'direct_answers' field back into the JSON files.

    This script assumes that the entries in your generated JSON files
    (e.g., train.json) are ordered according to the original dataset indices.
    Your original script's
    `metadata_list.sort(key=lambda x: x['image'])` ensures this alignment.
    """

    json_output_dir = os.path.join(base_output_dir, "json")
    print(f"JSON directory to patch: {json_output_dir}")

    # 1. Load the original dataset
    print("Loading HuggingFaceM4/A-OKVQA dataset (metadata only)...")
    try:
        # Specify a cache directory to avoid repeated downloads
        cache_dir = os.path.join(base_output_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        dataset = load_dataset("HuggingFaceM4/A-OKVQA", cache_dir=cache_dir)
    except Exception as e:
        print(f"Failed to load original dataset: {e}")
        return

    print(f"Available splits in dataset: {list(dataset.keys())}")

    # 2. Iterate over each split
    for split in dataset.keys():
        json_filename = os.path.join(json_output_dir, f"{split}.json")

        if not os.path.exists(json_filename):
            print(f"!! Warning: {json_filename} not found, skipping split '{split}'.")
            continue

        print(f"\n--- Patching split {split} ({json_filename}) ---")

        # 3. Load existing (incomplete) JSON data
        try:
            with open(json_filename, 'r', encoding='utf-8') as f:
                generated_data_list = json.load(f)
            print(f"  Loaded {len(generated_data_list)} processed entries.")
        except Exception as e:
            print(f"  !! Error: Failed to load {json_filename}: {e}")
            continue

        # 4. Load original split data
        original_split_data = dataset[split]
        print(f"  Loaded {len(original_split_data)} original entries.")

        # 5. Sanity check (ensure counts match)
        if len(generated_data_list) != len(original_split_data):
            print(f"  !! Critical error: Mismatch in number of entries!")
            print(f"  JSON ({split}.json) has {len(generated_data_list)} records.")
            print(f"  Original dataset ('{split}') has {len(original_split_data)} records.")
            print(f"  Skipping this split.")
            continue

        # 6. Core logic: merge data using zip (relying on aligned ordering)
        # Your 'image' field (e.g., "train_0000001.png") ensures that
        # the order of 'generated_data_list' matches 'original_split_data'.

        print(f"  Merging 'direct_answers'...")
        for generated_metadata, original_example in \
                tqdm(zip(generated_data_list, original_split_data),
                     total=len(generated_data_list),
                     desc=f"Merging {split}"):
            # Add (or overwrite) the missing field
            generated_metadata['direct_answers'] = original_example.get('direct_answers')

        # 7. Backup and overwrite
        backup_filename = os.path.join(json_output_dir, f"{split}.backup.json")
        try:
            if not os.path.exists(backup_filename):  # Backup only once
                shutil.copyfile(json_filename, backup_filename)
                print(f"  Backed up original file to {backup_filename}")
            else:
                print(f"  Backup file {backup_filename} already exists, will overwrite {json_filename} directly")
        except Exception as e:
            print(f"  !! Warning: Failed to create backup: {e}. Will overwrite directly.")

        print(f"  Writing {len(generated_data_list)} updated metadata entries back to {json_filename}...")
        with open(json_filename, 'w', encoding='utf-8') as f:
            # generated_data_list has already been modified in memory
            json.dump(generated_data_list, f, indent=4, ensure_ascii=False)

    print("\n--- Patching completed! ---")


if __name__ == "__main__":
    data_directory = "/path/to/data/aokvqa"

    print(f"Target root directory: {data_directory}")
    patch_json_with_direct_answers(base_output_dir=data_directory)
