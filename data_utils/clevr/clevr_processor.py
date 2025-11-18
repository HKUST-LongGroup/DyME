#!/usr/bin/env python
# -*- coding: utf-8 -*-
from accelerate.utils import gather_object

try:
    from transformers import activations

    activations.PytorchGELUTanh = activations.GELUTanh
except ImportError:
    print("Note: Unable to apply PytorchGELUTanh patch. If you encounter an ImportError, please check the transformers version.")
# --- End of patch ---

import os
import shutil
import json
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator

# (qwen_vl_utils import and fallback remain unchanged)
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Warning: Failed to import 'qwen_vl_utils.process_vision_info'.")


    def process_vision_info(messages):
        images = []
        for msg in messages:
            if msg['role'] == 'user':
                for content in msg['content']:
                    if content['type'] == 'image':
                        images.append(content['image'])
        return images, None  # Return (images, videos)

Image.MAX_IMAGE_PIXELS = None

# (RexOmni dependency and DummyRex remain unchanged)
try:
    from rex_omni import RexOmniWrapper
except ImportError:
    print("Warning: 'from rex_omni import RexOmniWrapper' failed.")
    print("Using a dummy RexOmniWrapper (DummyRex) for testing only.")


    class DummyRex:
        def __init__(self, *args, **kwargs):
            print("INFO: DUMMY: Using DummyRex detector.")

        def inference(self, images, task, categories, **kwargs):
            print("INFO: DUMMY: DummyRex returning fake center boxes.")

            # Batch-supporting Dummy
            results = []

            # Ensure images is a list
            if not isinstance(images, list):
                images = [images]

            for img in images:
                if isinstance(img, Image.Image):
                    w, h = img.size
                else:
                    w, h = 800, 600
                x0, y0 = w * 0.25, h * 0.25
                x1, y1 = w * 0.75, h * 0.75
                results.append({"extracted_predictions": {"anything": [{"type": "box", "coords": [x0, y0, x1, y1]}]}})
            return results


    RexOmniWrapper = DummyRex


def _strip_tags(text, tag_name):
    # (unchanged)
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(rf'<{tag_name}>', '', text, flags=re.IGNORECASE)
    text = re.sub(rf'</{tag_name}>', '', text, flags=re.IGNORECASE)
    return text.strip()


# --- Core VQA helper functions (moved to global scope) ---

def _crop_and_expand_box(image, box, padding_pixels=20):
    # (unchanged)
    x0, y0, x1, y1 = [int(c) for c in box]
    img_w, img_h = image.size
    x0_new = max(0, x0 - padding_pixels)
    y0_new = max(0, y0 - padding_pixels)
    x1_new = min(img_w, x1 + padding_pixels)
    y1_new = min(img_h, y1 + padding_pixels)
    return image.crop((x0_new, y0_new, x1_new, y1_new))


# --- ★★★ Optimization 1: Change VQA queries to batched processing ★★★ ---
def _query_qwen_vl_BATCH(crop_images_list, model, processor, accelerator):
    """
    Use Qwen-VL to query cropped image patches in batch and return a list of JSON strings.
    """
    if not crop_images_list:
        return []

    prompt = """This is an object from a CLEVR scene. Analyze the primary object in the image.
Respond *strictly* with a JSON list (containing one dictionary) in the following format:
[
    {"object": "object_name", "attributes": ["attr1", "attr2"]}
]
- "object": The shape of the object (e.g., "sphere", "cube", "cylinder", "cone").
- "attributes": A list of visual attributes (e.g., ["blue", "large", "metal", "shiny", "rubber"]).
Provide only the JSON list:"""

    # 1. Create messages for each image in the batch
    template_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder.jpg"},  # Placeholder
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        # 2. Generate chat prompt text once
        chat_prompt_text = processor.apply_chat_template(
            template_messages, tokenize=False, add_generation_prompt=True
        )

        num_crops = len(crop_images_list)
        batch_text = [chat_prompt_text] * num_crops
        batch_images = crop_images_list

        unwrapped_model = accelerator.unwrap_model(model)

        # 3. Use text list and image list for batched processing
        inputs = processor(
            text=batch_text,
            images=batch_images,
            padding=True,  # Important: enable padding to handle batching
            return_tensors="pt",
        ).to(unwrapped_model.device)

        # 4. Batch generation
        generated_ids = unwrapped_model.generate(**inputs, max_new_tokens=256, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 5. Batch decode
        output_texts_list = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_texts_list

    except Exception as e:
        print(f"Qwen-VL batched inference failed: {e}")
        # Return a list of empty JSON placeholders to avoid zip errors
        return ["[]"] * len(crop_images_list)


def _parse_qwen_json(response_text):
    # (unchanged)
    try:
        match = re.search(r'```json\s*(\[.*\])\s*```', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        match = re.search(r'(\[.*\])', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        return []
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {response_text}")
        return []
    except Exception as e:
        print(f"Unknown error occurred while parsing JSON: {e}")
        return []


def _load_and_preprocess_data(base_output_dir, image_output_dir):
    # (unchanged)
    print("Loading 'MMInstruction/Clevr_CoGenT_TrainA_R1'...")
    try:
        dataset = load_dataset("MMInstruction/Clevr_CoGenT_TrainA_R1", split='train')
    except Exception as e:
        print(f"Failed to load dataset 'MMInstruction/Clevr_CoGenT_TrainA_R1': {e}")
        return []

    # Only use the first few samples for testing (still 100 here)
    # dataset = dataset.select(range(100))
    print(f"Loaded {len(dataset)} samples.")

    job_list = []
    print("Preprocessing data (saving images and parsing text)...")
    for i, example in enumerate(tqdm(dataset, desc="Preprocessing progress")):
        prompt = example['problem']
        hint = _strip_tags(example['thinking'], 'think')
        answer = _strip_tags(example['solution'], 'answer')

        image = example['image']
        if not isinstance(image, Image.Image):
            print(f"Warning: sample {i} is not a PIL image, skipped.")
            continue

        image_filename = f"clevr_cogent_trainA_r1_{i:07d}.jpg"
        destination_image_path = os.path.join(image_output_dir, image_filename)

        try:
            os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
            if not os.path.exists(destination_image_path):
                image.convert("RGB").save(destination_image_path, "JPEG")
        except Exception as e:
            print(f"Warning: failed to save image for sample {i}, skipped. Error: {e}")
            continue

        job_list.append({
            "prompt": prompt,
            "answer": answer,
            "hint": hint,
            "destination_image_path": destination_image_path
        })

    print(f"Successfully preprocessed {len(job_list)} items.")

    job_list_path = os.path.join(base_output_dir, "job_list.json")
    with open(job_list_path, 'w', encoding='utf-8') as f:
        json.dump(job_list, f)

    print(f"Job list saved to: {job_list_path}")
    return job_list


def main():
    # (1. Initialize Accelerator - unchanged)
    accelerator = Accelerator()

    # (0. Define configuration - unchanged)
    MODEL_CONFIGS = {
        "rex_path": "IDEA-Research/Rex-Omni",
        "qwen_path": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
    }
    OUTPUT_DIR = "/path/to/data/clevr_cogent_output"
    IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
    JSON_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "json")

    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

    # (2. Preprocessing (run only on main process) - unchanged)
    job_list_path = os.path.join(OUTPUT_DIR, "job_list.json")
    if accelerator.is_main_process:
        print("Main process [Pre-processing]: loading and preprocessing data...")
        _load_and_preprocess_data(OUTPUT_DIR, IMAGE_OUTPUT_DIR)

    # (3. Synchronization - unchanged)
    accelerator.wait_for_everyone()

    # (4. Load and distribute jobs - unchanged)
    if not accelerator.is_main_process:
        print(f"Process {accelerator.process_index}: loading job_list.json...")
    try:
        with open(job_list_path, 'r', encoding='utf-8') as f:
            all_jobs = json.load(f)
    except Exception as e:
        print(f"Process {accelerator.process_index} failed to load job_list.json: {e}")
        return
    total_jobs = len(all_jobs)
    num_processes = accelerator.num_processes
    jobs_per_process = total_jobs // num_processes
    start_index = accelerator.process_index * jobs_per_process
    end_index = (accelerator.process_index + 1) * jobs_per_process
    if accelerator.is_last_process:
        end_index = total_jobs
    my_jobs = all_jobs[start_index:end_index]
    print(f"[Process {accelerator.process_index}]:"
          f" assigned {len(my_jobs)} jobs (indices from {start_index} to {end_index}).")

    # (5. Load models (each process loads its own copy) - unchanged)
    try:
        try:
            from transformers import activations
            activations.PytorchGELUTanh = activations.GELUTanh
        except ImportError:
            pass

        print(f"[Process {accelerator.process_index}]: loading RexOmni...")
        rex_model = RexOmniWrapper(
            model_path=MODEL_CONFIGS['rex_path'],
            backend="transformers",
            max_tokens=2048,
            temperature=0.0,
        )

        print(f"[Process {accelerator.process_index}]: loading Qwen-VL...")
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_CONFIGS['qwen_path'],
            torch_dtype="float16",
            device_map="cuda",
            attn_implementation="flash_attention_2"
        )
        qwen_processor = AutoProcessor.from_pretrained(MODEL_CONFIGS['qwen_path'])

        # Note: RexOmniWrapper (if not Dummy) may need .to(accelerator.device),
        # but Qwen-VL already has its device specified via device_map="cuda".
        # accelerator.prepare is still a good way to manage models.
        qwen_model, rex_model = accelerator.prepare(qwen_model, rex_model)

        print(f"[Process {accelerator.process_index}]: models loaded.")
    except Exception as e:
        print(f"[Process {accelerator.process_index}]: failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return

    # 1. Define a batch size (Batch Size)
    REX_BATCH_SIZE = 16  # <-- Adjust this value according to your VRAM

    print(f"[Process {accelerator.process_index}]:"
          f" starting to process {len(my_jobs)} jobs with Rex batch size {REX_BATCH_SIZE}.")

    processed_metadata_list = []

    # 2. Modify main loop: iterate with step REX_BATCH_SIZE
    for i in tqdm(range(0, len(my_jobs), REX_BATCH_SIZE),
                  desc=f"Worker {accelerator.process_index} batch progress",
                  disable=not accelerator.is_main_process):

        # 3. Prepare jobs and images for this batch
        batch_jobs = my_jobs[i: i + REX_BATCH_SIZE]
        batch_images = []
        batch_image_paths = []  # for debugging

        valid_jobs_in_batch = []

        for job in batch_jobs:
            try:
                img_path = job['destination_image_path']
                batch_image_paths.append(img_path)
                batch_images.append(Image.open(img_path).convert("RGB"))
                valid_jobs_in_batch.append(job)  # only jobs with successfully loaded images are valid
            except Exception as e:
                print(f"[Process {accelerator.process_index}]:"
                      f" failed to load image {img_path}: {e}, skipping this image in this batch.")
                # We do not add the image or job, keeping batch_images and valid_jobs_in_batch in sync

        if not batch_images:  # if all images in this batch failed to load
            continue

        try:
            # 4. ★ Key: run RexOmni in batch
            # (we only pass successfully loaded images)
            all_rex_results = rex_model.inference(
                images=batch_images,  # pass the image list
                task="detection",
                categories=["anything"]
            )

            # 5. Iterate over results in this batch
            # all_rex_results length should equal batch_images (and valid_jobs_in_batch)
            if len(all_rex_results) != len(valid_jobs_in_batch):
                print(f"[Process {accelerator.process_index}]: Warning: RexOmni "
                      f"returned {len(all_rex_results)} results, but "
                      f"{len(valid_jobs_in_batch)} inputs were provided. Skipping this batch.")
                continue

            for job, image, rex_result in zip(valid_jobs_in_batch, batch_images, all_rex_results):

                predictions = rex_result["extracted_predictions"]
                detected_boxes = predictions.get("anything", [])

                visual_facts = []
                crops_to_process = []
                box_coords_list = []

                # 6. Collect all crops to be processed (from this image)
                for annotation in detected_boxes:
                    if annotation.get("type") == "box" and len(annotation.get("coords", [])) == 4:
                        coords = annotation["coords"]
                        crop_image = _crop_and_expand_box(image, coords)
                        crops_to_process.append(crop_image)
                        box_coords_list.append(coords)

                # 7. Batch VQA (logic remains the same, still batch *per image* crops)
                if crops_to_process:
                    json_str_list = _query_qwen_vl_BATCH(
                        crops_to_process, qwen_model, qwen_processor, accelerator
                    )

                    # 8. Iterate over batched results and parse (logic unchanged)
                    for json_str, coords in zip(json_str_list, box_coords_list):
                        json_obj_list = _parse_qwen_json(json_str)
                        if json_obj_list:
                            try:
                                obj_dict = json_obj_list[0]
                                obj_dict["bounding_box"] = [round(c, 2) for c in coords]
                                visual_facts.append(obj_dict)
                            except (IndexError, TypeError, KeyError) as e:
                                print(f"[Process {accelerator.process_index}]: "
                                      f"Error while parsing batched result: {e} | JSON: {json_str}")

                # 9. Aggregate results for this job (logic unchanged)
                processed_metadata_list.append({
                    "question": job['prompt'],
                    "answer": job['answer'],
                    "question_wo_prompt": job['prompt'],
                    "hint": job['hint'],
                    "image": job['destination_image_path'],
                    "visual_fact": visual_facts
                })
                # --- End of inner loop logic ---

        except Exception as e:
            print(f"[Process {accelerator.process_index}]: "
                  f"Error while processing batch {i // REX_BATCH_SIZE} (images {batch_image_paths}): {e}")
            import traceback
            traceback.print_exc()

    # --- End of loop ---

    print(f"[Process {accelerator.process_index}]:"
          f" process finished, handled {len(processed_metadata_list)} items.")

    # (7. Gather all results - unchanged)
    print(f"[Process {accelerator.process_index}]: gathering results...")
    all_results_list_of_lists = gather_object(processed_metadata_list)

    # (8. Save (only on main process) - ★★★ using fixed GATHER logic ★★★)
    if accelerator.is_main_process:
        print("Main process [Saving]: aggregating and saving all results...")

        # --- Key fix ---
        # gather_object already returns a flattened list of dictionaries (List[dict]).
        final_metadata_list = all_results_list_of_lists
        # --- End of fix ---

        json_filename = os.path.join(JSON_OUTPUT_DIR, "clevr_cogent_trainA_r1_processed.json")

        # Verify the count
        print(f"Total number of aggregated items: {len(final_metadata_list)}")
        if len(final_metadata_list) > 0:
            print(f"Type of first item: {type(final_metadata_list[0])}")

        print(f"\nSaving {len(final_metadata_list)} metadata entries to {json_filename}...")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(final_metadata_list, f, indent=4, ensure_ascii=False)

        print(f"\n--- Processing completed! ---")
        print(f"All image files have been saved in: '{IMAGE_OUTPUT_DIR}'")
        print(f"Final JSON file has been saved in: '{JSON_OUTPUT_DIR}'")


if __name__ == "__main__":
    main()
