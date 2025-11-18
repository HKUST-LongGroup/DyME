#!/usr/bin/env python
# -*- coding: utf-8 -*-
from transformers import activations
activations.PytorchGELUTanh = activations.GELUTanh
import os
import json
from PIL import Image
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Import model wrapper
try:
    from rex_omni import RexOmniWrapper
except ImportError:
    # Import DummyRex matching the one in clevr_processor.py
    print("Warning: 'from rex_omni import RexOmniWrapper' failed.")
    print("Using a dummy RexOmniWrapper (DummyRex) for testing only.")


    class DummyRex:
        def __init__(self, *args, **kwargs):
            print("INFO: DUMMY: Using DummyRex detector.")

        def inference(self, images, task, categories, **kwargs):
            print("INFO: DUMMY: DummyRex returning a fake center box.")
            if isinstance(images, Image.Image):
                w, h = images.size
            else:
                w, h = 800, 600
            x0, y0 = w * 0.25, h * 0.25
            x1, y1 = w * 0.75, h * 0.75
            return [{"extracted_predictions": {"anything": [{"type": "box", "coords": [x0, y0, x1, y1]}]}}]


    RexOmniWrapper = DummyRex

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
        return images, None

from clevr_processor import ClevrFactExtractor, _strip_tags


def run_test(configs, paths, gpu_id=0, sample_index=0):
    """
    Run the test pipeline on a single sample.
    """
    print("--- Starting single-run test (CoGenT) ---")

    # --- 1. Set environment and load models ---
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")

    try:
        print(f"Loading RexOmni... ({configs['rex_path']})")
        rex_model = RexOmniWrapper(
            model_path=configs['rex_path'],
            backend="transformers",
            max_tokens=2048,
            temperature=0.0,
        )

        print(f"Loading Qwen-VL... ({configs['qwen_path']})")
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            configs['qwen_path'],
            torch_dtype="float16",
            device_map="cuda:0",
            attn_implementation="flash_attention_2"
        )
        qwen_processor = AutoProcessor.from_pretrained(configs['qwen_path'])

        print("Models loaded.")
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    print("Loading dataset metadata...")
    try:
        dataset = load_dataset("MMInstruction/Clevr_CoGenT_TrainA_R1", split='train', streaming=True)
        example_iter = iter(dataset)
        for _ in range(sample_index + 1):
            example = next(example_iter)

    except Exception as e:
        print(f"Failed to load or filter dataset: {e}")
        return

    print(f"Processing sample {sample_index}...")

    try:
        # 1. Preprocessing
        prompt = example['problem']
        hint = _strip_tags(example['thinking'], 'think')
        answer = _strip_tags(example['solution'], 'answer')
        image = example['image'].convert("RGB")  # Get PIL image and convert to RGB

        # Save image for testing
        destination_image_path = os.path.join(paths['output_dir'], "images", f"test_sample_{sample_index}.jpg")
        os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
        image.save(destination_image_path, "JPEG")
        print(f"Loaded and saved test image: {destination_image_path}")

        # --- Stage 1: RexOmni detection ---
        print("Running RexOmni detection...")
        rex_results = rex_model.inference(images=image, task="detection", categories=["anything"])
        predictions = rex_results[0]["extracted_predictions"]
        detected_boxes = predictions.get("anything", [])
        print(f"RexOmni detected {len(detected_boxes)} 'anything' boxes.")

        visual_facts = []

        # --- Stage 2: Qwen-VL VQA ---
        for i, annotation in enumerate(detected_boxes):
            if annotation.get("type") == "box" and len(annotation.get("coords", [])) == 4:

                coords = annotation["coords"]
                print(f"  Processing box {i}: {coords}")

                crop_image = ClevrFactExtractor._crop_and_expand_box(image, coords)

                # Save cropped image for debugging
                crop_filename = f"./test_crop_{sample_index}_{i}.jpg"
                crop_image.save(crop_filename)
                print(f"    -> Saved cropped image for inspection: {crop_filename}")

                json_str = ClevrFactExtractor._query_qwen_vl(
                    crop_image, qwen_model, qwen_processor
                )

                json_obj_list = ClevrFactExtractor._parse_qwen_json(json_str)

                if json_obj_list:
                    obj_dict = json_obj_list[0]
                    obj_dict["bounding_box"] = [round(c, 2) for c in coords]
                    visual_facts.append(obj_dict)
                    print(f"    -> Qwen-VL result: {obj_dict}")
                else:
                    print(f"    -> Qwen-VL did not return valid JSON.")

        # --- 4. Print final result ---
        final_result = {
            "prompt": prompt,
            "answer": answer,
            "hint": hint,
            "image": destination_image_path,
            "visual_fact": visual_facts
        }

        print("\n" + "=" * 30)
        print("--- Single test result ---")
        print(json.dumps(final_result, indent=4, ensure_ascii=False))
        print("=" * 30 + "\n")

    except Exception as e:
        print(f"Error while processing sample {sample_index}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # --- 1. Model configs ---
    MODEL_CONFIGS = {
        "rex_path": "IDEA-Research/Rex-Omni",
        "qwen_path": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
    }

    # --- 2. Paths config ---
    PATHS = {
        # !! Change this to the directory where you want to save images and JSON !!
        "output_dir": "./clevr_cogent_output"
    }

    # --- 3. Test parameters ---
    GPU_ID_TO_USE = 0
    SAMPLE_INDEX_TO_TEST = 0  # Test the first CLEVR sample

    # --- 4. Run test ---
    run_test(
        configs=MODEL_CONFIGS,
        paths=PATHS,
        gpu_id=GPU_ID_TO_USE,
        sample_index=SAMPLE_INDEX_TO_TEST
    )
