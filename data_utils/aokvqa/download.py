import os
import json
import base64
import io
import multiprocessing
from functools import partial
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI, BadRequestError
import time

# Ensure PIL can handle large images
Image.MAX_IMAGE_PIXELS = None

# --- API and MLLM prompt configuration ---

API_PORTS = list(range(23333, 23333 + 8))
API_URL_TEMPLATE = "http://127.0.0.1:{port}/v1/"

# 2. Define prompts used to obtain visual_fact
VISUAL_FACT_SYSTEM_PROMPT = """
You are a helpful assistant that analyzes images and provides visual facts.
Your response MUST be a single, valid JSON object.
The JSON object should contain:
1. "description": A detailed and accurate description of the image.
2. "objects": A list of key objects, including their name, attributes, and approximate position in the image.

Example format:
{
  "description": "A person riding a bicycle on a city street.... (detailed description here)",
  "objects": [
    {"name": "person", "attributes": ["wearing helmet", "blue shirt"], "position": "center"},
    {"name": "bicycle", "attributes": ["red", "mountain bike"], "position": "center"},
    {"name": "street", "attributes": ["asphalt", "wet"], "position": "bottom"}
  ]
}
"""

VISUAL_FACT_USER_PROMPT = """
Analyze the attached image and provide the visual facts in the required JSON format.
For context, the user will be asked this question about the image (do not answer the question, just use it for context):
"{question}"
"""


def encode_image_to_base64(pil_image):
    """Convert a PIL Image object to a Base64-encoded string"""
    buffered = io.BytesIO()
    if pil_image.mode == "RGBA" or "transparency" in pil_image.info:
        pil_image.save(buffered, format="PNG")
        mime_type = "image/png"
    else:
        pil_image.save(buffered, format="JPEG")
        mime_type = "image/jpeg"

    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:{mime_type};base64,{img_str}"


def get_visual_fact(api_url, pil_image, question):
    """
    Call an external MLLM API to obtain visual_fact.
    """
    try:
        client = OpenAI(base_url=api_url, api_key="DUMMY_KEY")
        image_url = encode_image_to_base64(pil_image)

        messages = [
            {"role": "system", "content": VISUAL_FACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url,}},
                    {"type": "text", "text": VISUAL_FACT_USER_PROMPT.format(question=question)}
                ]
            }
        ]

        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            response_content = response.choices[0].message.content
            return response_content

        except (BadRequestError, Exception):
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            response_content = response.choices[0].message.content

            if response_content.startswith("```json"):
                response_content = response_content[7:].strip("` \n")

            return response_content

    except json.JSONDecodeError as e:
        print(f"!! JSON parse failed: {e}. API: {api_url}. Raw response: {response_content[:100]}...")
        return {"error": "Failed to parse JSON response", "raw_response": response_content}
    except Exception as e:
        print(f"!! API call failed: {e}. API: {api_url}")
        time.sleep(1)
        return {"error": f"API call failed: {str(e)}"}


# --- (This is the modified function) ---
def process_example_worker(example_with_index, split, image_output_dir, api_ports_list, fetch_visual_facts):
    """
    Worker function for multiprocessing.
    Process a single example: save image and (optionally) call the API.
    """
    i, example = example_with_index

    try:
        # 1. Extract metadata
        question = example["question"]
        pil_image = example["image"]

        # --- (A) Extract new fields ---
        choices_list = example.get("choices")
        correct_idx = example.get("correct_choice_idx")
        direct_answers_list = example.get("direct_answers")
        rationales_list = example.get("rationales")  # A-OKVQA has a 'rationales' field

        # --- (B) Determine "answer" (prefer choice as requested) ---
        answer = None
        if choices_list and correct_idx is not None and 0 <= correct_idx < len(choices_list):
            answer = choices_list[correct_idx]
        elif direct_answers_list:
            answer = direct_answers_list[0]  # Fallback to direct_answers

        # --- (C) Determine "hint" (the longest rationale) ---
        hint = None
        if rationales_list:
            try:
                # Ensure rationales_list is a non-empty list of strings
                if isinstance(rationales_list, list) and len(rationales_list) > 0:
                    string_rationales = [r for r in rationales_list if isinstance(r, str)]
                    if string_rationales:
                        hint = max(string_rationales, key=len)  # Choose the longest string
            except Exception as e:
                print(f"!! Warning: Failed to compute 'hint' (index {i}). Error: {e}")
                pass  # hint remains None

        # --- (D) End of extraction ---
        # 2. Generate and save image filename
        generated_filename = f"{split}_{i:07d}.png"
        image_save_path = os.path.join(image_output_dir, generated_filename)

        # 3. Save image (I/O operation)
        if not os.path.exists(image_save_path):
            pil_image.save(image_save_path)

        # 4. (Key step) Get Visual Fact
        visual_fact_data = None
        if fetch_visual_facts:
            # Use API ports in a round-robin manner according to index 'i'
            port_to_use = api_ports_list[i % len(api_ports_list)]
            api_url = API_URL_TEMPLATE.format(port=port_to_use)

            visual_fact_data = get_visual_fact(api_url, pil_image, question)

        # 5. Build metadata dict (using the new format you specified)
        metadata = {
            "question": question,
            "question_wo_prompt": question,
            "answer": answer,
            "choices": choices_list,  # Store the full options list
            "image": image_save_path,
            "visual_fact": visual_fact_data,
            "hint": hint
        }

        return metadata

    except Exception as e:
        print(f"!! Fatal error in worker (index {i}): {e}")
        return None  # The main process will filter out None


def save_aokvqa_with_facts(base_output_dir="/aokvqa_output"):
    """
    Load A-OKVQA, save images with multiprocessing, and obtain visual_facts for the training split.
    """

    # 1. Define output directories
    base_dir_abs = os.path.abspath(base_output_dir)
    image_output_dir = os.path.join(base_dir_abs, "images")
    json_output_dir = os.path.join(base_dir_abs, "json")

    # 2. Create directories
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    print(f"Images will be saved to (absolute path): {image_output_dir}")
    print(f"JSON will be saved to (absolute path): {json_output_dir}")
    print(f"Using {len(API_PORTS)} API ports: {API_PORTS}")

    # 3. Load dataset
    print("Loading HuggingFaceM4/A-OKVQA dataset...")
    try:
        dataset = load_dataset("HuggingFaceM4/A-OKVQA")
        # Use a small subset for testing
        # dataset['train'] = dataset['train'].select(range(100))
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Dataset splits: {list(dataset.keys())}")

    # 4. Define number of worker processes
    num_workers = 64
    print(f"Starting {num_workers} worker processes...")

    # 5. Iterate over each split
    for split in dataset.keys():

        fetch_facts = (split == 'train')

        if fetch_facts:
            print(f"\n--- Processing split {split} (will call MLLM API) ---")
        else:
            print(f"\n--- ------------------- ---")
            print(f"--- Processing split {split} (only saving images, visual_fact=None) ---")
            print(f"--- ------------------- ---")

        metadata_list = []

        worker_func = partial(
            process_example_worker,
            split=split,
            image_output_dir=image_output_dir,
            api_ports_list=API_PORTS,
            fetch_visual_facts=fetch_facts
        )

        tasks = list(enumerate(dataset[split]))
        total_count = len(tasks)

        # 6. Use multiprocessing pool
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap_unordered(worker_func, tasks), total=total_count, desc=f"Processing {split}"):
                if result:
                    metadata_list.append(result)

        print(f"  Split {split} finished. Success: {len(metadata_list)} / {total_count}.")

        # 7. (Optional) Sort
        metadata_list.sort(key=lambda x: x['image'])

        # 8. Write JSON file
        json_filename = os.path.join(json_output_dir, f"{split}.json")
        print(f"Saving {len(metadata_list)} metadata entries to {json_filename}...")

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)

    print(f"\n--- All processing completed! ---")
    print(f"All image files saved in: '{image_output_dir}'")
    print(f"All JSON files saved in: '{json_output_dir}'")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # Change to your desired output directory
    save_aokvqa_with_facts(base_output_dir="/path/to/data/aokvqa")
