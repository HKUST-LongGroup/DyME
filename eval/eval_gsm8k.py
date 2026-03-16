import torch
import re  # Import the regular expression library
from accelerate import Accelerator
from datasets import load_dataset
from torch.distributed import all_gather_object
# Switch to CausalLM model
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from trl.models import unwrap_model_for_generation
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

# --- Helper functions: parsing GSM8K answers ---

def parse_ground_truth(answer_str):
    """Extract the numeric answer like '123' from a string in the format '...#### 123'"""
    try:
        return answer_str.split('####')[-1].strip().replace(",", "")
    except:
        return ""


def parse_prediction(pred_str):
    """Extract the last number from the model's generated text as the predicted answer"""

    # Prefer to look for the "Answer:" marker first
    answer_marker = "Answer:"
    if answer_marker in pred_str:
        pred_str = pred_str.split(answer_marker)[-1]

    # Remove thousands separators
    pred_str = pred_str.replace(",", "")

    # Find all numbers (including integers and decimals)
    # This regex matches: optional - or +, followed by digits, optional decimal point and more digits
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", pred_str)

    if matches:
        # Return the last matched number
        return matches[-1].strip()
    else:
        # If no number is found, return an empty string
        return ""


# --------------------------------------

accelerator = Accelerator()
DEVICE = accelerator.device

# --- Model and tokenizer configuration ---
model_args = {"torch_dtype": torch.bfloat16}  # Keep bf16 for better performance

model_id = '/path/to/dyme-qwen25-GSM8K-new/checkpoint-466'
model_id = '/path/to/dyme-qwen25-GSM8K-new/checkpoint-2097'


if accelerator.is_main_process:
    print(f"Loading model: {model_id}")

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, config=config, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(DEVICE)

model.eval()

# Configure tokenizer for batched generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Removed AutoProcessor and image processor configuration

# --- Prompt template ---
# This CoT template is well-suited for GSM8K
PROMPT_TEMPLATE = (
    "Your task is to answer the question below. "
    "Give step by step thinking before you answer, and when you're ready to answer, "
    "please use the format \"Answer: ..\"\n\n"
    "Question:\n\n{question}"
)


def run_model_batch(batch_data_list):  # Image-related processing removed
    batch_formatted_prompts_for_chat_template = []

    for item in batch_data_list:
        item_model_input_text = item['model_input_text'].strip()

        # Format the question using the template
        question_with_tags = PROMPT_TEMPLATE.format(question=item_model_input_text)

        # Build chat template input for Qwen
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {
                "role": "user",
                "content": question_with_tags
            },
        ]

        try:
            # tokenize=False so we can batch later
            templated_prompt_str = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        except Exception:
            # A fallback format suitable for Qwen
            templated_prompt_str = f"<|im_start|>user\n{question_with_tags}<|im_end|>\n<|im_start|>assistant\n"

        batch_formatted_prompts_for_chat_template.append(templated_prompt_str)

    # Batched tokenization
    inputs = tokenizer(
        batch_formatted_prompts_for_chat_template,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048  # A reasonable max length for Qwen
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with unwrap_model_for_generation(model, accelerator) as unwrapped_model_instance:
        generated_ids = unwrapped_model_instance.generate(
            **inputs,
            max_new_tokens=1024,  # GSM8K answers may need some CoT space
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id  # Explicitly specify pad_token_id
        )

    input_ids_length = inputs['input_ids'].shape[1]
    newly_generated_ids = generated_ids[:, input_ids_length:]

    generated_texts = tokenizer.batch_decode(
        newly_generated_ids,
        skip_special_tokens=True,
    )
    return [text.strip() for text in generated_texts]


# --- Main evaluation logic ---
task = 'gsm8k'

if task == 'gsm8k':
    if accelerator.is_main_process:
        print("Loading GSM8K dataset...")
    try:
        full_dataset = load_dataset("gsm8k", "main", trust_remote_code=True)['test']
    except Exception as e:
        if accelerator.is_main_process:
            print(f"Failed to load dataset directly. Error: {e}")
        raise

    # full_dataset = full_dataset.select(range(80)) # Uncomment for quick tests

    eval_datasets_all_prepared = []
    for d_item in tqdm(full_dataset, desc="Preparing dataset", disable=not accelerator.is_main_process):
        raw_question = d_item['question']
        ground_truth_answer_full = d_item.get('answer')
        if not ground_truth_answer_full:
            continue
        eval_datasets_all_prepared.append({
            'model_input_text': raw_question,
            'answer': ground_truth_answer_full,
            'original_question': raw_question
        })

    total_items = len(eval_datasets_all_prepared)
    if total_items == 0:
        if accelerator.is_main_process:
            print("No data prepared for evaluation. Exiting.")
    else:
        BATCH_SIZE = 2

        # 1. Create DataLoader
        # collate_fn=lambda x: x makes the DataLoader output a list of items per batch directly
        eval_dataloader = DataLoader(eval_datasets_all_prepared, batch_size=BATCH_SIZE, collate_fn=lambda x: x)

        # 2. Prepare model and DataLoader with accelerator
        # accelerator will automatically handle DistributedSampler
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

        # Store results for the current process
        local_scores = []

        pbar = None
        if accelerator.is_main_process:
            pbar = tqdm(total=len(eval_dataloader), desc=f"Eval on Main Proc", dynamic_ncols=True)

        # --- 3. Data processing loop ---
        # Each process only handles its own subset of data; no manual sharding needed
        for batch in eval_dataloader:
            if not batch:
                continue

            batch_predictions_texts = run_model_batch(batch)

            for item_idx_in_batch, full_pred_text in enumerate(batch_predictions_texts):
                original_item = batch[item_idx_in_batch]
                ground_truth_answer_full = original_item['answer']

                gt_answer_clean = parse_ground_truth(ground_truth_answer_full)
                pred_answer_clean = parse_prediction(full_pred_text)

                score = 1.0 if gt_answer_clean == pred_answer_clean and gt_answer_clean != "" else 0.0
                local_scores.append(score)

                # For debugging, print only on the main process to avoid noisy output
                if accelerator.is_main_process:
                    tqdm.write("-" * 20)
                    tqdm.write(f"Q: {original_item['original_question'][:50]}...")
                    tqdm.write(f"PRED: [{pred_answer_clean}] | GT: [{gt_answer_clean}] | Score: {score}")

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        # --- 4. Synchronization and reporting ---
        # Wait for all processes to finish the loop above
        accelerator.wait_for_everyone()

        # Now it is safe to collect results from all processes
        # Each process will create a gathered_scores list containing all processes' local_scores
        gathered_scores_list_of_lists = [None] * accelerator.num_processes
        all_gather_object(gathered_scores_list_of_lists, local_scores)

        # Only the main process computes and prints the final report
        if accelerator.is_main_process:
            print("\n--- Final Report ---")

            # Flatten results from all processes into a single list
            final_scores = [score for sublist in gathered_scores_list_of_lists for score in sublist]

            total_samples_processed = len(final_scores)

            if total_samples_processed > 0:
                final_accuracy = np.array(final_scores).mean()
                print(f"Global samples processed: {total_samples_processed} / {total_items}")
                # Note: Due to DistributedSampler, some samples may be dropped or duplicated
                # to ensure even distribution, so processed count may not equal total_items exactly.
                print(f"Final Global Mean Accuracy (EM): {final_accuracy:.4f}")
            else:
                print("No scores were gathered from any process.")

else:
    if accelerator.is_main_process:
        print(f"Task '{task}' is not configured for batched evaluation in this script.")
