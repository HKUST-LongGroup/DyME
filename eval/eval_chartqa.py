import torch
from PIL import Image
from accelerate import Accelerator
# Ensure this path is correct and the utility is available.
from datasets import load_dataset
from torch.distributed import all_gather_object
from transformers import AutoProcessor, AutoConfig, AutoTokenizer, LlavaOnevisionForConditionalGeneration
from trl.models import unwrap_model_for_generation

from data_utils.chart.evaluator import eval_one_chart
from reward_utils.compute_rewards import split_initial_context

accelerator = Accelerator()
from tqdm import tqdm
import numpy as np

DEVICE = accelerator.device

# Model and Processor Configuration
model_args = {}  # Use {"torch_dtype": torch.bfloat16} if desired and supported

model_id = '/path/to/dyme-k-8/final_checkpoint'

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, config=config, trust_remote_code=True)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to(DEVICE)

model.eval()
# Make sure model and processor are loaded before being potentially used in generate_inner if it were called
# model = Idefics3ForConditionalGeneration.from_pretrained(model_id, **model_args).to(DEVICE)

processor = AutoProcessor.from_pretrained(model_id)

# Configure image processor size
# This can consume significant VRAM. Ensure it's intended.
if hasattr(processor.image_processor, 'size') and isinstance(processor.image_processor.size, dict):
    # if 'longest_edge' in processor.image_processor.size:
    #     print('Setting image processor longest_edge to 2048')
    #     processor.image_processor.size['longest_edge'] = 512 * 4
    processor.tokenizer.padding_side = 'left'
else:
    print(
        f"Warning: Could not directly set 'longest_edge' via dict. Current image processor size config: {processor.image_processor.size}"
    )
    # Attempt an alternative if applicable, e.g.
    # processor.image_processor.size = {"longest_edge": 512 * 4} # if size itself can be replaced
    # Or this might indicate that `size` is a single value or a different structure.

PROMPT_TEMPLATE = (
    "Your task is to answer the question below. "
    "Give step by step thinking before you answer, and when you're ready to answer, "
    "please use the format \"Answer: ..\"\n\n"
    "Question:\n\n{question}"
)


def run_kh_batch(batch_data_list):  # Renamed from run_kh, takes a batch
    batch_images = []
    batch_formatted_prompts_for_chat_template = []

    for item in batch_data_list:
        image_path = item['image_path']
        # 'item_model_input_text' already contains chart instructions + raw_question
        item_model_input_text = item['model_input_text'].strip()

        # question_with_tags = prompt + item_model_input_text
        # question_with_tags = f"""{item_model_input_text} Think step by step and then answer the question."""
        question_with_tags = PROMPT_TEMPLATE.format(question=item_model_input_text)
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")  # Assuming image_path is already a PIL Image object
        batch_images.append(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_with_tags},
                ]
            },
        ]
        try:
            templated_prompt_str = processor.apply_chat_template(messages, add_generation_prompt=True)
            templated_prompt_str = templated_prompt_str.strip()
        except Exception:
            templated_prompt_str = f"USER: <image>\n{question_with_tags}\nASSISTANT:"
        batch_formatted_prompts_for_chat_template.append(templated_prompt_str)

    inputs = processor(
        text=batch_formatted_prompts_for_chat_template,
        images=batch_images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    # inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    inputs = {
        k: v.to(DEVICE).to(torch.bfloat16) if v.is_floating_point() else v.to(DEVICE)
        for k, v in inputs.items()
    }

    with unwrap_model_for_generation(model, accelerator) as unwrapped_model_instance:
        generated_ids = unwrapped_model_instance.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    input_ids_length = inputs['input_ids'].shape[1]
    newly_generated_ids = generated_ids[:, input_ids_length:]

    generated_texts = processor.batch_decode(
        newly_generated_ids,
        skip_special_tokens=True,  # Special tokens like <eos> are removed. <image> might be too.
    )
    return [text.strip('.').strip() for text in generated_texts]


# --- Main Evaluation Logic ---
task = 'chart'
# dt_record_local is initialized inside the if task == 'chart' block

if task == 'chart':
    dt_record_local = {}  # Store results for the current process
    if accelerator.is_main_process:
        print("Loading ChartQA dataset...")
    try:
        full_dataset = load_dataset("HuggingFaceM4/ChartQA", trust_remote_code=True)['test']
    except Exception as e:
        if accelerator.is_main_process:
            print(f"Failed to load dataset directly. Error: {e}")
            print("Attempting to load with specific revision if applicable, or check path/connection.")
        # For example, you can try a specific revision (if known) or ensure path and network connection are correct
        # full_dataset = load_dataset("HuggingFaceM4/ChartQA", revision="main", trust_remote_code=True)['test']
        raise  # Re-raise the exception since we cannot proceed without the dataset

    # full_dataset = full_dataset.select(range(80)) # Uncomment for quick testing

    eval_datasets_all_prepared = []
    # chart_instructions_prefix = (
    #     "For the question below, follow the following instructions:\n"
    #     # ... (your detailed instructions) ...
    #     + "-Try to include the full label from the graph when asked about an entity.\n"
    #     + "Question: "
    # )

    for d_item in tqdm(full_dataset, desc="Preparing dataset", disable=not accelerator.is_main_process):
        image_path = d_item['image']
        raw_question = d_item['query']
        answer_list = d_item.get('label')  # Use .get() in case 'label' field does not exist
        if not answer_list:  # If 'label' is missing or an empty list
            if accelerator.is_main_process:
                tqdm.write(
                    f"Warning: Item missing 'label' or 'label' is empty. Query: {raw_question[:50]}..."
                )
            # Decide how to handle this: skip this sample or use a default answer
            continue  # Skip this sample
        answer = answer_list[0]

        model_input_text_for_template = raw_question
        eval_datasets_all_prepared.append({
            'image_path': image_path,
            'model_input_text': model_input_text_for_template,
            'answer': answer,
            'original_question': raw_question
        })

    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    total_items = len(eval_datasets_all_prepared)

    if total_items == 0:
        if accelerator.is_main_process:
            print("No data prepared for evaluation after filtering. Exiting chart evaluation.")
    else:
        items_per_proc = total_items // num_processes
        extra_items = total_items % num_processes
        local_start_index = process_index * items_per_proc + min(process_index, extra_items)
        num_local_items = items_per_proc + (1 if process_index < extra_items else 0)
        local_end_index = local_start_index + num_local_items
        eval_datasets_local = eval_datasets_all_prepared[local_start_index:local_end_index]

        BATCH_SIZE = 32  # Adjust according to your VRAM
        REPORT_INTERVAL_BATCHES = 1  # Report once every N local batches (main process prints global stats)

        # if accelerator.is_main_process:
        #     print(f"Total items for evaluation: {total_items}")
        #     print(f"Process {process_index} handling {len(eval_datasets_local)} items.")
        #     print(f"Batch size per process: {BATCH_SIZE}, Reporting interval: {REPORT_INTERVAL_BATCHES} local batches.")

        pbar = None
        if accelerator.is_main_process and len(eval_datasets_local) > 0:  # Create pbar only if there is data
            pbar = tqdm(total=len(eval_datasets_local), desc=f"Eval Proc {process_index}", dynamic_ncols=True)

        dt_record_local['res'] = []
        num_local_batches = (len(eval_datasets_local) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx_local in range(num_local_batches):
            start_idx = batch_idx_local * BATCH_SIZE
            end_idx = min((batch_idx_local + 1) * BATCH_SIZE, len(eval_datasets_local))
            current_batch_list = eval_datasets_local[start_idx:end_idx]

            if not current_batch_list:
                continue

            batch_predictions_texts = run_kh_batch(current_batch_list)

            for item_idx_in_batch, full_pred_text in enumerate(batch_predictions_texts):
                original_item = current_batch_list[item_idx_in_batch]
                ground_truth_answer = original_item['answer']

                _, parsed_pred_answer = split_initial_context(full_pred_text)
                if not parsed_pred_answer.strip():
                    parsed_pred_answer = full_pred_text  # Fallback to full prediction if parsed answer is empty

                score = eval_one_chart(parsed_pred_answer, ground_truth_answer)  # nlp object is global
                dt_record_local['res'].append(score)

                # (Optional) Main process prints a few prediction details
                if accelerator.is_main_process:
                    print(full_pred_text, "######", ground_truth_answer, "######", score)

            if pbar:
                pbar.update(len(current_batch_list))

            # --- Intermediate reporting logic ---
            is_last_local_batch = (batch_idx_local == num_local_batches - 1)
            # Every REPORT_INTERVAL_BATCHES local batches, or on the last local batch of this process,
            # perform synchronization and reporting
            should_sync_and_report = ((batch_idx_local + 1) % REPORT_INTERVAL_BATCHES == 0) or is_last_local_batch

            # Make sure that even if REPORT_INTERVAL_BATCHES is 1, we do not report when there is no data
            # (e.g., len(eval_datasets_local) == 0)
            if len(eval_datasets_local) == 0:  # If the current process has no data, skip reporting logic
                should_sync_and_report = False
                # If num_local_batches > 0, this check ensures we report only when there is data

            if num_local_batches == 0 and is_last_local_batch:  # Special case: process has no data but must join final sync
                should_sync_and_report = True

            if should_sync_and_report:
                accelerator.wait_for_everyone()  # Wait for all processes to reach the sync point

                gathered_all_processes_data = [None] * num_processes
                # Each process sends its *current accumulated* dt_record_local
                # If a process has no data, dt_record_local['res'] is an empty list, which is fine
                all_gather_object(gathered_all_processes_data, dt_record_local)

                if accelerator.is_main_process:
                    current_global_scores_list = []
                    for process_data_dict in gathered_all_processes_data:
                        if process_data_dict and 'res' in process_data_dict:
                            current_global_scores_list.extend(process_data_dict['res'])

                    total_samples_processed_globally = len(current_global_scores_list)

                    report_title = "--- Intermediate Report ---"
                    # Check whether this is the final reporting point where all processes have finished
                    # A simple heuristic: if this is the last local batch on the main process
                    # and the total collected samples equal the total number of items
                    if is_last_local_batch and total_samples_processed_globally == total_items:
                        report_title = "--- Final Report ---"
                    elif is_last_local_batch:  # Last batch on main process but perhaps not all samples are done yet
                        report_title = (
                            f"--- Report (Main Proc Last Batch, {batch_idx_local + 1}/{num_local_batches}) ---"
                        )

                    tqdm.write(f"\n{report_title}")  # Use tqdm.write to avoid clashing with the progress bar
                    if current_global_scores_list:
                        mean_acc_global = np.array(current_global_scores_list).mean()
                        if accelerator.is_main_process:
                            print(f"Global samples processed: {total_samples_processed_globally} / {total_items}")
                            print(f"Current Global Mean Accuracy: {mean_acc_global:.4f}")
                            if pbar:
                                pbar.set_description(
                                    f"Global Acc: {mean_acc_global:.4f} ({total_samples_processed_globally}/{total_items})"
                                )
                    else:
                        if accelerator.is_main_process:
                            print(
                                f"No scores to report globally yet (Total processed: {total_samples_processed_globally})."
                            )

                accelerator.wait_for_everyone()  # Sync again after reporting in case some processes move ahead faster

        if pbar:
            pbar.close()

        # Final metrics have already been printed in the last report
        # (when is_last_local_batch is True)
        if accelerator.is_main_process and len(eval_datasets_local) == 0 and total_items > 0:
            print(
                "Main process had no data, but other processes might have. "
                "Final global metrics are printed by the last reporting sync."
            )
        elif accelerator.is_main_process and total_items == 0:
            print("No data was prepared for evaluation. Nothing to report.")

else:
    if accelerator.is_main_process:
        print(f"Task '{task}' is not configured for batched evaluation in this script.")
