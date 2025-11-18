import torch
from PIL import Image
from accelerate import Accelerator
# 确保此路径正确且实用程序可用。
from datasets import load_dataset
from torch.distributed import all_gather_object
from transformers import AutoProcessor, AutoConfig, AutoTokenizer, LlavaOnevisionForConditionalGeneration
from trl.models import unwrap_model_for_generation

from data_utils.aokvqa.evaluator import eval_aokvqa_direct
# 移除了 ChartQA 特定的评估器
# from data_utils.chart.evaluator import eval_one_chart
from reward_utils.compute_rewards import split_initial_context

accelerator = Accelerator()
from tqdm import tqdm
import numpy as np

DEVICE = accelerator.device

# Model and Processor Configuration
model_args = {}  # Use {"torch_dtype":torch.bfloat16} if desired and supported

# --- 您的模型 ID 列表保持不变 ---
# ... (model_id 定义) ...
model_id = 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf'
# model_id = '/path/to/dyme-aok-local/final_checkpoint'
# --- ---------------------- ---

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, config=config, trust_remote_code=True)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to(DEVICE)

model.eval()
processor = AutoProcessor.from_pretrained(model_id)

# Configure image processor size
if hasattr(processor.image_processor, 'size') and isinstance(processor.image_processor.size, dict):
    processor.tokenizer.padding_side = 'left'
else:
    print(
        f"Warning: Could not directly set 'longest_edge' via dict. Current image processor size config: {processor.image_processor.size}")

PROMPT_TEMPLATE = (
    "{question} Answer the question with a single word (or phrase)."
)


def run_kh_batch(batch_data_list):  # Renamed from run_kh, takes a batch
    batch_images = []
    batch_formatted_prompts_for_chat_template = []

    for item in batch_data_list:
        image_path = item['image_path']
        item_model_input_text = item['model_input_text'].strip()

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
        except:
            templated_prompt_str = f"USER: <image>\n{question_with_tags}\nASSISTANT:"
        batch_formatted_prompts_for_chat_template.append(templated_prompt_str)

    inputs = processor(
        text=batch_formatted_prompts_for_chat_template,
        images=batch_images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = {
        k: v.to(DEVICE).to(torch.bfloat16) if v.is_floating_point() else v.to(
            DEVICE)
        for k, v in inputs.items()
    }

    with unwrap_model_for_generation(model, accelerator) as unwrapped_model_instance:
        generated_ids = unwrapped_model_instance.generate(**inputs, max_new_tokens=1024, do_sample=False, )

    input_ids_length = inputs['input_ids'].shape[1]
    newly_generated_ids = generated_ids[:, input_ids_length:]

    generated_texts = processor.batch_decode(
        newly_generated_ids,
        skip_special_tokens=True,  # Special tokens like <eos> are removed. <image> might be too.
    )
    return [text.strip('.').strip() for text in generated_texts]



# --- Main Evaluation Logic ---
# --- 修改：将 task 更改为 'aokvqa' ---
task = 'aokvqa'
dt_record_local = {}  # 为当前进程存储结果

if task == 'aokvqa':
    if accelerator.is_main_process:
        # --- 修改：更新打印信息 ---
        print("Loading A-OKVQA dataset...")
    try:
        # --- 修改：加载 A-OKVQA 和 'validation' 拆分 ---
        full_dataset = load_dataset("HuggingFaceM4/A-OKVQA", trust_remote_code=True)['validation']
    except Exception as e:
        if accelerator.is_main_process:
            print(f"Failed to load dataset directly. Error: {e}")
            print("Attempting to load with specific revision if applicable, or check path/connection.")
        raise

    # full_dataset = full_dataset.select(range(80)) # 取消注释以进行快速测试

    eval_datasets_all_prepared = []

    # ChartQA 特定的指令前缀已被移除

    for d_item in tqdm(full_dataset, desc="Preparing dataset", disable=not accelerator.is_main_process):
        image_path = d_item['image']
        # --- 修改：'query' -> 'question' ---
        raw_question = d_item['question']
        # --- 修改：'label' -> 'direct_answers' ---
        ground_truth_answers = d_item.get('direct_answers')

        if not ground_truth_answers:  # 如果 'direct_answers' 不存在或为空列表
            if accelerator.is_main_process:
                # --- 修改：更新警告信息 ---
                tqdm.write(
                    f"Warning: Item missing 'direct_answers' or 'direct_answers' is empty. Question: {raw_question[:50]}...")
            continue  # 跳过此样本

        model_input_text_for_template = raw_question
        eval_datasets_all_prepared.append({
            'image_path': image_path,
            'model_input_text': model_input_text_for_template,
            # --- 修改：存储整个答案列表 ---
            'direct_answers_list': ground_truth_answers,
            'original_question': raw_question
        })

    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    total_items = len(eval_datasets_all_prepared)

    if total_items == 0:
        if accelerator.is_main_process:
            # --- 修改：更新退出信息 ---
            print("No data prepared for evaluation after filtering. Exiting A-OKVQA evaluation.")
    else:
        # --- (分布式采样的逻辑保持不变) ---
        items_per_proc = total_items // num_processes
        extra_items = total_items % num_processes
        local_start_index = process_index * items_per_proc + min(process_index, extra_items)
        num_local_items = items_per_proc + (1 if process_index < extra_items else 0)
        local_end_index = local_start_index + num_local_items
        eval_datasets_local = eval_datasets_all_prepared[local_start_index:local_end_index]

        BATCH_SIZE = 32  # 根据您的 VRAM 调整
        REPORT_INTERVAL_BATCHES = 1

        pbar = None
        if accelerator.is_main_process and len(eval_datasets_local) > 0:
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
                # --- 修改：获取 'direct_answers_list' ---
                ground_truth_answers_list = eval(original_item['direct_answers_list'])

                _, parsed_pred_answer = split_initial_context(full_pred_text)
                if not parsed_pred_answer.strip():
                    parsed_pred_answer = full_pred_text  # 回退

                # --- 修改：使用新的评估函数 ---
                score = eval_aokvqa_direct(parsed_pred_answer, ground_truth_answers_list)
                dt_record_local['res'].append(score)

                if accelerator.is_main_process:
                    # --- 修改：打印真实答案列表 ---
                    print(parsed_pred_answer, "######", ground_truth_answers_list, "######", score)

            if pbar:
                pbar.update(len(current_batch_list))

            # --- (中间报告逻辑保持不变) ---
            is_last_local_batch = (batch_idx_local == num_local_batches - 1)
            should_sync_and_report = ((batch_idx_local + 1) % REPORT_INTERVAL_BATCHES == 0) or is_last_local_batch

            if len(eval_datasets_local) == 0:
                should_sync_and_report = False

            if num_local_batches == 0 and is_last_local_batch:
                should_sync_and_report = True

            if should_sync_and_report:
                accelerator.wait_for_everyone()

                gathered_all_processes_data = [None] * num_processes
                all_gather_object(gathered_all_processes_data, dt_record_local)

                if accelerator.is_main_process:
                    current_global_scores_list = []
                    for process_data_dict in gathered_all_processes_data:
                        if process_data_dict and 'res' in process_data_dict:
                            current_global_scores_list.extend(process_data_dict['res'])

                    total_samples_processed_globally = len(current_global_scores_list)

                    report_title = "--- Intermediate Report ---"
                    if is_last_local_batch and total_samples_processed_globally == total_items:
                        report_title = "--- Final Report ---"
                    elif is_last_local_batch:
                        report_title = f"--- Report (Main Proc Last Batch, {batch_idx_local + 1}/{num_local_batches}) ---"

                    tqdm.write(f"\n{report_title}")
                    if current_global_scores_list:
                        # 'res' 列表现在是 0 和 1，均值即为准确率
                        mean_acc_global = np.array(current_global_scores_list).mean()
                        if accelerator.is_main_process:
                            print(f"Global samples processed: {total_samples_processed_globally} / {total_items}")
                            print(f"Current Global Mean Accuracy (VQA Acc): {mean_acc_global:.4f}")  # 标签更新为 VQA Acc
                            if pbar:
                                pbar.set_description(
                                    f"Global Acc: {mean_acc_global:.4f} ({total_samples_processed_globally}/{total_items})")
                    else:
                        if accelerator.is_main_process:
                            print(
                                f"No scores to report globally yet (Total processed: {total_samples_processed_globally}).")

                accelerator.wait_for_everyone()

        if pbar:
            pbar.close()

        # (其余报告逻辑保持不变)
        if accelerator.is_main_process and len(eval_datasets_local) == 0 and total_items > 0:
            print(
                f"Main process had no data, but other processes might have. Final global metrics are printed by the last reporting sync.")
        elif accelerator.is_main_process and total_items == 0:
            print("No data was prepared for evaluation. Nothing to report.")


else:
    if accelerator.is_main_process:
        print(f"Task '{task}' is not configured for batched evaluation in this script.")