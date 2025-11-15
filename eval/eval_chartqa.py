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
model_args = {}  # Use {"torch_dtype":torch.bfloat16} if desired and supported

## 这个是DyME后的llava，使用/path/to/data/chartqa_output/json/train_new_prerefine.json 这个数据
model_id = '/path/to/code/DyME/output/test/checkpoint-2900'
model_id = '/path/to/code/DyME/output-dist/test/final_checkpoint'

## 这个是sft后的llava，使用/path/to//data/chartqa_output/json/train_new_prerefine.json 这个数据
# model_id = '/path/to//sft-llavaov-chart/checkpoint-400'

model_id = '/path/to/sft-llavaov-chart-llava_cot/checkpoint-1203'
model_id = '/path/to/grpo-chart-llava/checkpoint-144'

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
    if 'longest_edge' in processor.image_processor.size:
        print('Setting image processor longest_edge to 2048')
        processor.image_processor.size['longest_edge'] = 512 * 4
    processor.tokenizer.padding_side = 'left'
else:
    print(
        f"Warning: Could not directly set 'longest_edge' via dict. Current image processor size config: {processor.image_processor.size}")
    # Attempt an alternative if applicable, e.g.
    # processor.image_processor.size = {"longest_edge": 512 * 4} # if size itself can be replaced
    # Or this might indicate that `size` is a single value or a different structure.

PROMPT_TEMPLATE = (
    "Your task is to answer the question below. "
    "Give step by step thinking before you answer, and when you're ready to answer, "
    "please use the format \"Answer: ..\"\n\n"
    "Question:\n\n{question}"
)

# PROMPT_TEMPLATE = (
#     "{question}"
# )

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
    # inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
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
task = 'chart'
# dt_record_local 在 if task == 'chart' 块内初始化

if task == 'chart':
    dt_record_local = {}  # 为当前进程存储结果
    if accelerator.is_main_process:
        print("Loading ChartQA dataset...")
    try:
        full_dataset = load_dataset("HuggingFaceM4/ChartQA", trust_remote_code=True)['test']
    except Exception as e:
        if accelerator.is_main_process:
            print(f"Failed to load dataset directly. Error: {e}")
            print("Attempting to load with specific revision if applicable, or check path/connection.")
        # 作为示例，您可以尝试特定版本（如果知道）或确保路径和网络连接正确
        # full_dataset = load_dataset("HuggingFaceM4/ChartQA", revision="main", trust_remote_code=True)['test']
        raise  # 重新抛出异常，因为没有数据集无法继续

    # full_dataset = full_dataset.select(range(80)) # 取消注释以进行快速测试

    eval_datasets_all_prepared = []
    # chart_instructions_prefix = (
    #         "For the question below, follow the following instructions:\n"
    #         # ... (您的详细指令) ...
    #         + "-Try to include the full label from the graph when asked about an entity.\n"
    #         + "Question: "
    # )

    for d_item in tqdm(full_dataset, desc="Preparing dataset", disable=not accelerator.is_main_process):
        image_path = d_item['image']
        raw_question = d_item['query']
        answer_list = d_item.get('label')  # 使用 .get() 以防 'label' 字段不存在
        if not answer_list:  # 如果 'label' 不存在或为空列表
            if accelerator.is_main_process:
                tqdm.write(f"Warning: Item missing 'label' or 'label' is empty. Query: {raw_question[:50]}...")
            # 根据您的需求决定如何处理：跳过此样本或使用默认答案
            continue  # 跳过此样本
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

        BATCH_SIZE = 32  # 根据您的 VRAM 调整
        REPORT_INTERVAL_BATCHES = 1  # 每处理 N 个本地批次后报告一次（主进程将打印全局统计）

        # if accelerator.is_main_process:
        #     print(f"Total items for evaluation: {total_items}")
        #     print(f"Process {process_index} handling {len(eval_datasets_local)} items.")
        #     print(f"Batch size per process: {BATCH_SIZE}, Reporting interval: {REPORT_INTERVAL_BATCHES} local batches.")

        pbar = None
        if accelerator.is_main_process and len(eval_datasets_local) > 0:  # 仅当有数据时创建 pbar
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
                    parsed_pred_answer = full_pred_text  # 如果解析答案为空，则回退到完整预测

                score = eval_one_chart(parsed_pred_answer, ground_truth_answer)  # nlp 对象是全局的
                dt_record_local['res'].append(score)

                # (可选) 主进程打印少量样本的预测详情
                if accelerator.is_main_process:
                    print(full_pred_text, "######", ground_truth_answer, "######", score)

            if pbar:
                pbar.update(len(current_batch_list))

            # --- 中间报告逻辑 ---
            is_last_local_batch = (batch_idx_local == num_local_batches - 1)
            # 每隔 REPORT_INTERVAL_BATCHES 个本地批次，或在当前进程的最后一个本地批次时，执行同步和报告
            should_sync_and_report = ((batch_idx_local + 1) % REPORT_INTERVAL_BATCHES == 0) or is_last_local_batch

            # 确保即使 REPORT_INTERVAL_BATCHES 为1，也不会在没有数据时报告 (例如 len(eval_datasets_local) == 0)
            if len(eval_datasets_local) == 0:  # 如果当前进程没有数据，则不参与报告逻辑
                should_sync_and_report = False  # 除非它是最后一个批次（此时 num_local_batches 为0，循环不会运行）
                # 但如果 num_local_batches > 0, 此检查确保仅在有数据时报告

            if num_local_batches == 0 and is_last_local_batch:  # 特殊情况：进程无数据，但仍需参与最终同步
                should_sync_and_report = True

            if should_sync_and_report:
                accelerator.wait_for_everyone()  # 等待所有进程到达同步点

                gathered_all_processes_data = [None] * num_processes
                # 每个进程发送其*当前累积*的 dt_record_local
                # 如果某进程没有数据，dt_record_local['res'] 是空列表，这是正常的
                all_gather_object(gathered_all_processes_data, dt_record_local)

                if accelerator.is_main_process:
                    current_global_scores_list = []
                    for process_data_dict in gathered_all_processes_data:
                        if process_data_dict and 'res' in process_data_dict:
                            current_global_scores_list.extend(process_data_dict['res'])

                    total_samples_processed_globally = len(current_global_scores_list)

                    report_title = "--- Intermediate Report ---"
                    # 检查这是否是所有进程都已完成的最终报告点
                    # 一个简单的启发式方法是：如果这是主进程的最后一个批次，并且所有收集到的样本数等于总样本数
                    if is_last_local_batch and total_samples_processed_globally == total_items:
                        report_title = "--- Final Report ---"
                    elif is_last_local_batch:  # 主进程的最后一个批次，但可能并非所有样本都已处理（如果其他进程较慢/数据更多）
                        report_title = f"--- Report (Main Proc Last Batch, {batch_idx_local + 1}/{num_local_batches}) ---"

                    tqdm.write(f"\n{report_title}")  # 使用 tqdm.write 避免与进度条冲突
                    if current_global_scores_list:
                        mean_acc_global = np.array(current_global_scores_list).mean()
                        if accelerator.is_main_process:
                            print(f"Global samples processed: {total_samples_processed_globally} / {total_items}")
                            print(f"Current Global Mean Accuracy: {mean_acc_global:.4f}")
                            if pbar:
                                pbar.set_description(
                                    f"Global Acc: {mean_acc_global:.4f} ({total_samples_processed_globally}/{total_items})")
                    else:
                        if accelerator.is_main_process:
                            print(
                                f"No scores to report globally yet (Total processed: {total_samples_processed_globally}).")

                accelerator.wait_for_everyone()  # 报告后再次同步，以防某些进程快速进入下一计算

        if pbar:
            pbar.close()

        # 最终的指标已在循环的最后一次报告中打印（当 is_last_local_batch 为 True 时）
        if accelerator.is_main_process and len(eval_datasets_local) == 0 and total_items > 0:
            print(
                f"Main process had no data, but other processes might have. Final global metrics are printed by the last reporting sync.")
        elif accelerator.is_main_process and total_items == 0:
            print("No data was prepared for evaluation. Nothing to report.")


else:
    if accelerator.is_main_process:
        print(f"Task '{task}' is not configured for batched evaluation in this script.")