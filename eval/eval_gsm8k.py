import torch
import re  # 导入正则表达式库
from accelerate import Accelerator
from datasets import load_dataset
from torch.distributed import all_gather_object
# 切换到 CausalLM 模型
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from trl.models import unwrap_model_for_generation
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

# --- 辅助函数：用于解析 GSM8K 答案 ---

def parse_ground_truth(answer_str):
    """从 '...#### 123' 格式的字符串中提取 '123'"""
    try:
        return answer_str.split('####')[-1].strip().replace(",", "")
    except:
        return ""


def parse_prediction(pred_str):
    """从模型的生成文本中提取最后一个数字作为答案"""

    # 优先尝试寻找 "Answer:" 标记
    answer_marker = "Answer:"
    if answer_marker in pred_str:
        pred_str = pred_str.split(answer_marker)[-1]

    # 移除千位分隔符
    pred_str = pred_str.replace(",", "")

    # 查找所有数字 (包括整数和小数)
    # 这个正则表达式匹配：可选的-或+，后跟数字，可选的小数点和更多数字
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", pred_str)

    if matches:
        # 返回最后一个找到的数字
        return matches[-1].strip()
    else:
        # 如果没有找到数字，返回空字符串
        return ""


# --------------------------------------

accelerator = Accelerator()
DEVICE = accelerator.device

# --- 模型和 Tokenizer 配置 ---
model_args = {"torch_dtype": torch.bfloat16}  # 保持 bf16 以获得高性能

# 切换为 Qwen 2.5 0.5B Instruct
# model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
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

# 为批量生成设置 Tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# 移除 AutoProcessor 和图像处理器配置

# --- Prompt 模板 ---
# 这个 CoT 模板非常适合 GSM8K
PROMPT_TEMPLATE = (
    "Your task is to answer the question below. "
    "Give step by step thinking before you answer, and when you're ready to answer, "
    "please use the format \"Answer: ..\"\n\n"
    "Question:\n\n{question}"
)


def run_model_batch(batch_data_list):  # 移除了图像处理
    batch_formatted_prompts_for_chat_template = []

    for item in batch_data_list:
        item_model_input_text = item['model_input_text'].strip()

        # 使用模板格式化问题
        question_with_tags = PROMPT_TEMPLATE.format(question=item_model_input_text)

        # 为 Qwen 构建 chat template 输入
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {
                "role": "user",
                "content": question_with_tags
            },
        ]

        try:
            # Tokenize=False 以便批量处理
            templated_prompt_str = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        except:
            # 一个适用于 Qwen 的回退
            templated_prompt_str = f"<|im_start|>user\n{question_with_tags}<|im_end|>\n<|im_start|>assistant\n"

        batch_formatted_prompts_for_chat_template.append(templated_prompt_str)

    # 批量 Tokenize
    inputs = tokenizer(
        batch_formatted_prompts_for_chat_template,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048  # 为 Qwen 设置一个合理的最大长度
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with unwrap_model_for_generation(model, accelerator) as unwrapped_model_instance:
        generated_ids = unwrapped_model_instance.generate(
            **inputs,
            max_new_tokens=1024,  # GSM8K 答案可能需要一些 CoT 空间
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id  # 明确指定 pad_token_id
        )

    input_ids_length = inputs['input_ids'].shape[1]
    newly_generated_ids = generated_ids[:, input_ids_length:]

    generated_texts = tokenizer.batch_decode(
        newly_generated_ids,
        skip_special_tokens=True,
    )
    return [text.strip() for text in generated_texts]


# --- Main 评估逻辑 ---
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

    # full_dataset = full_dataset.select(range(80)) # 取消注释以进行快速测试

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

        # 1. 创建 DataLoader
        # collate_fn=lambda x: x 使得 DataLoader 直接输出一个 item 列表的批次
        eval_dataloader = DataLoader(eval_datasets_all_prepared, batch_size=BATCH_SIZE, collate_fn=lambda x: x)

        # 2. 将模型和 DataLoader 交给 accelerator 准备
        # accelerator 会自动处理 DistributedSampler
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

        # 用于存储当前进程处理的结果
        local_scores = []

        pbar = None
        if accelerator.is_main_process:
            pbar = tqdm(total=len(eval_dataloader), desc=f"Eval on Main Proc", dynamic_ncols=True)

        # --- 3. 数据处理循环 ---
        # 每个进程只处理自己的数据子集，不再需要手动分片
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

                # 调试时可以只在主进程打印，避免输出混乱
                if accelerator.is_main_process:
                    tqdm.write("-" * 20)
                    tqdm.write(f"Q: {original_item['original_question'][:50]}...")
                    tqdm.write(f"PRED: [{pred_answer_clean}] | GT: [{gt_answer_clean}] | Score: {score}")

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        # --- 4. 同步和报告阶段 ---
        # 等待所有进程都完成上面的循环
        accelerator.wait_for_everyone()

        # 现在可以安全地收集所有进程的结果了
        # 每个进程都会创建一个 gathered_scores 列表，里面是所有进程的 local_scores
        gathered_scores_list_of_lists = [None] * accelerator.num_processes
        all_gather_object(gathered_scores_list_of_lists, local_scores)

        # 只有主进程负责计算和打印最终报告
        if accelerator.is_main_process:
            print("\n--- Final Report ---")

            # 将所有进程的结果展平成一个列表
            final_scores = [score for sublist in gathered_scores_list_of_lists for score in sublist]

            total_samples_processed = len(final_scores)

            if total_samples_processed > 0:
                final_accuracy = np.array(final_scores).mean()
                print(f"Global samples processed: {total_samples_processed} / {total_items}")
                # 注意：由于 DistributedSampler 可能会为了均匀分配而丢弃或复制样本，
                # processed 数量可能与 total_items 不完全相等。
                print(f"Final Global Mean Accuracy (EM): {final_accuracy:.4f}")
            else:
                print("No scores were gathered from any process.")

else:
    if accelerator.is_main_process:
        print(f"Task '{task}' is not configured for batched evaluation in this script.")