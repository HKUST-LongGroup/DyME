# train.py (优化和注释版)

import argparse
import os
from functools import partial
from typing import Dict, Any, Optional

import torch
import wandb
from accelerate import Accelerator, PartialState
from datasets import Dataset, load_dataset
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    TrainingArguments,
)
from trl import SFTConfig

# --- 假设您的模块存在 ---

from data_utils.chart.evaluator import eval_one_chart
from data_utils.commom_util import define_task_data_func, collate_fn
from reward_utils.compute_rewards import split_initial_context


# #####################################################################
# #                      通用设置函数                                 #
# #####################################################################

def setup_accelerator_and_wandb(bf16) -> Accelerator:
    """
    Initializes Weights & Biases and the Hugging Face Accelerator.

    Returns:
        Accelerator: The configured accelerator instance.
    """
    # It's recommended to use environment variables for keys for better security.
    # e.g., os.environ.get("WANDB_API_KEY")
    wandb.login(key="a07e39e43f1a318a12a9b43a73d79d6ad4f4d2e2")
    if bf16:
        accelerator = Accelerator(mixed_precision="bf16", log_with="wandb")
    else:
        accelerator = Accelerator(log_with="wandb")
    return accelerator

def load_model_and_processor(model_config: Dict[str, Any], training_mode: str):
    """
    加载预训练的VLM模型和处理器。
    """
    model_id = model_config['pretrained_model_path']

    # 当使用 accelerate launch 和 DeepSpeed ZeRO-3 时，
    # from_pretrained 会被自动拦截，以节省内存的方式加载模型。
    # 对于ZeRO-2，low_cpu_mem_usage=True 也是一个好习惯。
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, model_config['torch_dtype']),
        attn_implementation='flash_attention_2' if model_config.get('use_flash_attention_2', False) else 'sdpa',
        low_cpu_mem_usage=True,
    )

    processor = AutoProcessor.from_pretrained(model_id)

    # 根据训练模式设置 padding side，您的原始逻辑是完全正确的
    if training_mode == 'sft':
        processor.tokenizer.padding_side = "right"
    else:  # grpo
        processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
        processor.tokenizer.padding_side = "left"


    print(f"Model and processor loaded. Tokenizer padding side set to '{processor.tokenizer.padding_side}'.")
    model.base_model.vision_tower.requires_grad_(False)
    return model, processor


def prepare_datasets(task: str, dataset_config: Dict[str, Any], mode='rl') -> (Dataset, Dataset):
    """
    Prepares the training and evaluation datasets based on the specified task.

    Args:
        task (str): The name of the task (e.g., 'chartqa').
        dataset_config (Dict[str, Any]): Configuration for datasets.

    Returns:
        Tuple[Dataset, Dataset]: The training and evaluation datasets.
    """
    data_func = define_task_data_func(task, mode=mode)

    # Create training dataset
    train_data_list = data_func(json_path=dataset_config['train_dataset'])
    train_dataset = Dataset.from_list(train_data_list)

    # Create evaluation dataset
    if 'chart' in task:
        eval_dataset = load_dataset(dataset_config['eval_dataset'])['test']
        # Note: You can uncomment the line below for quick testing/debugging.
        # eval_dataset = eval_dataset.select(range(1000, 1100))
    else:
        # Extend this section for other tasks if needed in the future.
        raise NotImplementedError(f"Task '{task}' is not supported for evaluation in this script.")

    return train_dataset, eval_dataset

# #####################################################################
# #                         主执行函数                                 #
# #####################################################################

def main():
    parser = argparse.ArgumentParser(description="Train a Llava model using either SFT or GRPO.")
    parser.add_argument(
        '--mode', type=str, required=True, choices=['sft', 'grpo'],
        help="The training mode: 'sft' or 'grpo'."
    )
    parser.add_argument(
        '--config', type=str, default='norm',
        help="config file to use: 'norm' or 'llavacot'..."
    )
    args = parser.parse_args()
    training_mode = args.mode
    config_select = args.config

    if config_select == 'norm':
        from config import CONFIG
    elif config_select == 'llavacot':
        from config_llavacot import CONFIG
    elif config_select == 'low':
        from config_low import CONFIG

    # 1. 加载配置
    model_config = CONFIG['model']
    training_config = CONFIG['training']
    dataset_config = CONFIG['dataset']
    task = training_config['task']

    # 2. **在主进程上初始化WandB**
    # 注意：不再需要手动创建 Accelerator 实例。
    # accelerate launch 会自动处理环境，Trainer会利用它。
    accelerator = setup_accelerator_and_wandb(training_mode)
    device_id = accelerator.process_index
    # 3. 初始化模型和处理器
    model, processor = load_model_and_processor(model_config, training_mode=training_mode)
    print('rl' if training_mode == 'grpo' else 'sft')
    train_dataset, eval_dataset = prepare_datasets(task, dataset_config, mode='rl' if training_mode == 'grpo' else 'sft')
    trainer = None

    # 4. 根据模式选择并执行特定流程
    if training_mode == 'sft':
        from trl import SFTTrainer
        print("--- Running in SFT mode ---")

        # **关键点**: TrainingArguments 会自动识别 accelerate 的环境。
        # 您在 ds_config.json 中的配置会被自动应用。
        training_args_config = training_config.get('sft_args')
        if not training_args_config:
            raise ValueError("CONFIG file must contain 'sft_args' for SFT mode.")

        # 告诉Trainer使用wandb进行日志记录
        training_args_config['report_to'] = 'wandb'

        training_args = SFTConfig(**training_args_config)
        collate_fn_with_processor = partial(collate_fn, processor=processor)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
            data_collator=collate_fn_with_processor
        )

    elif training_mode == 'grpo':
        print("--- Running in GRPO mode ---")
        import re
        def format_reward(completions, **kwargs):
            """
            奖励函数，用于检查补全（completion）是否满足特定格式。

            该格式要求:
            1. 关键词 "answer:" (不区分大小写) 必须只出现一次。
            2. "answer:" 后面的内容，在去除前后空白字符后，其长度不能超过20个字符。
            """
            rewards = []
            for content in completions:
                # 将内容转换为小写，以便进行不区分大小写的检查
                lower_content = content.lower()

                # 条件 1: 检查 "answer:" 是否只出现一次
                if lower_content.count("answer:") == 1:
                    # 以 "answer:" 为分隔符来获取其后的内容
                    content_after = lower_content.split("answer:", 1)[1]

                    # 条件 2: 检查去除前后空格后的内容长度是否小于等于20
                    if len(content_after.strip()) <= 20:
                        rewards.append(1.0)  # 如果两个条件都满足，则奖励为 1.0
                    else:
                        rewards.append(0.0)  # 如果内容超长，则奖励为 0.0
                else:
                    rewards.append(0.0)  # 如果 "answer:" 出现0次或多于1次，则奖励为 0.0

            return rewards

        def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[
            Optional[float]]:
            """Reward function that checks if the completion matches the ground truth.
            - If both gold and prediction are parseable → use math verification.
            - If not parseable → compare as normalized text.
            """
            scores = []
            for i, completion in enumerate(completions):
                lower_content = completion.lower()
                score = 0
                # 条件 1: 检查 "answer:" 是否只出现一次
                if lower_content.count("answer:") == 1:
                    # 以 "answer:" 为分隔符来获取其后的内容
                    content_after = lower_content.split("answer:", 1)[1]

                    # 条件 2: 检查去除前后空格后的内容长度是否小于等于20
                    if len(content_after.strip()) <= 20:
                        _, parsed_pred_answer = split_initial_context(completion)
                        if not parsed_pred_answer.strip():
                            parsed_pred_answer = completion  # 如果解析答案为空，则回退到完整预测
                        score = eval_one_chart(parsed_pred_answer, solution[i])  # nlp 对象是全局的

                scores.append(score)
            return scores

        from trl import GRPOTrainer, GRPOConfig
        training_args_config = training_config.get('grpo_args')
        if not training_args_config:
            raise ValueError("CONFIG file must contain 'grpo_args' for GRPO mode.")

        training_args_config['report_to'] = 'wandb'

        training_args = GRPOConfig(**training_args_config)

        rename_map = {
            'image': 'image',  # 示例：'img' -> 'image'
            'prompt': 'problem',  # 示例：'q' -> 'problem'
            'answer': 'solution',  # 示例：'a' -> 'solution'
            'question_wo_prompt': 'original_question',  # 示例：'orig_q' -> 'original_question'
        }

        def copy_answer_column(example):
            """这个函数会复制 'answer' 列的内容到 'original_answer'"""
            example['original_answer'] = example['answer'].replace('answer:', "")
            image = example['image']
            if not os.path.exists(image):
                image = image.replace('/chartqa_output/',
                                      '/apdcephfs_nj4/share_300377003/realzliu/data/chartqa_output/')
                example['image'] = image

            return example

        # 使用 .map() 应用这个函数
        # 这会创建 'original_answer' 列
        train_dataset = train_dataset.map(copy_answer_column)

        train_dataset = train_dataset.rename_columns(rename_map)

        def make_conversation(example):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            return {
                "prompt": prompt,
                "image": example["image"],
            }

        train_dataset = train_dataset.map(make_conversation)


        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
            reward_funcs=[format_reward, accuracy_reward],
        )

    # 5. 开始训练
    if trainer:
        print(f"Starting training with {type(trainer).__name__}...")
        trainer.train()

        # 6. 保存模型
        print("Training finished. Saving model...")
        # Trainer 的 save_model 会自动处理分布式保存，确保只在主进程上
        # 保存完整的模型权重，或按照DeepSpeed的格式分片保存。
        output_dir = trainer.args.output_dir
        trainer.save_model(output_dir)

        # 确保只有主进程保存processor等非模型文件
        if PartialState().is_main_process:
            processor.save_pretrained(output_dir)

        print(f"Model and processor saved to {output_dir}")
    else:
        raise ValueError(f"Invalid training mode: {training_mode}")


if __name__ == "__main__":
    main()