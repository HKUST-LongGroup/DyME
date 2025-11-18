# train_grpo.py
"""
Main script for training a Llava-based model using the custom MyGRPOTrainer.

This script handles:
1. Configuration loading.
2. Initialization of Weights & Biases (wandb) and Hugging Face Accelerate.
3. Loading the model and processor.
4. Preparing the training and evaluation datasets.
5. Setting up and running the GPRO trainer.
"""
import argparse
import os
from functools import partial
from typing import Dict, Any

import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from trl import GRPOConfig

from config import CONFIG
from data_utils.commom_util import collate_fn, define_task_data_func
from DyMETrainer_change3 import DyMETrainer
from reward_utils.checker import RewardCalculator, RewardCalculatorLocal
from reward_utils.refiner import ContextRefiner, ContextRefinerLocal


def setup_accelerator_and_wandb(bf16) -> Accelerator:
    """
    Initializes Weights & Biases and the Hugging Face Accelerator.

    Returns:
        Accelerator: The configured accelerator instance.
    """
    # It's recommended to use environment variables for keys for better security.
    # e.g., os.environ.get("WANDB_API_KEY")
    wandb.login(key="YOUR_WANDB_API")
    if bf16:
        accelerator = Accelerator(mixed_precision="bf16", log_with="wandb")
    else:
        accelerator = Accelerator(log_with="wandb")
    return accelerator


def load_model_and_processor(model_config: Dict[str, Any]):
    """
    Loads the pre-trained vision-language model and its associated processor.

    Args:
        model_config (Dict[str, Any]): Configuration dictionary for the model.

    Returns:
        Tuple[LlavaOnevisionForConditionalGeneration, PreTrainedProcessor]: The loaded model and processor.
    """
    model_id = model_config['pretrained_model_path']

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, model_config['torch_dtype']),
        attn_implementation='flash_attention_2' if model_config['use_flash_attention_2'] else 'sdpa',
        low_cpu_mem_usage=True,
    )

    # Freeze the vision tower to save memory and computation
    model.base_model.vision_tower.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"

    return model, processor


def prepare_datasets(task: str, dataset_config: Dict[str, Any]) -> (Dataset, Dataset):
    """
    Prepares the training and evaluation datasets based on the specified task.

    Args:
        task (str): The name of the task (e.g., 'chartqa').
        dataset_config (Dict[str, Any]): Configuration for datasets.

    Returns:
        Tuple[Dataset, Dataset]: The training and evaluation datasets.
    """
    data_func = define_task_data_func(task)

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


def main():
    """
    Main function to orchestrate the model training pipeline.
    """

    parser = argparse.ArgumentParser(description="Train a Llava model using either SFT or GRPO.")

    parser.add_argument(
        '--config', type=str, default='norm',
        help="config file to use: 'norm' or 'llavacot'..."
    )
    args = parser.parse_args()
    config_select = args.config

    if config_select == 'norm':
        from config import CONFIG
    elif config_select == 'llavacot':
        from config_llavacot import CONFIG
    elif config_select == 'low':
        from config_low import CONFIG
    elif config_select == 'change':
        from config_change import CONFIG

    # 1. Load Configurations
    model_config = CONFIG['model']
    training_config = CONFIG['training']
    rl_config = CONFIG['rl']
    client_config = CONFIG['client']
    dataset_config = CONFIG['dataset']
    task = training_config['task']

    # 2. Setup Environment
    accelerator = setup_accelerator_and_wandb(bf16=training_config['dyme_args']['bf16'])
    device_id = accelerator.process_index

    # 3. Initialize Model and Processor
    model, processor = load_model_and_processor(model_config)

    # 4. Prepare Datasets
    train_dataset, eval_dataset = prepare_datasets(task, dataset_config)

    # 5. Initialize Reward Calculator
    # checker = RewardCalculator(rl_config, client_config.copy(), gpu_id=device_id)
    # refiner = ContextRefiner(rl_config, client_config.copy(), gpu_id=device_id)

    checker = RewardCalculatorLocal(rl_config, client_config.copy(), gpu_id=device_id)
    refiner = ContextRefinerLocal(rl_config, client_config.copy(), gpu_id=device_id)
    # 6. Define Training Arguments
    training_args = GRPOConfig(**training_config['dyme_args'])

    collate_fn_with_processor = partial(collate_fn, processor=processor)
    # 7. Initialize the Trainer
    dyme_trainer = DyMETrainer(
        model=model,
        checker=checker,
        refiner=refiner,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        processing_func=collate_fn_with_processor,
        task_name=task,
        end_flag=rl_config['end_flag'],
    )

    # 8. Start Training
    dyme_trainer.train()

    output_dir = training_args.output_dir
    output_dir = os.path.join(output_dir, "final_checkpoint")
    dyme_trainer.save_model(output_dir)
    # 确保只有主进程保存 processor 等非模型文件
    if accelerator.is_main_process:
        processor.save_pretrained(output_dir)
        print(f"Model and processor saved to {output_dir}")
if __name__ == "__main__":
    main()