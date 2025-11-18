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
# ## --- LoRA 修改 Start --- ##
# 1. 引入 PEFT (Parameter-Efficient Fine-Tuning) 相关的库
from peft import LoraConfig, get_peft_model, TaskType
# ## --- LoRA 修改 End --- ##
from transformers import AutoProcessor, AutoModelForCausalLM
from trl import GRPOConfig
from config_llm import CONFIG  
from data_utils.commom_util import collate_fn, define_task_data_func, collate_fn_woI
from DyMETrainer_llm import DyMETrainer
from reward_utils.checker import RewardCalculator, RewardCalculatorLocal
from reward_utils.refiner import ContextRefiner, ContextRefinerLocal

# ## --- LoRA 修改 Start --- ##
# 2. (可选但推荐) 添加一个辅助函数来打印模型的可训练参数
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )
# ## --- LoRA 修改 End --- ##

def setup_accelerator_and_wandb(bf16) -> Accelerator:
    """
    Initializes Weights & Biases and the Hugging Face Accelerator.

    Returns:
        Accelerator: The configured accelerator instance.
    """
    wandb.login(key="YOUR_WANDB_API")
    if bf16:
        accelerator = Accelerator(mixed_precision="bf16", log_with="wandb")
    else:
        accelerator = Accelerator(log_with="wandb")
    return accelerator


# ## --- LoRA 修改 Start --- ##
# 3. 修改函数签名，接收 peft_config
def load_model_and_processor(model_config: Dict[str, Any], peft_config: Dict[str, Any]):
    """
    Loads the base model, applies LoRA configuration, and loads its processor.

    Args:
        model_config (Dict[str, Any]): Configuration dictionary for the model.
        peft_config (Dict[str, Any]): Configuration dictionary for PEFT (LoRA).

    Returns:
        Tuple[PeftModel, PreTrainedProcessor]: The loaded PEFT model and processor.
    """
    model_id = model_config['pretrained_model_path']

    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, model_config['torch_dtype']),
        attn_implementation='flash_attention_2' if model_config['use_flash_attention_2'] else 'sdpa',
        low_cpu_mem_usage=True,
    )

    processor = AutoProcessor.from_pretrained(model_id, padding_side='left')
    # 确保 tokenizer 和 model 的 padding side 设置一致
    processor._tokenizer.padding_side = "left"

    # 定义 LoRA 配置
    lora_config = peft_config

    # 使用 get_peft_model 将 LoRA 配置应用到基础模型上
    model = get_peft_model(base_model, lora_config)

    # 打印可训练参数，验证LoRA是否生效
    print("LoRA model created:")
    print_trainable_parameters(model)

    return model, processor
# ## --- LoRA 修改 End --- ##


def prepare_datasets(task: str, dataset_config: Dict[str, Any]) -> (Dataset, Dataset):
    """
    Prepares the training and evaluation datasets based on the specified task.
    (此函数无需修改)
    """
    data_func = define_task_data_func(task)
    train_data_list = data_func(json_path=dataset_config['train_dataset'])
    train_dataset = Dataset.from_list(train_data_list)

    if 'chart' in task:
        eval_dataset = load_dataset(dataset_config['eval_dataset'])['test']
    else:
        eval_dataset = None

    return train_dataset, eval_dataset


def main():
    """
    Main function to orchestrate the model training pipeline.
    """

    parser = argparse.ArgumentParser(description="Train a model using GRPO with LoRA.")

    parser.add_argument(
        '--config', type=str, default='norm',
        help="config file to use: 'norm' or 'llavacot'..."
    )
    args = parser.parse_args()
    config_select = args.config

    if config_select == 'norm':
        from config_llm import CONFIG

    # 1. Load Configurations
    model_config = CONFIG['model']
    training_config = CONFIG['training']
    rl_config = CONFIG['rl']
    client_config = CONFIG['client']
    dataset_config = CONFIG['dataset']
    # ## --- LoRA 修改 Start --- ##
    # 4. 从主配置中加载 peft_config
    # 确保你的 config_llm.py 文件中包含了 peft_config
    # 示例 'peft_config' 结构:
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    # ## --- LoRA 修改 End --- ##
    task = training_config['task']

    # 2. Setup Environment
    accelerator = setup_accelerator_and_wandb(bf16=training_config['dyme_args']['bf16'])
    device_id = accelerator.process_index

    # 3. Initialize Model and Processor
    # ## --- LoRA 修改 Start --- ##
    # 5. 将 peft_config 传递给模型加载函数
    model, processor = load_model_and_processor(model_config, peft_config)
    # ## --- LoRA 修改 End --- ##

    # 4. Prepare Datasets
    train_dataset, eval_dataset = prepare_datasets(task, dataset_config)

    # 5. Initialize Reward Calculator
    checker = RewardCalculatorLocal(rl_config, client_config.copy(), gpu_id=device_id)
    refiner = ContextRefinerLocal(rl_config, client_config.copy(), gpu_id=device_id)

    # 6. Define Training Arguments
    training_args = GRPOConfig(**training_config['dyme_args'])

    collate_fn_with_processor = partial(collate_fn_woI, processor=processor)

    # 7. Initialize the Trainer
    # Trainer 会自动处理 PeftModel，无需额外修改
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

    # 保存模型时，Trainer 会自动只保存 LoRA 适配器权重
    output_dir = training_args.output_dir
    output_dir = os.path.join(output_dir, "final_checkpoint")
    dyme_trainer.save_model(output_dir)

    if accelerator.is_main_process:
        # Processor 等非模型文件仍需手动保存
        processor.save_pretrained(output_dir)
        print(f"LoRA adapters and processor saved to {output_dir}")

if __name__ == "__main__":
    main()