# train_grpo.py
"""
Main script for training a Llava-based model using the custom MyGRPOTrainer.

This script handles:
1. Configuration loading.
2. Initialization of Weights & Biases (wandb) and Hugging Face Accelerate.
3. Loading the model and processor.
4. Preparing the training and evaluation datasets.
5. Setting up and running the GRPO trainer.
"""
import argparse
import os
from functools import partial
from typing import Dict, Any

import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoProcessor, AutoModelForCausalLM
from trl import GRPOConfig
from config.config_llm import CONFIG  
from data_utils.commom_util import collate_fn, define_task_data_func, collate_fn_woI
from trainer.DyMETrainer_llm import DyMETrainer
from reward_utils.checker import RewardCalculator, RewardCalculatorLocal
from reward_utils.refiner import ContextRefiner, ContextRefinerLocal

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

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, model_config['torch_dtype']),
        attn_implementation='flash_attention_2' if model_config['use_flash_attention_2'] else 'sdpa',
        low_cpu_mem_usage=True,
    )

    processor = AutoProcessor.from_pretrained(model_id, padding_side='left')
    processor._tokenizer.padding_side = "left"
    lora_config = peft_config

    model = get_peft_model(base_model, lora_config)

    print("LoRA model created:")
    print_trainable_parameters(model)

    return model, processor
# ## --- LoRA modification End --- ##


def prepare_datasets(task: str, dataset_config: Dict[str, Any]) -> (Dataset, Dataset):
    """
    Prepares the training and evaluation datasets based on the specified task.
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
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    task = training_config['task']

    # 2. Setup Environment
    accelerator = setup_accelerator_and_wandb(bf16=training_config['dyme_args']['bf16'])
    device_id = accelerator.process_index

    # 3. Initialize Model and Processor
    # ## --- LoRA modification Start --- ##
    #  Pass peft_config to the model loading function
    model, processor = load_model_and_processor(model_config, peft_config)
    # ## --- LoRA modification End --- ##

    # 4. Prepare Datasets
    train_dataset, eval_dataset = prepare_datasets(task, dataset_config)

    # 5. Initialize Reward Calculator
    checker = RewardCalculatorLocal(rl_config, client_config.copy(), gpu_id=device_id)
    refiner = ContextRefinerLocal(rl_config, client_config.copy(), gpu_id=device_id)

    # 6. Define Training Arguments
    training_args = GRPOConfig(**training_config['dyme_args'])

    collate_fn_with_processor = partial(collate_fn_woI, processor=processor)

    # 7. Initialize the Trainer
    # Trainer handles PeftModel automatically
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

    # When saving, the Trainer automatically saves only the LoRA adapter weights
    output_dir = training_args.output_dir
    output_dir = os.path.join(output_dir, "final_checkpoint")
    dyme_trainer.save_model(output_dir)

    if accelerator.is_main_process:
        # Non-model files like the processor still need to be saved manually
        processor.save_pretrained(output_dir)
        print(f"LoRA adapters and processor saved to {output_dir}")

if __name__ == "__main__":
    main()
