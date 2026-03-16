import os
import torch

# ====== Model Configuration ======
MODEL_CONFIG = {
    "pretrained_model_path": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",  # two-stage grpo
    "use_flash_attention_2": True,
    "torch_dtype": "bfloat16",
}

# ====== Training Configuration ======
TRAINING_CONFIG = {
    "task": 'chart',
    "num_gpus": 8, 
    "num_client": 8,  

    "dyme_args": {
        "output_dir": '/path/to/dyme-llavaov-chart-change3',
        "logging_steps": 1,
        "num_generations": 4,  
        "max_completion_length": 200,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16, # norm 16; 20 for sft hybrid
        "num_train_epochs": 10,
        "learning_rate": 8e-5,
        "bf16": True,  
        "gradient_checkpointing": False,
        "ddp_find_unused_parameters": False,
        "max_grad_norm": 1.0,
        "save_strategy": "epoch",
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "beta": 0.0,  # GRPO specific
        "loss_type": 'grpo',  # GRPO specific
        "seed": 42,
    },
}

RL_CONFIG = {
    "answer_flag": "Answer:",
    "end_flag": "<|im_end|>"
}

# ====== Client Configuration for Reward Calculation ======
CLIENT_CONFIG = {
    "client_type": "openai", 
    "api_key": "none",  # 客户端主机
    "api_base": "http://127.0.0.1:%s/v1",  
    "timeout": 60,  
    "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
    "init_port": 23333, 
    "num_server": 8
}

# ====== Dataset Configuration ======
DATASET_CONFIG = {
    "train_dataset": "/path/to/data/chartqa_output/json/train_new_prerefine.json",
    "eval_dataset": "HuggingFaceM4/ChartQA",  
}

# ====== Full Configuration ======
CONFIG = {
    "model": MODEL_CONFIG,
    "training": TRAINING_CONFIG,
    "rl": RL_CONFIG,
    "client": CLIENT_CONFIG,
    "dataset": DATASET_CONFIG,
}

# Save configuration to a file for reference
def save_config(config, config_path="./config.json"):
    import json
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

# Example usage to save config
if __name__ == "__main__":
    save_config(CONFIG)

