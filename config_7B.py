import os
import torch

# ====== Model Configuration ======
MODEL_CONFIG = {
    "pretrained_model_path": "Qwen/Qwen2.5-VL-7B-Instruct",  
    "use_flash_attention_2": False,
    "torch_dtype": "bfloat16",
}

# ====== Training Configuration ======
TRAINING_CONFIG = {
    "task": 'chart',
    "num_gpus": 8,  
    "num_client": 8, 

    "dyme_args": {
        "output_dir": '/path/to/dyme-qwen25_7B-chart-llava_cot',
        "logging_steps": 1,
        "num_generations": 8,  
        "max_completion_length": 300,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 10,
        "learning_rate": 1e-5,
        "bf16": True, 
        "gradient_checkpointing": False,
        "ddp_find_unused_parameters": False,
        "max_grad_norm": 1.0,
        "save_steps": 100,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "eval_strategy": "steps",
        "eval_steps": 10000,
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
    "api_key": "none",  
    "api_base": "http://127.0.0.1:%s/v1",  
    "timeout": 60,  
    "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",  
    "init_port": 23333, 
    "num_server": 8
}

# ====== Dataset Configuration ======
DATASET_CONFIG = {

    "train_dataset": "/path/to/data/chartqa_output/llavacot/json/chartqa_train_processed.json", 
    "eval_dataset": "HuggingFaceM4/ChartQA",  
}


CONFIG = {
    "model": MODEL_CONFIG,
    "training": TRAINING_CONFIG,
    "rl": RL_CONFIG,
    "client": CLIENT_CONFIG,
    "dataset": DATASET_CONFIG,
}


def save_config(config, config_path="./config.json"):
    import json
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    save_config(CONFIG)

