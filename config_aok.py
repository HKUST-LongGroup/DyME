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
    "task": 'world',
    "num_gpus": 8,  # 使用的 GPU 数量
    "num_client": 8,  # 并发客户端数量，通常与 GPU 数量相同
    # RL阶段的参数 (根据原脚本的rl_args)
    "dyme_args": {
        "output_dir": '/path/to/dyme-aok-online',
        "logging_steps": 1,
        "num_generations": 4,
        "max_completion_length": 300,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 10,
        "learning_rate": 1e-5,
        "bf16": True, 
        "gradient_checkpointing": False,
        "ddp_find_unused_parameters": False,
        "max_grad_norm": 1.0,
        "save_strategy": "epoch",
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "beta": 0.0,
        "loss_type": 'grpo', 
        "seed": 42,
    },
    "sft_args": {
        "output_dir": '/path/to/sft-aok',
        "logging_steps": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 10,
        "learning_rate": 1e-5,
        "bf16": True,  
        "gradient_checkpointing": False,
        "ddp_find_unused_parameters": False,
        "max_grad_norm": 1.0,
        "save_strategy": "epoch",
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "seed": 42,
        "remove_unused_columns": False
    },
    "grpo_args":{
        "output_dir": '/path/to/grpo-aok',
        "logging_steps": 1,
        "num_generations": 4,  
        "max_completion_length": 576,
        "max_prompt_length": None,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 10,
        "learning_rate": 1e-5,
        "bf16": True, 
        "gradient_checkpointing": False,
        "ddp_find_unused_parameters": False,
        "max_grad_norm": 1.0,
        "save_strategy": "epoch",
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "beta": 0.04,
        "loss_type": 'grpo',
        "seed": 42,
    }

}

RL_CONFIG = {
    "answer_flag": "Answer:",
    "end_flag": "<|im_end|>"
}

# ====== Client Configuration for Reward Calculation ======
CLIENT_CONFIG = {
    "client_type": "openai",  # 客户端主机地址
    "api_key": "none",  # 客户端主机
    "api_base": "http://127.0.0.1:%s/v1",  # 客户端，如果是本地服务需要预留端口
    "timeout": 60,  # 请求超时时间
    "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",  # 使用的模型ID
    "init_port": 23333, 
    "num_server": 8
}

# ====== Dataset Configuration ======
DATASET_CONFIG = {
    "train_dataset": "/path/to/data/aokvqa/json/train_cleaned.json",
    "eval_dataset": "HuggingFaceM4/A-OKVQA",  # 验证数据路径
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

