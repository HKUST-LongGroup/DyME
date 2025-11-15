import os
import torch

# ====== Model Configuration ======
MODEL_CONFIG = {
    "pretrained_model_path": "HuggingFaceTB/SmolVLM-500M-Instruct",  # 预训练模型路径
    "use_flash_attention_2": True,
    "torch_dtype": "bfloat16",
}

# ====== Training Configuration ======
TRAINING_CONFIG = {
    "task": 'chart',
    "num_gpus": 8,  # 使用的 GPU 数量
    "num_client": 8,  # 并发客户端数量，通常与 GPU 数量相同
    # RL阶段的参数 (根据原脚本的rl_args)
    "dyme_args": {
        "output_dir": os.path.join('output', "test"),
        "logging_steps": 1,
        "num_generations": 4,  # RL 阶段可以生成多个响应进行比较
        "max_completion_length": 300,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 10,
        "learning_rate": 8e-5,
        "bf16": True,  # 使用 bf16 而不是 fp16
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
    "sft_args": {
        "output_dir": '/path/to/sft-llavaov-chart',
        "logging_steps": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 10,
        "learning_rate": 1e-5,
        "bf16": True,  # 使用 bf16 而不是 fp16
        "gradient_checkpointing": False,
        "ddp_find_unused_parameters": False,
        "max_grad_norm": 1.0,
        "save_steps": 100,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "eval_strategy": "steps",
        "eval_steps": 10000,
        "seed": 42,
        "remove_unused_columns": False
    }

}

RL_CONFIG = {
    "answer_flag": "Answer:",
    "end_flag": "<end_of_utterance>"
}

# ====== Client Configuration for Reward Calculation ======
CLIENT_CONFIG = {
    "client_type": "openai",  # 客户端主机地址
    "api_key": "none",  # 客户端主机
    "api_base": "http://127.0.0.1:%s/v1",  # 客户端，如果是本地服务需要预留端口
    "timeout": 60,  # 请求超时时间
    "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",  # 使用的模型ID
    "init_port": 23333, # 或者none代表在线服务
    "num_server": 8
}

# ====== Dataset Configuration ======
DATASET_CONFIG = {
    # "train_dataset": "/chartqa_output/json/train_new_prerefine.json",  # 训练数据路径
    "train_dataset": "/path/to/data/chartqa_output/json/train_new_prerefine.json",  # 训练数据路径
    "eval_dataset": "HuggingFaceM4/ChartQA",  # 验证数据路径
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

