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
    "num_gpus": 8,  # 使用的 GPU 数量
    "num_client": 8,  # 并发客户端数量，通常与 GPU 数量相同
    # RL阶段的参数 (根据原脚本的rl_args)
    "dyme_args": {
        "output_dir": '/apdcephfs_nj4/share_300377003/realzliu/dyme-llavaov-chart-change3',
        "logging_steps": 1,
        "num_generations": 4,  # norm 4; 5 for sft hybrid
        "max_completion_length": 300,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16, # norm 16; 20 for sft hybrid
        "num_train_epochs": 10,
        "learning_rate": 8e-5,
        "bf16": True,  # 使用 bf16 而不是 fp16
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
    "client_type": "openai",  # 客户端主机地址
    "api_key": "none",  # 客户端主机
    "api_base": "http://29.39.226.175:%s/v1",  # 客户端，如果是本地服务需要预留端口
    "timeout": 60,  # 请求超时时间
    "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",  # 使用的模型ID
    "init_port": 23333, # 或者none代表在线服务
    "num_server": 8
}

# ====== Dataset Configuration ======
DATASET_CONFIG = {
    # "train_dataset": "/chartqa_output/json/train_new_prerefine.json",  # 训练数据路径
    "train_dataset": "/apdcephfs_nj4/share_300377003/realzliu/data/chartqa_output/json/train_new_prerefine.json",
    # 训练数据路径
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

