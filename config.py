import os
import torch

# ====== Model Configuration ======
MODEL_CONFIG = {
    "pretrained_model_path": "llava-hf/llava-onevision-qwen2-0.5b-si-hf",  # 预训练模型路径
    "use_flash_attention_2": True,
    "torch_dtype": torch.bfloat16,
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
        "max_completion_length": 512,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
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
        "eval_steps": 100,
        "beta": 0.0,  # GRPO specific
        "loss_type": 'grpo',  # GRPO specific
        "seed": 42,
    }

}

RL_CONFIG = {
    "answer_flag": "answer:",  # 客户端主机地址
    "answer_template": "Answer: %s<|im_end|>",
}

# ====== Client Configuration for Reward Calculation ======
CLIENT_CONFIG = {
    "client_type": "openai",  # 客户端主机地址
    "api_key": "none",  # 客户端主机
    "api_base": "http://locpu2.cse.ust.hk:%s/v1",  # 客户端，如果是本地服务需要预留端口
    "timeout": 60,  # 请求超时时间
    "model_id": "gpt-4o-mini",  # 使用的模型ID
    "init_port": 3000 # 或者none代表在线服务
}

# ====== Dataset Configuration ======
DATASET_CONFIG = {
    "train_dataset": "json_path",  # 训练数据路径
    "eval_dataset": "HuggingFaceM4/ChartQA",  # 验证数据路径
    "batch_size": 8,  # 每次加载数据的批次大小
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
save_config(CONFIG)

