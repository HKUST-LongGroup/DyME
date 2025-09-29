import os

# ====== Model Configuration ======
MODEL_CONFIG = {
    "model_name": "llama-2-7b",  # 使用的模型名称（可以是任何 HuggingFace 模型）
    "pretrained_model_path": "path_to_pretrained_model",  # 预训练模型路径
    "checkpoint_path": "path_to_checkpoints",  # 模型保存的路径
    "num_labels": 2,  # 分类任务时的标签数
    "hidden_size": 4096,  # 模型的隐藏层大小
    "num_layers": 24,  # Transformer 的层数
    "attention_heads": 16,  # Attention heads 数量
    "dropout": 0.1,  # Dropout 比例
    "max_seq_length": 512,  # 输入序列的最大长度
}

# ====== Training Configuration ======
TRAINING_CONFIG = {
    "batch_size": 8,  # 每批次训练数据量
    "learning_rate": 5e-5,  # 学习率
    "num_epochs": 10,  # 训练周期数
    "warmup_steps": 1000,  # 学习率预热步骤
    "weight_decay": 0.01,  # 权重衰减
    "gradient_accumulation_steps": 2,  # 梯度累积的步数
    "logging_dir": "./logs",  # 日志保存路径
    "save_steps": 500,  # 每 500 步保存一次模型
    "eval_steps": 1000,  # 每 1000 步进行一次评估
    "fp16": True,  # 是否启用混合精度训练
    "seed": 42,  # 随机种子
    "task": 'chart',
    "num_gpus": 8,  # 使用的 GPU 数量
    "num_client": 8,  # 并发客户端数量，通常与 GPU 数量相同
}

RL_CONFIG = {
    "answer_flag": "answer:",  # 客户端主机地址
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
    "train_dataset": "path_to_train_data",  # 训练数据路径
    "eval_dataset": "path_to_eval_data",  # 验证数据路径
    "batch_size": 8,  # 每次加载数据的批次大小
}

# ====== Miscellaneous Settings ======
MISC_CONFIG = {
    "log_level": "INFO",  # 日志级别
    "use_wandb": True,  # 是否使用 Weights & Biases 进行实验记录
    "wandb_project_name": "mllm-training",  # W&B 项目名称
}

# ====== Full Configuration ======
CONFIG = {
    "model": MODEL_CONFIG,
    "training": TRAINING_CONFIG,
    "rl": RL_CONFIG,
    "client": CLIENT_CONFIG,
    "dataset": DATASET_CONFIG,
    "misc": MISC_CONFIG,
}

# Save configuration to a file for reference
def save_config(config, config_path="./config.json"):
    import json
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

# Example usage to save config
save_config(CONFIG)

