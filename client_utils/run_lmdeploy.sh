#!/bin/bash

# --- 可配置参数 ---

# 1. 要部署的模型ID
MODEL_ID="Qwen/Qwen2.5-14B-Instruct-AWQ"

# 2. 起始端口号
BASE_PORT=23333

# 3. 要使用的GPU数量
NUM_GPUS=8

# 4. KV Cache 占用显存的比例 (0.8 表示 80%)
# 对于AWQ模型，权重占用显存较少，可以多分一些给Cache以提高并发和长文本性能
CACHE_MAX_ENTRY_COUNT=0.1

# --- 脚本主逻辑 ---

echo "Starting $NUM_GPUS lmdeploy server instances..."

# 循环遍历 0 到 (NUM_GPUS - 1)
for i in $(seq 0 $((NUM_GPUS - 1)))
do
    # 计算当前实例的端口号
    PORT=$((BASE_PORT + i))

    # 使用 CUDA_VISIBLE_DEVICES 来指定当前实例使用的GPU
    # 注意：这里不再需要 --tp 参数，因为每个实例只使用一张卡
    echo "--> Launching server on GPU $i, Port: $PORT"

    CUDA_VISIBLE_DEVICES=$i lmdeploy serve api_server $MODEL_ID \
        --server-name 0.0.0.0 \
        --server-port $PORT \
        --cache-max-entry-count $CACHE_MAX_ENTRY_COUNT &

    # 短暂等待，避免同时初始化对系统造成太大压力（可选）
    sleep 2
done

echo ""
echo "All $NUM_GPUS server instances have been launched in the background."
echo "You can check their status using: ps -ef | grep lmdeploy"
echo "API endpoints are available at http://<your_server_ip>:$BASE_PORT through http://<your_server_ip>:$((BASE_PORT + NUM_GPUS - 1))"