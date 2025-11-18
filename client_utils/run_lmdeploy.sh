#!/bin/bash
MODEL_ID="Qwen/Qwen2.5-14B-Instruct-AWQ"

BASE_PORT=23333

NUM_GPUS=8

CACHE_MAX_ENTRY_COUNT=0.1



echo "Starting $NUM_GPUS lmdeploy server instances..."
for i in $(seq 0 $((NUM_GPUS - 1)))
do
    PORT=$((BASE_PORT + i))
    echo "--> Launching server on GPU $i, Port: $PORT"

    CUDA_VISIBLE_DEVICES=$i lmdeploy serve api_server $MODEL_ID \
        --server-name 0.0.0.0 \
        --server-port $PORT \
        --cache-max-entry-count $CACHE_MAX_ENTRY_COUNT &
    sleep 2
done

echo ""
echo "All $NUM_GPUS server instances have been launched in the background."
echo "You can check their status using: ps -ef | grep lmdeploy"
echo "API endpoints are available at http://<your_server_ip>:$BASE_PORT through http://<your_server_ip>:$((BASE_PORT + NUM_GPUS - 1))"