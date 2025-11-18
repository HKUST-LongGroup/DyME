#!/bin/bash

# ... (保留您之前的配置变量) ...
REMOTE_USER="root"
PROJECT_PATH="/path/to/code/DyME"
#TRAIN_SCRIPT="main_rebuttal"
#SCRIPT_ARGS="--mode grpo --config llavacot"

TRAIN_SCRIPT="main"
SCRIPT_ARGS="--config norm"

#TRAIN_SCRIPT="main_change"
#SCRIPT_ARGS="--config change"
# 节点列表 (根据您的hostfile整理)
readonly WORKER_HOSTS=(
    "xx.xx.xx.xx"
)

# ... (保留您的环境变量设置 ENV_SETUP_CMDS) ...
ENV_SETUP_CMDS="
export WANDB_API_KEY=YOUR_WANDB_KEY;
export NCCL_IB_GID_INDEX=3;
export NCCL_IB_SL=3;
export NCCL_CHECKS_DISABLE=1;
export NCCL_P2P_DISABLE=0;
export NCCL_IB_DISABLE=0;
export NCCL_LL_THRESHOLD=16384;
export NCCL_IB_CUDA_SUPPORT=1;
export NCCL_SOCKET_IFNAME=bond1;
export UCX_NET_DEVICES=bond1;
export NCCL_IB_HCA='mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6';
export NCCL_COLLNET_ENABLE=0;
export SHARP_COLL_ENABLE_SAT=0;
export NCCL_NET_GDR_LEVEL=2;
export NCCL_IB_QPS_PER_CONNECTION=4;
export NCCL_IB_TC=160;
export NCCL_PXN_DISABLE=1;
export http_proxy='YOUR_PROXY';
export https_proxy='YOUR_PROXY';
"

CONFIG_FILE="${PROJECT_PATH}/multi_node_config_raw.yaml"

# ============================================================
# 核心修复 1: 定义清理函数
# ============================================================
cleanup() {
    echo "🛑 检测到中断或退出，正在清理所有节点进程..."
    bash kill_all.sh
    wait
    echo "✅ 清理完成。"
}


trap cleanup SIGINT EXIT


echo "🧹 启动前预清理..."
cleanup 

echo "环境安装"
pip install -r ${PROJECT_PATH}/requirements.txt &
LOCAL_PID=$! # 获取本地安装的进程ID

for HOST in "${WORKER_HOSTS[@]}"; do
    echo "在节点 ${HOST} 上启动安装..."
    # 使用 & 将 ssh 命令放入后台执行
    ssh ${REMOTE_USER}@${HOST} "pip install -r ${PROJECT_PATH}/requirements.txt;http_proxy='YOUR_PROXY' https_proxy='YOUR_PROXY' python -m spacy download en_core_web_sm" &
done

echo "等待所有安装任务完成..."
wait
echo "所有节点环境安装完成！"

echo "🚀 正在启动主节点 (rank 0)..."
cd ${PROJECT_PATH} || exit

MASTER_CMD="${ENV_SETUP_CMDS} accelerate launch \
    --config_file ${CONFIG_FILE} \
    --machine_rank 0 \
    -m ${TRAIN_SCRIPT} ${SCRIPT_ARGS}"

eval "${MASTER_CMD}" 2>&1 | tee master.log &
MASTER_PID=$! 

RANK=1
for HOST in "${WORKER_HOSTS[@]}"; do
    echo "🚀 正在启动从属节点 ${HOST} (rank ${RANK})..."
    REMOTE_CMD="cd ${PROJECT_PATH}; ${ENV_SETUP_CMDS} accelerate launch \
        --config_file ${CONFIG_FILE} \
        --machine_rank ${RANK} \
        -m ${TRAIN_SCRIPT} ${SCRIPT_ARGS}"

    ssh -n ${REMOTE_USER}@${HOST} "${REMOTE_CMD}" 2>&1 | tee worker_${RANK}.log &
    RANK=$((RANK+1))
done

echo "✅ 所有进程启动完毕。主进程 PID: $MASTER_PID"
echo "⏳ 等待训练结束... (按 Ctrl+C 可强制终止所有节点)"

wait $MASTER_PID