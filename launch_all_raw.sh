#!/bin/bash

# ... (保留您之前的配置变量) ...
REMOTE_USER="root"
PROJECT_PATH="/apdcephfs_qy4/share_302593112/realzliu/code/DyME"
TRAIN_SCRIPT="main_rebuttal"
SCRIPT_ARGS="--mode grpo"

# 节点列表 (根据您的hostfile整理)
WORKER_HOSTS=(
    "30.203.137.220"
    "30.203.130.57"
    "30.203.133.39"
    "30.203.136.188"
    "30.203.129.144"
    "30.203.128.24"
    "30.203.129.237"
)

# ... (保留您的环境变量设置 ENV_SETUP_CMDS) ...
ENV_SETUP_CMDS="
export WANDB_API_KEY=a07e39e43f1a318a12a9b43a73d79d6ad4f4d2e2;
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
export http_proxy='http://9.21.0.122:11113';
export https_proxy='http://9.21.0.122:11113';
"

CONFIG_FILE="${PROJECT_PATH}/multi_node_config_raw.yaml"

# ============================================================
# 核心修复 1: 定义清理函数
# ============================================================
cleanup() {
    echo "🛑 检测到中断或退出，正在清理所有节点进程..."
    # 1. 清理本机
    bash kill_all.sh

#    # 2. 清理远程
#    for HOST in "${WORKER_HOSTS[@]}"; do
#        ssh -o ConnectTimeout=5 ${REMOTE_USER}@${HOST} "bash kill_all.sh" &
#    done
    wait
    echo "✅ 清理完成。"
}

# 注册信号捕获：当收到 Ctrl+C (SIGINT) 或脚本退出时，自动运行 cleanup
trap cleanup SIGINT EXIT

# ============================================================
# 核心修复 2: 启动前强制清理一次 (解决端口占用)
# ============================================================
echo "🧹 启动前预清理..."
cleanup # 调用上面的函数先清理一遍

# 启动前清理旧进程 (可选，但强烈建议)
echo "环境安装"
# 1. 本地安装（可以和远程并行，也可以先做）
pip install -r ${PROJECT_PATH}/requirements.txt &
LOCAL_PID=$! # 获取本地安装的进程ID

# 2. 循环启动所有远程节点的安装任务，并放入后台
for HOST in "${WORKER_HOSTS[@]}"; do
    echo "在节点 ${HOST} 上启动安装..."
    # 使用 & 将 ssh 命令放入后台执行
    ssh ${REMOTE_USER}@${HOST} "pip install -r ${PROJECT_PATH}/requirements.txt;http_proxy='http://9.21.0.122:11113' https_proxy='http://9.21.0.122:11113' python -m spacy download en_core_web_sm" &
done

#;http_proxy='http://9.21.0.122:11113' https_proxy='http://9.21.0.122:11113' python -m spacy download en_core_web_sm

# 3. 等待所有后台任务（包括本地和所有远程的）全部完成
echo "等待所有安装任务完成..."
wait
echo "所有节点环境安装完成！"

# ============================================================
# 启动训练
# ============================================================

# 步骤 A: 启动主节点
echo "🚀 正在启动主节点 (rank 0)..."
cd ${PROJECT_PATH} || exit

MASTER_CMD="${ENV_SETUP_CMDS} accelerate launch \
    --config_file ${CONFIG_FILE} \
    --machine_rank 0 \
    -m ${TRAIN_SCRIPT} ${SCRIPT_ARGS}"

# 记录日志以便排查
eval "${MASTER_CMD}" 2>&1 | tee master.log &
MASTER_PID=$! # 记录主进程PID

# 步骤 B: 启动从节点
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

# ============================================================
# 核心修复 3: 正确等待
# ============================================================
# 等待主进程结束。如果主进程挂了，脚本也会继续向下执行从而触发 cleanup
wait $MASTER_PID