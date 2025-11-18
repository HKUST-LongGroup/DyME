#!/bin/bash

# 脚本在遇到错误时立即退出
set -e

# --- 用户配置区 ---
# 确保这个列表和你的 launch_all.sh 脚本完全一致
readonly WORKER_HOSTS=(
    "xx.xx.xx.xx"
)
# 您的登录用户名
readonly REMOTE_USER="root"
# 确保这个脚本名和你启动的脚本名完全一致
readonly TRAIN_SCRIPT="main"

# --- 脚本执行区 ---

# 1. 杀死本地可能存在的进程 (作为安全措施)
echo "--- Killing local processes matching '${TRAIN_SCRIPT}' first ---"
# 使用 pkill 并通过 `|| true` 忽略“找不到进程”的错误
pkill -9 -f "${TRAIN_SCRIPT}" || true
pkill -f python
echo "Local check complete."
echo

# 2. 并行杀死远程主机上的进程
echo "🛑 Sending targeted kill signal to processes matching '${TRAIN_SCRIPT}' on all remote hosts in parallel..."

for HOST in "${WORKER_HOSTS[@]}"; do
    # 将每个 SSH 连接放入后台 (&) 以实现并行执行
    (
        echo "--- Processing host: ${HOST} ---"
        # 远程命令被封装在一个多行字符串中，以提高可读性
        # 逻辑：精确查找 -> 报告 -> 杀死
        ssh -n "${REMOTE_USER}@${HOST}" "
            set -e # 远程脚本也应该在出错时停止
            pkill -f python
            # 精确查找由 python 启动的、且包含 TRAIN_SCRIPT 名称的进程
            # 这是为了避免误杀其他同名进程（比如一个名为 'main_rebuttal' 的shell脚本）
            PIDS=\$(pgrep -f \"python.*${TRAIN_SCRIPT}\")

            if [ -z \"\$PIDS\" ]; then
                echo '[INFO] ✅ No matching processes found on this host.'
            else
                echo '[WARN] 🔥 Found processes to kill:'
                # 在杀死前显示详细信息，增加安全性
                ps -fp \$PIDS
                echo '[KILL] Killing PIDs: '\$PIDS'...'
                kill -9 \$PIDS
                echo '[OK] ✅ Processes killed successfully.'
            fi
        "
        echo "--- Finished host: ${HOST} ---"
        echo
    ) &
done

# 等待所有在后台运行的 SSH 任务执行完毕
wait

echo "🎉 All hosts have been processed."