#!/bin/bash
set -e

readonly WORKER_HOSTS=(
    "xx.xx.xx.xx"
)

readonly REMOTE_USER="root"

readonly TRAIN_SCRIPT="main"




echo "--- Killing local processes matching '${TRAIN_SCRIPT}' first ---"

pkill -9 -f "${TRAIN_SCRIPT}" || true
pkill -f python
echo "Local check complete."
echo


echo "🛑 Sending targeted kill signal to processes matching '${TRAIN_SCRIPT}' on all remote hosts in parallel..."

for HOST in "${WORKER_HOSTS[@]}"; do

    (
        echo "--- Processing host: ${HOST} ---"

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

wait

echo "🎉 All hosts have been processed."