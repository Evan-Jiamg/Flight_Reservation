G=/tmp2/mzjiang_usersim/grpo_planner
setsid nohup bash $G/run_v17_smoke.sh 2 > $G/run_v17_smoke.log 2>&1 < /dev/null &
sleep 30
echo "=== smoke launcher log"; cat $G/run_v17_smoke.log
pgrep -u mzjiang -af "run_v17_smoke|gpu_holder2|vllm serve|smoke_v17" | grep -v pgrep | cut -c1-90
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
