nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do echo "pid $p user $(ps -o user= -p $p 2>/dev/null)"; done
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
pgrep -u mzjiang -af "train_planner_rl|eval_test_rl|vllm serve|gpu_holder2|run_v1" | grep -v pgrep | cut -c1-80; echo "(none above = ours idle)"
