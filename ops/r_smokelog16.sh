G=/tmp2/mzjiang_usersim/grpo_planner
grep -v "Loading weights" $G/runs/v16_prelim/smoke.log | tail -30 | cut -c1-300
echo "--- prelim"; tail -8 $G/run_v16_prelim.log | cut -c1-200
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
pgrep -u mzjiang -af "vllm|train_planner|smoke|holder" | cut -c1-100
