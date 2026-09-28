G=/tmp2/mzjiang_usersim/grpo_planner; RUN=$G/runs/pend_f2_v16
echo "=== script log"; grep -vE "^\s*[\{\}\"]" $G/run_v16_reselect.log | tail -8 | cut -c1-200
echo "=== processes"; pgrep -u mzjiang -af "run_v16_reselect|train_planner_rl|gpu_holder2|vllm serve" | grep -v pgrep | cut -c1-80; echo "(none above = finished)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== verify"; grep -E "PIPELINE VERIFICATION|FAIL" $RUN/verify_reselect.txt 2>/dev/null | head -8 | cut -c1-200
echo "=== reselect_best"; cat $RUN/reselect_best.json 2>/dev/null
echo "=== bootstrap"; cat $RUN/reselect_boot.txt 2>/dev/null | cut -c1-200
bash $G/../grpo_planner/watch_v16.sh > /dev/null 2>&1
