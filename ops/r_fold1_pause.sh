G=/tmp2/mzjiang_usersim/grpo_planner
# user 2026-09-28: pause fold 1 (focus on testing u5 vs u0 of fold 2). Only our own processes.
pkill -u mzjiang -f "run_v16_fold.sh 1"
pkill -u mzjiang -f "train_planner_rl.py --fold 1"
pkill -u mzjiang -f "start_servers5.sh"
for i in $(seq 1 45); do pgrep -u mzjiang -f "run_v16_folds01.sh" > /dev/null || break; sleep 2; done
echo "=== sequencer log"; cat $G/run_v16_folds.log
echo "=== fold 1 log"; tail -4 $G/run_v16_fold1.log
echo "=== run dir"; ls $G/runs/pend_f1_v16 2>&1 | head
pgrep -u mzjiang -af "run_v16|gpu_holder2|vllm serve|train_planner_rl|EngineCore" | grep -v pgrep | cut -c1-90; echo "(none above = all released)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
