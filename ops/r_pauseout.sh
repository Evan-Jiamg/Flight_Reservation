cat /tmp2/mzjiang_usersim/grpo_planner/pause_after_u5.log | cut -c1-240
pgrep -u mzjiang -af "run_v15_reselect|train_planner_rl.py --fold 2" | cut -c1-100
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
