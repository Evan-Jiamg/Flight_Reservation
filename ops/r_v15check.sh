G=/tmp2/mzjiang_usersim/grpo_planner
pgrep -u mzjiang -af "run_v15_reselect|train_planner_rl.py --fold 2" | cut -c1-140
echo "--- v15 log"; cat $G/run_v15_reselect.log 2>&1 | cut -c1-200
ls -la $G/archive/pend_f2_v11_u25_20260927 | cut -c1-120
bash $G/watch.sh
