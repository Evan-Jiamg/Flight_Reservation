echo "before:"; pgrep -u mzjiang -af "train_planner_rl.py|run_v12_continue|run_v12_guard" | cut -c1-110
TP=$(pgrep -u mzjiang -f train_planner_rl.py | head -1)
pkill -u mzjiang -f run_v12_guard.sh
sleep 2
pkill -u mzjiang -f run_v12_continue.sh
sleep 5
echo "after:"; pgrep -u mzjiang -af "train_planner_rl.py|run_v12_continue|run_v12_guard" | cut -c1-110
echo "trainer $TP alive: $(kill -0 $TP 2>/dev/null && echo yes || echo NO)"
echo $TP > /tmp2/mzjiang_usersim/grpo_planner/detached_trainer_pid.txt
tail -2 /tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/train.log | cut -c1-200
