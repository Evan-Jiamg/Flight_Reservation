grep -q "PAUSE DONE" /tmp2/mzjiang_usersim/grpo_planner/pause_after_u5.log && echo DONE || echo "WAIT $(tail -1 /tmp2/mzjiang_usersim/grpo_planner/pause_after_u5.log | cut -c1-80)"
