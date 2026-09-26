#!/bin/bash
# v11 launcher (3): servers on whichever cfda5 GPU frees first, the trainer on any other GPU with room; gate then
# formal continuation. Waits for any older launcher chain of ours to be gone first.
G=/tmp2/mzjiang_usersim/grpo_planner
while pgrep -u mzjiang -f "run_v11_launch.sh|run_v11_formal2.sh|run_v11_continue2.sh" > /dev/null; do sleep 30; done
echo "older chain gone $(date)"
bash $G/run_v11_formal3.sh > $G/run_v11_formal3.log 2>&1
bash $G/run_v11_continue3.sh > $G/run_v11_continue3.log 2>&1
echo "LAUNCHER3 DONE $(date)"
