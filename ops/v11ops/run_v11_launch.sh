#!/bin/bash
# Relaunch after the first v11 gate aborted (GPU0 held by another user's job): wait for the old scripts to exit,
# then the gate (servers start only when GPU0 has room) and the formal continuation, one after the other.
G=/tmp2/mzjiang_usersim/grpo_planner
while pgrep -u mzjiang -f "run_v11_formal.sh|run_v11_continue.sh" > /dev/null; do sleep 30; done
echo "old scripts gone $(date)"
bash $G/run_v11_formal2.sh > $G/run_v11_formal2.log 2>&1
bash $G/run_v11_continue2.sh > $G/run_v11_continue2.log 2>&1
echo "LAUNCHER DONE $(date)"
