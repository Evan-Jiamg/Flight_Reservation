#!/bin/bash
# v11 launcher (5): the all-at-once GPU placeholder takes a whole requirement the moment a cfda5 GPU has room for it
# (servers 91 GiB, training 45 GiB) and hands it over; gate then formal continuation; placeholder released at the end.
G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
while pgrep -u mzjiang -f "run_v11_launch4.sh|run_v11_formal4.sh|run_v11_continue4.sh|start_servers4.sh|gpu_holder.py" > /dev/null; do sleep 2; done
echo "older chain gone $(date)"
rm -rf $H; mkdir -p $H
PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder2.py $H > $G/gpu_holder2.log 2>&1 < /dev/null &
sleep 15; tail -2 $G/gpu_holder2.log
bash $G/run_v11_formal5.sh > $G/run_v11_formal5.log 2>&1
bash $G/run_v11_continue5.sh > $G/run_v11_continue5.log 2>&1
touch $H/stop
echo "LAUNCHER5 DONE $(date)"
