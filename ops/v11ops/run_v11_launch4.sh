#!/bin/bash
# v11 launcher (4): the GPU placeholder grabs free memory on every cfda5 GPU the moment it appears and hands it over
# to the servers and the trainer; gate then formal continuation; the placeholder is stopped (memory released) at the end.
G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
while pgrep -u mzjiang -f "run_v11_launch3.sh|run_v11_formal3.sh|run_v11_continue3.sh|start_servers3.sh" > /dev/null; do sleep 5; done
echo "older chain gone $(date)"
rm -rf $H; mkdir -p $H
PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder.py $H > $G/gpu_holder.log 2>&1 < /dev/null &
sleep 20; tail -2 $G/gpu_holder.log
bash $G/run_v11_formal4.sh > $G/run_v11_formal4.log 2>&1
bash $G/run_v11_continue4.sh > $G/run_v11_continue4.log 2>&1
touch $H/stop
echo "LAUNCHER4 DONE $(date)"
