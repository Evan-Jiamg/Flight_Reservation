#!/bin/bash
# OOM auto-recovery for the v12 continuation (same rule as run_v11_guard.sh): only a training stop caused by GPU OUT OF
# MEMORY (another user's job on the training GPU) restarts the placeholder with the kept roles and re-runs the
# continuation (--resume from the latest checkpoint); any other stop is reported and left for inspection.
G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold; RUN=$G/runs/pend_f2_v11
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
L=$G/run_v12_continue.log
while pgrep -u mzjiang -f "run_v12_continue.sh|run_v12_launch.sh" > /dev/null; do sleep 30; done
for retry in $(seq 1 20); do
  if grep -q "V12 FORMAL DONE" $L; then echo "GUARD: formal run finished $(date)"; exit 0; fi
  if grep -q "STOP: training failed" $L && tail -300 $RUN/train.log | grep -qE "CUDA out of memory|OutOfMemoryError"; then
    echo "GUARD: training stopped on GPU OOM -> recovery $retry $(date)"
  else
    echo "GUARD: STOP not caused by OOM -> needs attention $(date)"; tail -8 $L | cut -c1-250; exit 1
  fi
  pkill -u mzjiang -f gpu_holder2.py; sleep 3; rm -f $H/stop
  PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder2.py $H >> $G/gpu_holder2.log 2>&1 < /dev/null &
  sleep 10
  echo 45 > $H/target_$(cat $H/role_train).tmp && mv -f $H/target_$(cat $H/role_train).tmp $H/target_$(cat $H/role_train)
  L=$G/run_v12_continue_retry$retry.log
  bash $G/run_v12_continue.sh > $L 2>&1
  echo "GUARD: continuation (retry $retry) ended $(date)"
done
touch $H/stop
echo "GUARD: 20 recoveries used, giving up $(date)"
