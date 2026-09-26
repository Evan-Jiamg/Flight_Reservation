#!/bin/bash
# Auto-recovery for the v11 formal continuation (2026-09-27): another user's job keeps landing on the training GPU.
# When the continuation stops because training ran OUT OF GPU MEMORY, the placeholder is restarted with the existing
# roles (server GPU keeps the running servers; the training GPU is re-taken all at once when 45 GiB are free) and the
# continuation is re-run with --resume from the latest checkpoint. Any other failure stops here and is reported.
G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold; RUN=$G/runs/pend_f2_v11
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
L=$G/run_v11_continue5.log
while pgrep -u mzjiang -f "run_v11_continue5.sh|run_v11_launch5.sh" > /dev/null; do sleep 30; done
for retry in $(seq 1 20); do
  if grep -q "V11 FORMAL DONE" $L; then echo "GUARD: formal run finished $(date)"; exit 0; fi
  if grep -q "STOP: training failed" $L && tail -300 $RUN/train.log | grep -qE "CUDA out of memory|OutOfMemoryError"; then
    echo "GUARD: training stopped on GPU OOM -> recovery $retry $(date)"
  else
    echo "GUARD: STOP not caused by OOM -> needs attention $(date)"; tail -8 $L | cut -c1-250; exit 1
  fi
  # the placeholder with the existing roles (role_server / role_train / targets are kept in $H)
  pkill -u mzjiang -f gpu_holder2.py; sleep 3; rm -f $H/stop
  PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder2.py $H >> $G/gpu_holder2.log 2>&1 < /dev/null &
  sleep 10
  echo 45 > $H/target_$(cat $H/role_train).tmp && mv -f $H/target_$(cat $H/role_train).tmp $H/target_$(cat $H/role_train)
  L=$G/run_v11_continue5_retry$retry.log
  bash $G/run_v11_continue5.sh > $L 2>&1
  echo "GUARD: continuation (retry $retry) ended $(date)"
done
touch $H/stop
echo "GUARD: 20 recoveries used, giving up $(date)"
