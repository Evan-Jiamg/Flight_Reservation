#!/bin/bash
# OOM auto-recovery for the v16 formal run (same rule as run_v11_guard.sh): only a training stop caused by GPU OUT OF
# MEMORY (another user's job on the training GPU) restarts the placeholder with the kept roles and re-runs the
# continuation (--resume from the latest checkpoint); any other stop is reported and left for inspection.
G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold; RUN=$G/runs/pend_f2_v16
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
L=$G/run_v16_formal.log
while pgrep -u mzjiang -f "run_v16_formal.sh|run_v16_launch.sh" > /dev/null; do sleep 30; done
release() {
  touch $H/stop; sleep 3; pkill -u mzjiang -f gpu_holder2.py
  for port in 8029 8031; do
    pkill -u mzjiang -f "vllm serve .*--port $port"
    for i in $(seq 1 60); do pgrep -u mzjiang -f "vllm serve .*--port $port" > /dev/null || break; sleep 2; done
    pkill -9 -u mzjiang -f "vllm serve .*--port $port"
  done
  sleep 5
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do     # orphaned engines of our servers
    [ "$(ps -o user= -p $p 2>/dev/null)" = "mzjiang" ] || continue
    ps -o args= -p $p 2>/dev/null | grep -qE "vllm|VLLM|EngineCore" && kill -9 $p 2>/dev/null
  done
  echo "GUARD: GPUs released $(date)"
}
if grep -q "not launching" $G/run_v16_launch.log 2>/dev/null; then
  echo "GUARD: the launcher refused to start - nothing to guard, nothing released $(date)"; exit 1; fi
for retry in $(seq 1 20); do
  if grep -q "V16 FORMAL DONE" $L; then echo "GUARD: formal run finished $(date)"; release; exit 0; fi
  if grep -q "STOP: training failed" $L && tail -300 $RUN/train.log | grep -qE "CUDA out of memory|OutOfMemoryError"; then
    echo "GUARD: training stopped on GPU OOM -> recovery $retry $(date)"
  else
    echo "GUARD: STOP not caused by OOM -> needs attention $(date)"; tail -8 $L | cut -c1-250; release; exit 1
  fi
  pkill -u mzjiang -f gpu_holder2.py; sleep 3; rm -f $H/stop
  PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder2.py $H >> $G/gpu_holder2.log 2>&1 < /dev/null &
  sleep 10
  echo 45 > $H/target_$(cat $H/role_train).tmp && mv -f $H/target_$(cat $H/role_train).tmp $H/target_$(cat $H/role_train)
  L=$G/run_v16_formal_retry$retry.log
  bash $G/run_v16_formal.sh > $L 2>&1
  echo "GUARD: continuation (retry $retry) ended $(date)"
done
release
echo "GUARD: 20 recoveries used, giving up $(date)"
