#!/bin/bash
# Launcher of the v16 formal run: a fresh all-at-once GPU placeholder (roles decided when a GPU has room), the formal
# run (run_v16_formal.sh, its own log), then every GPU we hold is released (servers 8029/8031 + placeholder).
# The OOM guard (run_v16_guard.sh) is started separately with its own nohup and waits for this launcher.
G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
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
  echo "LAUNCHER: GPUs released $(date)"
}
if pgrep -u mzjiang -f "train_planner_rl.py|run_v16_formal.sh|vllm serve|gpu_holder2" > /dev/null; then
  echo "LAUNCHER: our training / servers / placeholder already running - not launching"; exit 1; fi
rm -rf $H; mkdir -p $H
PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder2.py $H >> $G/gpu_holder2.log 2>&1 < /dev/null &
sleep 10
echo "LAUNCHER: placeholder up, formal run starting $(date)"
bash $G/run_v16_formal.sh > $G/run_v16_formal.log 2>&1
echo "LAUNCHER: formal run ended: $(tail -1 $G/run_v16_formal.log | cut -c1-150)"
# an OOM stop is recovered by the guard (it restarts the placeholder); every other end releases the GPUs now
if grep -q "STOP: training failed" $G/run_v16_formal.log && tail -300 $G/runs/pend_f2_v16/train.log | grep -qE "CUDA out of memory|OutOfMemoryError"; then
  echo "LAUNCHER: OOM stop - left to the guard"
else
  release
fi
