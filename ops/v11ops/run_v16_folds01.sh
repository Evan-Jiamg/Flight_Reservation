#!/bin/bash
# Sequencer (user 2026-09-28: "先跑 fold 0 和 fold 1"): one placeholder for both folds, fold 0 then fold 1 (the two
# GPUs hold one run at a time: servers 91 GiB + training 45 GiB), each through run_v16_fold.sh; a fold that stops
# does not start the next one; at the end (or on any stop) every GPU we hold is released.
G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
FOLDS="${*:-0 1}"          # user 2026-09-28: fold 1 first, fold 0 later (its validation has 1 conversation) -> run as "run_v16_folds01.sh 1"
for F in $FOLDS; do case $F in 0|1) ;; *) echo "FOLDS: fold must be 0 or 1"; exit 1;; esac; done
release() {
  touch $H/stop; sleep 3; pkill -u mzjiang -f gpu_holder2.py
  for port in 8029 8031; do
    pkill -u mzjiang -f "vllm serve .*--port $port"
    for i in $(seq 1 60); do pgrep -u mzjiang -f "vllm serve .*--port $port" > /dev/null || break; sleep 2; done
    pkill -9 -u mzjiang -f "vllm serve .*--port $port"
  done
  sleep 5
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "mzjiang" ] || continue
    ps -o args= -p $p 2>/dev/null | grep -qE "vllm|VLLM|EngineCore" && kill -9 $p 2>/dev/null
  done
  echo "FOLDS: GPUs released $(date)"
}
if pgrep -u mzjiang -f "train_planner_rl.py|vllm serve|gpu_holder2|run_v16_fold.sh|run_v16_guard|run_v16_launch|run_v16_formal|run_v16_reselect|run_v16_night" > /dev/null; then
  echo "FOLDS: our training / servers / placeholder already running - not starting"; exit 1; fi
for F in $FOLDS; do
  [ -d $G/runs/pend_f${F}_v16 ] && { echo "FOLDS: runs/pend_f${F}_v16 already exists - not starting (resume by hand)"; exit 1; }
done
trap release EXIT
rm -rf $H; mkdir -p $H
PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder2.py $H >> $G/gpu_holder2.log 2>&1 < /dev/null &
sleep 10
for F in $FOLDS; do
  bash $G/run_v16_fold.sh $F > $G/run_v16_fold$F.log 2>&1
  if ! grep -q "FOLD $F DONE" $G/run_v16_fold$F.log; then
    echo "FOLDS: fold $F stopped$([ "$F" != "${FOLDS##* }" ] && echo " - later folds not started"): $(grep -E '^STOP' $G/run_v16_fold$F.log | tail -1)"; exit 1; fi
  echo "FOLDS: fold $F done $(date)"
done
echo "FOLDS $FOLDS DONE $(date)"
