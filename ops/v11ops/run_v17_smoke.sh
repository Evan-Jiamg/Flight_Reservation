#!/bin/bash
# GPU smoke test of the v17 path (sep-sim/smoke_v17.py; SPEC v17 S14 + fix round 1, 2026-09-30) BEFORE any v17 fold run:
# real Qwen3-4B learner + Planner vLLM + Ditto-8B + gpt-oss (a few Task 2 episodes), on a subset of fold F, in a fresh
# scratch dir runs/smoke_v17_f<F>_<time>. Same exports / snapshot sha check / servers / placeholder handoff as
# run_v17_fold.sh (the training GPU is taken from the placeholder only when it holds the whole 45 GiB, i.e. nobody else
# uses it). Fix round 2: it releases ONLY what it started -- the placeholder (and the vLLM servers) are stopped at the
# end only if this script started them; otherwise the training GPU is handed back to the running placeholder and the
# servers are left up. After a smoke run that started and released the placeholder, run_v17_fold.sh needs the
# placeholder started again (ops/v11ops/RESUME.md).
# 2026-09-30: placeholder gpu_holder3.py (incremental, dynamic placement P1/P2) + start_servers6.sh; in P2 the Planner
# vLLM runs on the training GPU (its memory belongs to its own process, so the smoke's per-pid / memory_allocated checks
# are unaffected); release() kills our vLLM servers by port / nvidia-smi pid on BOTH GPUs, wherever they run.
# Usage: run_v17_smoke.sh [F]   (default F = 2)
set -uo pipefail
F=${1:-2}
case $F in 0|1|2) ;; *) echo "STOP: fold must be 0, 1 or 2"; exit 1;; esac
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v17; H=${HOLD_DIR:-$G/hold}   # HOLD_DIR: only for a cut-over next to an old placeholder
OUT=$G/runs/smoke_v17_f${F}_$(date +%Y%m%d_%H%M%S)
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # torch GPU indices = nvidia-smi / placeholder indices (fix round 3)
export R0_BASE_URL=http://127.0.0.1:8029/v1 R0_MODEL=gpt-oss-120b R0_REASONING_EFFORT=low
export JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
export CONTROLLER_BASE_URL=http://127.0.0.1:8029/v1 OPENAI_API_KEY=local-vllm-unused
export Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
held() { cat $H/status_$1 2>/dev/null || echo 0; }
settarget() { echo $2 > $H/target_$1.tmp && mv -f $H/target_$1.tmp $H/target_$1; }
TN=${TRAIN_NEED_GIB:-45}                 # the training share the placeholder keeps for the trainer (gpu_holder3.py TRAIN_NEED)
TG=""
holder_ok() { pgrep -u mzjiang -f "python.* .*gpu_holder3\.py" > /dev/null && [ -f $H/plan ] && [ $(( $(date +%s) - $(stat -c %Y $H/plan) )) -lt 60 ]; }
holder_check() { holder_ok || { echo "STOP: the GPU holder (gpu_holder3.py) is not running or not writing $H/plan (HOLD_DIR?)"; exit 1; }; }
take_train_gpu() {
  n=0; until [ -f $H/role_train ]; do [ $((n % 300)) -eq 0 ] && { holder_check; echo "WAITING_GPU: no committed placement yet: $(cat $H/plan 2>/dev/null) ($(date +%H:%M))"; }; n=$((n + 1)); sleep 1; done
  TG=$(cat $H/role_train); n=0
  settarget $TG $TN                  # never wait for more than the placeholder is asked to hold (audit B1)
  until [ "$(held $TG)" -ge $TN ]; do [ $((n % 300)) -eq 0 ] && { holder_check; echo "WAITING_GPU: training GPU $TG holds $(held $TG)/$TN GiB ($(date +%H:%M))"; }; n=$((n + 1)); sleep 1; done
  settarget $TG 0; n=0
  until [ "$(held $TG)" -le 0 ]; do n=$((n + 1)); [ $((n % 300)) -eq 0 ] && { holder_check; echo "WAITING_GPU: GPU $TG still holds $(held $TG) GiB after the hand-over ($(date +%H:%M))"; }; sleep 1; done
  GPU=$TG
}
STARTED_HOLDER=0; STARTED_SERVERS=0
release() {
  [ -n "$TG" ] && settarget $TG $TN
  if [ "$STARTED_HOLDER" != "1" ]; then
    echo "SMOKE: training GPU handed back to the placeholder we did not start; placeholder / servers left running $(date)"
    return
  fi
  touch $H/stop; sleep 3; pkill -u mzjiang -f "python.* .*gpu_holder3\.py"
  if [ "$STARTED_SERVERS" != "1" ]; then echo "SMOKE: placeholder stopped; servers were already up - left running"; return; fi
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
  echo "SMOKE: GPUs released $(date)"
}
if pgrep -u mzjiang -f "train_planner_rl.py|eval_test_rl.py|smoke_v17.py|run_v17_fold|run_v17_test" > /dev/null; then
  echo "SMOKE: our training / evaluation is running - not starting"; exit 1; fi
cd $C || { echo "STOP: no snapshot $C"; exit 1; }
sha256sum -c --quiet local_sha_v17.txt || { echo "STOP: pend_v17 snapshot sha mismatch"; exit 1; }
pgrep -u mzjiang -f "python.* .*(gpu_holder2|gpu_grab)\.py" > /dev/null && { echo "STOP: an old placeholder (gpu_holder2.py / gpu_grab.py) is running - finish the cut-over first (ops/v11ops/RESUME.md)"; exit 1; }
trap release EXIT
pgrep -u mzjiang -f "vllm serve" > /dev/null || STARTED_SERVERS=1
if ! pgrep -u mzjiang -f "python.* .*gpu_holder3\.py" > /dev/null; then
  STARTED_HOLDER=1
  rm -rf $H; mkdir -p $H
  PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder3.py $H >> $G/gpu_holder3.log 2>&1 < /dev/null &
  sleep 10
fi
bash $G/start_servers6.sh > $G/servers_check_smoke_v17.log 2>&1 || { echo "STOP: a server failed to start (not a memory race; those are waited out inside start_servers6.sh)"; tail -8 $G/servers_check_smoke_v17.log; exit 1; }
take_train_gpu
mkdir -p $OUT
echo "=== SMOKE v17 fold $F on GPU $GPU -> $OUT $(date)"
$PY smoke_v17.py --fold $F --planner-path $Q4 --gpu $GPU --rollout-workers 2 --out $OUT --smoke-convs 3 --val-convs 2 \
    > $OUT/smoke.log 2>&1
rc=$?
settarget $TG $TN; TG=""
tail -3 $OUT/smoke.log
grep -q "SMOKE V17 PASSED" $OUT/smoke.log || { echo "STOP: smoke failed (rc $rc)"; grep -E "FAILED|Error|Traceback" $OUT/smoke.log | tail -5 | cut -c1-300; exit 1; }
echo "SMOKE v17 PASSED $(date)"
