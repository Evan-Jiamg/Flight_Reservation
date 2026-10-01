#!/bin/bash
# SPEC v18 §6.1-6.3: the reranker's data and gate. (1) train_reranker.py generate: the v17 u0 policy (served by the Planner
# vLLM as p0-<sha12>) + Ditto-8B on the training GPU + the CURRENT Borda selector, teacher-forced Task 1 on fold 2's
# train_all with seeds 0 and 1 (the training pairs) and on validation_all with seed 0 (the gate's candidates);
# (2) train_reranker.py fit on CPU: pairwise LR, LOCO with PCA inside the folds, C by LOCO accuracy, Platt, the gate
# (LOCO >= 0.60 and validation >= 0.55) with cluster-bootstrap CIs -> labels_v18/reranker_v18_f2.json. A failed gate is
# not an error: the selector stays Borda (run_v18_fold.sh reads the gate).
# GPU: the placeholder hands the training GPU over (45 GiB) for Ditto; a GPU OOM (another job) is retried up to 10 times
# (the output of a finished generation is kept; an unfinished one is regenerated into a fresh file). Releases only what
# it started (as run_v17_smoke.sh). Usage: run_v18_reranker.sh [F]   (F = 2)
set -uo pipefail
F=${1:-2}
[ "$F" = "2" ] || { echo "STOP: SPEC v18 is a fold-2 trial"; exit 1; }
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v18; H=${HOLD_DIR:-$G/hold}; L=$G/labels_v18
INIT=$G/runs/pend_f${F}_v17/ckpt/u00000
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID OPENAI_API_KEY=local-vllm-unused
export R0_CONTEXT=${OSS_MAX_MODEL_LEN:-32768}   # 2026-10-02: = gpt-oss max_model_len (start_servers6.sh); 12288 overflowed
export Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
held() { cat $H/status_$1 2>/dev/null || echo 0; }
settarget() { echo $2 > $H/target_$1.tmp && mv -f $H/target_$1.tmp $H/target_$1; }
TN=${TRAIN_NEED_GIB:-45}
TG=""
holder_ok() { pgrep -u mzjiang -f "python.* .*gpu_holder3\.py" > /dev/null && [ -f $H/plan ] && [ $(( $(date +%s) - $(stat -c %Y $H/plan) )) -lt 60 ]; }
holder_check() { holder_ok || { echo "STOP: the GPU holder (gpu_holder3.py) is not running or not writing $H/plan (HOLD_DIR?)"; exit 1; }; }
take_train_gpu() {
  n=0; until [ -f $H/role_train ]; do [ $((n % 300)) -eq 0 ] && { holder_check; echo "WAITING_GPU: no committed placement yet ($(date +%H:%M))"; }; n=$((n + 1)); sleep 1; done
  TG=$(cat $H/role_train); n=0
  settarget $TG $TN
  until [ "$(held $TG)" -ge $TN ]; do [ $((n % 300)) -eq 0 ] && { holder_check; echo "WAITING_GPU: GPU $TG holds $(held $TG)/$TN GiB ($(date +%H:%M))"; }; n=$((n + 1)); sleep 1; done
  settarget $TG 0; n=0
  until [ "$(held $TG)" -le 0 ]; do n=$((n + 1)); [ $((n % 300)) -eq 0 ] && { holder_check; echo "WAITING_GPU: GPU $TG still holds $(held $TG) GiB ($(date +%H:%M))"; }; sleep 1; done
  GPU=$TG
}
STARTED_HOLDER=0; STARTED_SERVERS=0
release() {
  [ -n "$TG" ] && settarget $TG $TN
  if [ "$STARTED_HOLDER" != "1" ]; then echo "RERANK: training GPU handed back; placeholder / servers left running $(date)"; return; fi
  touch $H/stop; sleep 3; pkill -u mzjiang -f "python.* .*gpu_holder3\.py"
  if [ "$STARTED_SERVERS" != "1" ]; then echo "RERANK: placeholder stopped; servers were already up - left running"; return; fi
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
  echo "RERANK: GPUs released $(date)"
}
if pgrep -u mzjiang -f "train_planner_rl.py|eval_test_rl.py|smoke_v1[78].py|bench_tf_generate.py|train_reranker.py" > /dev/null; then
  echo "RERANK: another job of ours is running - not starting"; exit 1; fi
cd $C || { echo "STOP: no snapshot $C"; exit 1; }
sha256sum -c --quiet local_sha_v18.txt || { echo "STOP: pend_v18 snapshot sha mismatch"; exit 1; }
[ -f $INIT/adapter/adapter_model.safetensors ] || { echo "STOP: no v17 u0 at $INIT"; exit 1; }
pgrep -u mzjiang -f "python.* .*(gpu_holder2|gpu_grab)\.py" > /dev/null && { echo "STOP: an old placeholder is running"; exit 1; }
mkdir -p $L
trap release EXIT
pgrep -u mzjiang -f "vllm serve" > /dev/null || STARTED_SERVERS=1
if ! pgrep -u mzjiang -f "python.* .*gpu_holder3\.py" > /dev/null; then
  STARTED_HOLDER=1; rm -rf $H; mkdir -p $H
  PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder3.py $H >> $G/gpu_holder3.log 2>&1 < /dev/null &
  sleep 10
fi
for JOB in "train_all 0 1" "validation_all 0"; do
  set -- $JOB; SPLIT=$1; shift; SEEDS="$*"
  case $SPLIT in train_all) OUT=$L/rerank_cands_train_f${F}.jsonl;; validation_all) OUT=$L/rerank_cands_val_f${F}.jsonl;; esac
  if [ -f $OUT ] && [ -f $OUT.provenance.json ]; then echo "$OUT exists - kept"; continue; fi
  rc=1
  for attempt in $(seq 1 10); do
    rm -f $OUT $OUT.tmp $OUT.provenance.json
    bash $G/start_servers6.sh > $G/servers_check_rerank_v18.log 2>&1 || { echo "STOP: a server failed to start"; tail -8 $G/servers_check_rerank_v18.log; exit 1; }
    take_train_gpu
    SZ=$(stat -c %s $OUT.log 2>/dev/null || echo 0)
    echo "=== candidates $SPLIT seeds $SEEDS on GPU $GPU (attempt $attempt) $(date)"
    $PY train_reranker.py generate --fold $F --split $SPLIT --seeds $SEEDS --init-adapter $INIT --planner-path $Q4 \
        --gpu $GPU --workers 4 --force-unload --out $OUT >> $OUT.log 2>&1   # no job of ours runs (checked above)
    rc=$?; settarget $TG $TN; TG=""
    [ $rc -eq 0 ] && break
    if tail -c +$((SZ + 1)) $OUT.log | grep -cE "CUDA out of memory|CUDA error: out of memory|OutOfMemoryError" > /dev/null; then
      echo "OOM (another job) -> retry"; sleep 60; continue; fi
    echo "STOP: candidate generation failed"; tail -8 $OUT.log | cut -c1-300; exit 1
  done
  [ $rc -eq 0 ] || { echo "STOP: 10 OOM retries used"; exit 1; }
done
echo "=== fit (CPU) $(date)"
$PY train_reranker.py fit --fold $F --cands-train $L/rerank_cands_train_f${F}.jsonl --cands-val $L/rerank_cands_val_f${F}.jsonl \
    --out $L/reranker_v18_f${F}.json > $L/reranker_v18_f${F}.log 2>&1 || { echo "STOP: fit failed"; tail -8 $L/reranker_v18_f${F}.log; exit 1; }
tail -2 $L/reranker_v18_f${F}.log
echo "RERANKER DONE $(date) (a failed gate keeps the Borda selector; run_v18_fold.sh reads it)"
