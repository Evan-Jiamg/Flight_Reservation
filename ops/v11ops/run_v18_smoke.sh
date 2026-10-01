#!/bin/bash
# GPU smoke test of the SPEC v18 path (sep-sim/smoke_v18.py; SPEC v18 §16 step 5) BEFORE the fold-2 run: the real Qwen3-4B
# learner + Planner vLLM + Ditto-8B + the reranker + gpt-oss (a few Task 2 episodes), the init stage (v17 u0 copied, ref),
# a Task 1 validation of u0 and ONE update on a subset, re-checked by the verifier's own recomputation (rewards, fixed
# scales / kappa, the token-weighted tau = the targets on the u1 batch, advantages, turn-1 value masks, the reranked
# selections, 8 optimizer steps). Scratch dir runs/smoke_v18_f2_<time>. Same exports / snapshot check / servers /
# placeholder hand-over as run_v18_fold.sh; it needs the approved labels and the reranker json (as the fold run).
# Releases ONLY what it started (as run_v17_smoke.sh). Usage: run_v18_smoke.sh [F]   (F = 2)
set -uo pipefail
F=${1:-2}
[ "$F" = "2" ] || { echo "STOP: SPEC v18 is a fold-2 trial"; exit 1; }
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v18; H=${HOLD_DIR:-$G/hold}; L=$G/labels_v18
OUT=$G/runs/smoke_v18_f${F}_$(date +%Y%m%d_%H%M%S)
LABELS=$L/act_labels_train_f${F}.jsonl; LABELS_VAL=$L/act_labels_val_f${F}.jsonl; RERANK=$L/reranker_v18_f${F}.json
INIT=$G/runs/pend_f${F}_v17/ckpt/u00000
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export R0_BASE_URL=http://127.0.0.1:8029/v1 R0_MODEL=gpt-oss-120b R0_REASONING_EFFORT=low
export JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
export CONTROLLER_BASE_URL=http://127.0.0.1:8029/v1 OPENAI_API_KEY=local-vllm-unused
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
  if [ "$STARTED_HOLDER" != "1" ]; then echo "SMOKE: training GPU handed back; placeholder / servers left running $(date)"; return; fi
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
if pgrep -u mzjiang -f "train_planner_rl.py|eval_test_rl.py|smoke_v1[78].py|run_v1[78]_fold|run_v1[78]_test|bench_tf_generate.py|train_reranker.py" > /dev/null; then
  echo "SMOKE: our training / evaluation is running - not starting"; exit 1; fi
cd $C || { echo "STOP: no snapshot $C"; exit 1; }
sha256sum -c --quiet local_sha_v18.txt || { echo "STOP: pend_v18 snapshot sha mismatch"; exit 1; }
for f in $LABELS $LABELS_VAL $RERANK $INIT/state.json; do [ -f $f ] || { echo "STOP: missing $f"; exit 1; }; done
LSHA=$(sha256sum $LABELS | cut -d' ' -f1)
grep -q "$LSHA" $LABELS.APPROVED 2>/dev/null || { echo "STOP: the act labels are not approved ($LABELS.APPROVED)"; exit 1; }
pgrep -u mzjiang -f "python.* .*(gpu_holder2|gpu_grab)\.py" > /dev/null && { echo "STOP: an old placeholder is running"; exit 1; }
trap release EXIT
pgrep -u mzjiang -f "vllm serve" > /dev/null || STARTED_SERVERS=1
if ! pgrep -u mzjiang -f "python.* .*gpu_holder3\.py" > /dev/null; then
  STARTED_HOLDER=1; rm -rf $H; mkdir -p $H
  PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder3.py $H >> $G/gpu_holder3.log 2>&1 < /dev/null &
  sleep 10
fi
bash $G/start_servers6.sh > $G/servers_check_smoke_v18.log 2>&1 || { echo "STOP: a server failed to start"; tail -8 $G/servers_check_smoke_v18.log; exit 1; }
take_train_gpu
mkdir -p $OUT
echo "=== SMOKE v18 fold $F on GPU $GPU -> $OUT $(date)"
$PY smoke_v18.py --fold $F --planner-path $Q4 --gpu $GPU --rollout-workers 2 --out $OUT --smoke-convs 3 --val-convs 2 \
    --init-adapter $INIT --act-labels $LABELS --act-labels-val $LABELS_VAL --reranker $RERANK > $OUT/smoke.log 2>&1
rc=$?
settarget $TG $TN; TG=""
tail -3 $OUT/smoke.log
grep -q "SMOKE V18 PASSED" $OUT/smoke.log || { echo "STOP: smoke failed (rc $rc)"; grep -E "FAILED|Error|Traceback" $OUT/smoke.log | tail -5 | cut -c1-300; exit 1; }
echo "SMOKE v18 PASSED $(date)"
