#!/bin/bash
# Test evaluation of the finished v18 fold-2 run (ops/SPEC_v18_multiobj_rerank.md §8): eval_test_rl.py on
# runs/pend_f2_v18 with the run's own trainer arguments (--spec v18, the reranker when its gate passed), ONLY the final
# policy of final.json (u0 when GRPO was not adopted), seeds 0..7 -> test.jsonl; Task 2 judged by clean_v18 (a lost
# ledger verdict makes only the coverage diagnostic None). Then eval_test_boot.py (check + paired bootstrap against v17's
# existing test rows -- v17 SFT (u0), v17 GRPO (u5), never re-run; their file's sha is recorded) and verify_pipeline
# (v18 branch). A GPU OUT OF MEMORY (another job; THIS attempt's lines only) is retried up to 10 times (finished rows are
# reused). Reuses the placeholder / servers left by run_v18_fold.sh or starts them; releases every GPU we hold at the
# end unless KEEP_SERVERS=1 (run_v18_chain.sh keeps them for the benchmark step).
# Usage: run_v18_test.sh [F]   (F = 2)
set -uo pipefail
F=${1:-2}
[ "$F" = "2" ] || { echo "STOP: SPEC v18 is a fold-2 trial"; exit 1; }
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v18; RUN=$G/runs/pend_f${F}_v18; H=${HOLD_DIR:-$G/hold}
L=$G/labels_v18; RUN17=$G/runs/pend_f${F}_v17
LABELS=$L/act_labels_train_f${F}.jsonl; LABELS_VAL=$L/act_labels_val_f${F}.jsonl; RERANK=$L/reranker_v18_f${F}.json
INIT=$RUN17/ckpt/u00000
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export R0_BASE_URL=http://127.0.0.1:8029/v1 R0_MODEL=gpt-oss-120b R0_REASONING_EFFORT=low
export JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
export CONTROLLER_BASE_URL=http://127.0.0.1:8029/v1 OPENAI_API_KEY=local-vllm-unused
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
release() {
  [ -n "$TG" ] && settarget $TG $TN
  if [ "${KEEP_SERVERS:-0}" = "1" ]; then echo "TEST: placeholder / servers kept (KEEP_SERVERS=1) $(date)"; return; fi
  touch $H/stop; sleep 3; pkill -u mzjiang -f "python.* .*gpu_holder3\.py"
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
  echo "TEST: GPUs released $(date)"
}
if pgrep -u mzjiang -f "train_planner_rl.py|eval_test_rl.py|smoke_v1[78].py|run_v1[78]_fold|bench_tf_generate.py|train_reranker.py" > /dev/null; then
  echo "TEST: our training / evaluation is running - not starting"; exit 1; fi
cd $C || { echo "STOP: no snapshot $C"; exit 1; }
sha256sum -c --quiet local_sha_v18.txt || { echo "STOP: pend_v18 snapshot sha mismatch"; exit 1; }
pgrep -u mzjiang -f "python.* .*(gpu_holder2|gpu_grab)\.py" > /dev/null && { echo "STOP: an old placeholder is running"; exit 1; }
[ -f $RUN/final.json ] || { echo "STOP: $RUN has no final.json (training not finished)"; exit 1; }
[ -f $RUN17/test.jsonl ] || { echo "STOP: no v17 comparison file $RUN17/test.jsonl"; exit 1; }
FU=$($PY -c "import json; f = json.load(open('$RUN/final.json')); assert f['validated'], 'final.json not validated'; print(f['final_update'])") \
  || { echo "STOP: final.json is not validated"; exit 1; }
REASON=$($PY -c "import json; f = json.load(open('$RUN/final.json')); print(f['stop_reason'], 'GRPO adopted' if f['grpo_adopted'] else 'GRPO NOT adopted (final = u0)')")
V18ARGS="--spec v18 --init-adapter $INIT --act-labels $LABELS --act-labels-val $LABELS_VAL --reranker $RERANK"
trap release EXIT
if pgrep -u mzjiang -f "python.* .*gpu_holder3\.py" > /dev/null; then
  echo "TEST: reusing the running placeholder ($H)"
else
  rm -rf $H; mkdir -p $H
  PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder3.py $H >> $G/gpu_holder3.log 2>&1 < /dev/null &
  sleep 10
fi
echo "=== TEST v18 fold $F start $(date): final u$FU ($REASON), seeds 0..7"
rc=1
for attempt in $(seq 1 10); do
  bash $G/start_servers6.sh > $G/servers_check_test_f${F}_v18.log 2>&1 || { echo "STOP: a server failed to start"; tail -8 $G/servers_check_test_f${F}_v18.log; exit 1; }
  take_train_gpu
  SZ=$(stat -c %s $RUN/test.log 2>/dev/null || echo 0)
  echo "=== test evaluation on GPU $GPU (attempt $attempt) $(date)"
  $PY eval_test_rl.py --final --test-seeds 0 1 2 3 4 5 6 7 --test-updates $FU --v17-run $RUN17 \
      --fold $F --planner-path $Q4 --gpu $GPU --rollout-workers 4 --out $RUN $V18ARGS >> $RUN/test.log 2>&1
  rc=$?; settarget $TG $TN; TG=""; echo "test rc=$rc $(date)"
  [ $rc -eq 0 ] && break
  if tail -c +$((SZ + 1)) $RUN/test.log | grep -cE "CUDA out of memory|CUDA error: out of memory|OutOfMemoryError|GPU budget not available" > /dev/null; then
    echo "OOM (another job on the GPU) -> retry (finished rows are reused)"; sleep 60; continue; fi
  echo "STOP: test evaluation failed"; grep -v "Loading weights" $RUN/test.log | tail -8 | cut -c1-300; exit 1
done
[ $rc -eq 0 ] || { echo "STOP: 10 OOM retries used"; exit 1; }
$PY eval_test_boot.py $RUN > $RUN/test_boot.txt 2>&1
brc=$?; cat $RUN/test_boot.txt
[ $brc -eq 0 ] || { echo "STOP: the test check failed (rc $brc)"; exit 1; }
$PY verify_pipeline.py --rl-dir $RUN --splits $G/splits_v1.json --fold $F --split train --arm pend \
    --sepsim-path $G/trees/e1r_cf19400 > $RUN/verify_test.txt 2>&1
grep -q "PIPELINE VERIFICATION PASSED" $RUN/verify_test.txt || {
  echo "STOP: verify of the training files failed"; grep -E "FAIL" $RUN/verify_test.txt | head -10 | cut -c1-260; exit 1; }
echo "verify passed (training files unchanged, tested update = final)"
echo "TEST v18 fold $F DONE $(date) -- benchmark: run_v18_bench.sh $F"
