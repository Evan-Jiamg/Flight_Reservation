#!/bin/bash
# Test evaluation of a finished v17 fold (ops/SPEC_v17_sft_pend.md §5-§7, user-approved 2026-09-30): eval_test_rl.py on
# runs/pend_f<F>_v17 with the run's own trainer arguments, the updates exactly {0 = SFT (u0), final} read from final.json,
# seeds 0..7 -> test.jsonl; EVERY fold also the untrained base re-scored under the v17 definitions (--include-base:
# ckpt/sft_e0 -> test_base.jsonl; fix round 2, user decision: fp32 P_end, capped turn stats); then eval_test_boot.py
# (check + paired bootstrap), verify_pipeline (the training files still verify) and threshold_control.py (the v17 base
# test_base.jsonl with one fitted logit offset). A GPU OUT OF MEMORY
# (another job; judged on THIS attempt's log lines only) is retried up to 10 times (finished rows are reused); anything
# else stops. Reuses the placeholder / servers left by run_v17_fold.sh (fix round 1, B-N4) or starts them; releases
# every GPU we hold at the end. Fold 2 also prints the v16 run's u0 test rows, only as a labelled REFERENCE (v16 base:
# bf16 P_end, uncapped |diff|), never as "base".
# Usage: run_v17_test.sh F   (F = 0, 1 or 2)
set -uo pipefail
F=${1:-}
case $F in 0|1|2) ;; *) echo "STOP: fold must be 0, 1 or 2"; exit 1;; esac
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v17; RUN=$G/runs/pend_f${F}_v17; H=$G/hold
V16F2=$G/runs/pend_f2_v16
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # torch GPU indices = nvidia-smi / placeholder indices (fix round 3)
export R0_BASE_URL=http://127.0.0.1:8029/v1 R0_MODEL=gpt-oss-120b R0_REASONING_EFFORT=low
export JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
export CONTROLLER_BASE_URL=http://127.0.0.1:8029/v1 OPENAI_API_KEY=local-vllm-unused
export Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
held() { cat $H/status_$1 2>/dev/null || echo 0; }
settarget() { echo $2 > $H/target_$1.tmp && mv -f $H/target_$1.tmp $H/target_$1; }
take_train_gpu() {
  n=0; until [ -f $H/role_train ]; do [ $((n % 600)) -eq 0 ] && echo "WAITING_GPU: no training GPU with room yet ($(date +%H:%M))"; n=$((n + 1)); sleep 1; done
  TG=$(cat $H/role_train); n=0
  until [ "$(held $TG)" -ge 45 ]; do [ $((n % 300)) -eq 0 ] && echo "WAITING_GPU: GPU $TG holds $(held $TG)/45 GiB ($(date +%H:%M))"; n=$((n + 1)); sleep 1; done
  settarget $TG 0; until [ "$(held $TG)" -le 0 ]; do sleep 1; done
  GPU=$TG
}
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
  echo "TEST: GPUs released $(date)"
}
if pgrep -u mzjiang -f "train_planner_rl.py|eval_test_rl.py|smoke_v17.py|run_v17_fold|run_v16_fold|run_v16_test" > /dev/null; then
  echo "TEST: our training / evaluation is running - not starting"; exit 1; fi
cd $C || { echo "STOP: no snapshot $C"; exit 1; }
sha256sum -c --quiet local_sha_v17.txt || { echo "STOP: pend_v17 snapshot sha mismatch"; exit 1; }
[ -f $RUN/final.json ] || { echo "STOP: $RUN has no final.json (training not finished)"; exit 1; }
FU=$($PY -c "import json; f = json.load(open('$RUN/final.json')); assert f['validated'], 'final.json not validated'; print(f['final_update'])") \
  || { echo "STOP: final.json is not validated"; exit 1; }
REASON=$($PY -c "import json; print(json.load(open('$RUN/final.json'))['stop_reason'])")
BASE="--include-base"                 # every fold (fix round 2): the base under the v17 definitions
trap release EXIT
if pgrep -u mzjiang -f gpu_holder2.py > /dev/null; then
  echo "TEST: reusing the running placeholder ($H)"
else
  rm -rf $H; mkdir -p $H
  PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder2.py $H >> $G/gpu_holder2.log 2>&1 < /dev/null &
  sleep 10
fi
echo "=== TEST fold $F start $(date): SFT (u0) and final u$FU ($REASON), seeds 0..7 $BASE"
rc=1
for attempt in $(seq 1 10); do
  bash $G/start_servers5.sh > $G/servers_check_test_f${F}_v17.log 2>&1 || { echo "STOP: servers not available"; tail -5 $G/servers_check_test_f${F}_v17.log; exit 1; }
  take_train_gpu
  SZ=$(stat -c %s $RUN/test.log 2>/dev/null || echo 0)
  echo "=== test evaluation on GPU $GPU (attempt $attempt) $(date)"
  $PY eval_test_rl.py --final --test-seeds 0 1 2 3 4 5 6 7 --test-updates 0 $FU $BASE \
      --fold $F --planner-path $Q4 --gpu $GPU --rollout-workers 4 --out $RUN >> $RUN/test.log 2>&1
  rc=$?; settarget $TG 45; echo "test rc=$rc $(date)"
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
echo "verify passed (training files unchanged, tested updates = {0, final})"
if [ "$F" = "2" ]; then
  echo "=== REFERENCE ONLY -- v16 base (runs/pend_f2_v16 u0; bf16 P_end, uncapped |diff|; NOT the v17 base above):"
  grep -E "^u0 +Task (1|2)" $V16F2/test_boot.txt | sed -E 's/^u0( +)/v16-base-ref\1/' || echo "(no u0 rows in $V16F2/test_boot.txt)"
fi
TT="--test $RUN/test_base.jsonl --test-update base"
$PY threshold_control.py --train $RUN/base_pend_train.jsonl $TT --splits $G/splits_v1.json --fold $F \
    --json-out $RUN/threshold_control.json > $RUN/threshold_control.txt 2>&1
trc=$?; cat $RUN/threshold_control.txt
[ $trc -eq 0 ] || { echo "STOP: threshold control failed (rc $trc)"; exit 1; }
echo "TEST fold $F DONE $(date)"
