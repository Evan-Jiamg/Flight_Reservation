#!/bin/bash
# Fold-2 test evaluation (user 2026-09-28: "先測 u5", "u5 和 u0，u10 先不要", 8 seeds): eval_test_rl.py on runs/pend_f2_v16
# with the run's own trainer arguments (as its re-selection), u0 and the re-selected u5, seeds 0..7 -> test.jsonl;
# then eval_test_boot.py (check + paired bootstrap) and verify_pipeline (the training files still verify). A GPU OUT OF
# MEMORY (another job) is retried up to 10 times (finished rows are reused); anything else stops. Starts the placeholder,
# releases every GPU we hold at the end. Usage: run_v16_test.sh
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v16; RUN=$G/runs/pend_f2_v16; H=$G/hold
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
if pgrep -u mzjiang -f "train_planner_rl.py|eval_test_rl.py|vllm serve|gpu_holder2|run_v16_fold|run_v16_guard|run_v16_launch|run_v16_formal|run_v16_reselect|run_v16_night" > /dev/null; then
  echo "TEST: our training / servers / placeholder already running - not starting"; exit 1; fi
cd $C
sha256sum -c --quiet local_sha_v16.txt || { echo "STOP: pend_v16 snapshot sha mismatch"; exit 1; }
sha256sum -c --quiet local_sha_v16_test.txt || { echo "STOP: eval_test_rl / test_boot sha mismatch"; exit 1; }
BU=$($PY -c "import json; print(json.load(open('$RUN/reselect_best.json'))['best']['update'])")
[ "$BU" = "5" ] || { echo "STOP: the re-selected checkpoint is u$BU, not u5 as agreed with the user"; exit 1; }
LAST=$($PY -c "import json; print(json.load(open('$RUN/ckpt/LATEST.json'))['update'])")
trap release EXIT
rm -rf $H; mkdir -p $H
PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder2.py $H >> $G/gpu_holder2.log 2>&1 < /dev/null &
sleep 10
echo "=== TEST start $(date): u0 u$BU, seeds 0..7"
for attempt in $(seq 1 10); do
  bash $G/start_servers5.sh > $G/servers_check_test.log 2>&1 || { echo "STOP: servers not available"; tail -5 $G/servers_check_test.log; exit 1; }
  take_train_gpu
  SZ=$(stat -c %s $RUN/test.log 2>/dev/null || echo 0)
  echo "=== test evaluation on GPU $GPU (attempt $attempt) $(date)"
  $PY eval_test_rl.py --final --test-seeds 0 1 2 3 4 5 6 7 --test-updates 0 $BU \
      --fold 2 --planner-path $Q4 --gpu $GPU --G 4 --scenarios-per-update 4 --updates $LAST --val-every 5 \
      --rollout-workers 4 --out $RUN >> $RUN/test.log 2>&1
  rc=$?; settarget $TG 45; echo "test rc=$rc $(date)"
  [ $rc -eq 0 ] && break
  if tail -c +$((SZ + 1)) $RUN/test.log | grep -cE "CUDA out of memory|OutOfMemoryError" > /dev/null; then
    echo "OOM (another job on the GPU) -> retry (finished rows are reused)"; sleep 60; continue; fi
  echo "STOP: test evaluation failed"; grep -v "Loading weights" $RUN/test.log | tail -8 | cut -c1-300; exit 1
done
[ $rc -eq 0 ] || { echo "STOP: 10 OOM retries used"; exit 1; }
$PY eval_test_boot.py $RUN > $RUN/test_boot.txt 2>&1
brc=$?; cat $RUN/test_boot.txt
[ $brc -eq 0 ] || { echo "STOP: the test check failed (rc $brc)"; exit 1; }
$PY verify_pipeline.py --rl-dir $RUN --splits $G/splits_v1.json --fold 2 --split train --arm pend \
    --sepsim-path $G/trees/e1r_cf19400 > $RUN/verify_test.txt 2>&1
grep -q "PIPELINE VERIFICATION PASSED" $RUN/verify_test.txt || {
  echo "STOP: verify of the training files failed"; grep -E "FAIL" $RUN/verify_test.txt | head -10 | cut -c1-260; exit 1; }
echo "verify passed (training files unchanged)"
echo "TEST DONE $(date)"
