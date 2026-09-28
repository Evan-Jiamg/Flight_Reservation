#!/bin/bash
# 8-seed re-selection of the v16 run (user 2026-09-28: "用 8 seeds 重新評估 u0、u5、u10"): the validated checkpoints
# u0, u5, u10 of runs/pend_f2_v16 are validated again on the same validation ids with seeds 0..7 (same procedure as
# training validation: train_planner_rl.py --reselect-seeds --reselect-updates), into reselect.jsonl /
# reselect_best.json; training files are never written. Same code snapshot (pend_v16) and arguments as the formal run
# (no --allow-code-change: the provenance check must pass unchanged). GPUs: the all-at-once placeholder (servers 91 GiB
# on one GPU, evaluation 45 GiB on another; a GPU in use by someone else is never taken). At the end: verify, paired
# bootstrap (reselect_boot.py), every GPU we hold released.
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
  echo "RESELECT: GPUs released $(date)"
}
cd $C
sha256sum -c --quiet local_sha_v16.txt || { echo "STOP: pend_v16 snapshot sha mismatch"; exit 1; }
if pgrep -u mzjiang -f "train_planner_rl.py|vllm serve|gpu_holder2|run_v16_formal" > /dev/null; then
  echo "STOP: our training / servers / placeholder already running"; exit 1; fi
[ -f $RUN/ckpt/u00010/state.json ] || { echo "STOP: checkpoint u10 missing"; exit 1; }
trap release EXIT
rm -rf $H; mkdir -p $H
PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder2.py $H >> $G/gpu_holder2.log 2>&1 < /dev/null &
sleep 10
bash $G/start_servers5.sh > $G/servers_check.log 2>&1 || { echo "STOP: servers not available"; tail -5 $G/servers_check.log; exit 1; }
n=0; until [ -f $H/role_train ]; do [ $((n % 600)) -eq 0 ] && echo "WAITING_GPU: no evaluation GPU with room yet ($(date +%H:%M))"; n=$((n + 1)); sleep 1; done
TG=$(cat $H/role_train); n=0
until [ "$(held $TG)" -ge 45 ]; do [ $((n % 300)) -eq 0 ] && echo "WAITING_GPU: GPU $TG holds $(held $TG)/45 GiB ($(date +%H:%M))"; n=$((n + 1)); sleep 1; done
settarget $TG 0; until [ "$(held $TG)" -le 0 ]; do sleep 1; done
echo "=== re-selection of u0 u5 u10 with seeds 0..7 on GPU $TG $(date)"
$PY train_planner_rl.py --fold 2 --planner-path $Q4 --gpu $TG --G 4 --scenarios-per-update 4 --updates 10 --val-every 5 \
    --rollout-workers 4 --out $RUN --reselect-seeds 0 1 2 3 4 5 6 7 --reselect-updates 0 5 10 >> $RUN/reselect.log 2>&1
rc=$?; echo "reselect rc=$rc $(date)"
if [ $rc -ne 0 ]; then echo "STOP: re-selection failed"; grep -v "Loading weights" $RUN/reselect.log | tail -8 | cut -c1-300; exit 1; fi
$PY verify_pipeline.py --rl-dir $RUN --splits $G/splits_v1.json --fold 2 --split train --arm pend \
    --sepsim-path $G/trees/e1r_cf19400 > $RUN/verify_reselect.txt 2>&1
if ! grep -q "PIPELINE VERIFICATION PASSED" $RUN/verify_reselect.txt; then
  echo "STOP: verify failed after the re-selection"; grep -E "FAIL" $RUN/verify_reselect.txt | head -10 | cut -c1-260; exit 1; fi
echo "verify passed after the re-selection"
cat $RUN/reselect_best.json
$PY $G/reselect_boot.py $RUN > $RUN/reselect_boot.txt 2>&1
brc=$?; cat $RUN/reselect_boot.txt
[ $brc -eq 0 ] || { echo "STOP: the paired bootstrap failed (rc $brc)"; exit 1; }
echo "V16 RESELECT DONE $(date)"
