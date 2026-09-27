#!/bin/bash
# Checkpoint re-selection, narrowed by the user (2026-09-27): "只想請你對 u5、u25 進行重新評估，先不要訓練到 u30".
# u5 and u25 are validated again on the same validation ids with seeds 0..7 (same temperature, env, clean re-run
# rule and selection formula as training validation; train_planner_rl.py --reselect-seeds --reselect-updates),
# results in reselect.jsonl / reselect_best.json; training files are never written. Training does NOT continue.
# Code snapshot pend_v15 = pend_v14 + --reselect-updates (train_planner_rl.py, test_reselect.py).
# The final test run is NOT started (it needs the user's go and --final).
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v15; RUN=$G/runs/pend_f2_v11
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export R0_BASE_URL=http://127.0.0.1:8029/v1 R0_MODEL=gpt-oss-120b R0_REASONING_EFFORT=low
export JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
export CONTROLLER_BASE_URL=http://127.0.0.1:8029/v1 OPENAI_API_KEY=local-vllm-unused
export Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
H=$G/hold
held() { cat $H/status_$1 2>/dev/null || echo 0; }
settarget() { echo $2 > $H/target_$1.tmp && mv -f $H/target_$1.tmp $H/target_$1; }
take_train_gpu() {
  n=0; until [ -f $H/role_train ]; do [ $((n % 600)) -eq 0 ] && echo "WAITING_GPU: no training GPU with room yet ($(date +%H:%M))"; n=$((n + 1)); sleep 1; done
  TG=$(cat $H/role_train); n=0
  until [ "$(held $TG)" -ge 45 ]; do [ $((n % 300)) -eq 0 ] && echo "WAITING_GPU: training GPU $TG holds $(held $TG)/45 GiB ($(date +%H:%M))"; n=$((n + 1)); sleep 1; done
  settarget $TG 0; until [ "$(held $TG)" -le 0 ]; do sleep 1; done
  GPU=$TG
}
cd $C
sha256sum -c --quiet local_sha_v15.txt || { echo "STOP: pend_v15 snapshot sha mismatch"; exit 1; }
if pgrep -u mzjiang -f "train_planner_rl.py --fold 2" > /dev/null; then echo "STOP: a training / re-selection process is still running"; exit 1; fi
settarget $(cat $H/role_train) 45          # the stopped v14 re-selection had the training GPU; the holder takes it back
bash $G/start_servers5.sh > $G/servers_check.log 2>&1 || { echo "STOP: servers not available"; tail -5 $G/servers_check.log; exit 1; }
take_train_gpu
echo "=== re-selection of u5 and u25 with seeds 0..7 on GPU $GPU $(date)"
$PY train_planner_rl.py --fold 2 --planner-path $Q4 --gpu $GPU --G 4 --scenarios-per-update 4 --task1-convs 4 --updates 30 --val-every 5 --rollout-workers 4 --out $RUN --allow-code-change --reselect-seeds 0 1 2 3 4 5 6 7 --reselect-updates 5 25 >> $RUN/reselect.log 2>&1
rc=$?; settarget $TG 45; echo "reselect rc=$rc $(date)"
if [ $rc -ne 0 ]; then echo "STOP: re-selection failed"; grep -v "Loading weights" $RUN/reselect.log | tail -6 | cut -c1-300; exit 1; fi
$PY verify_pipeline.py --rl-dir $RUN --splits $G/splits_v1.json --fold 2 --split train --arm pend --sepsim-path $G/trees/e1r_cf19400 > $RUN/verify_reselect.txt 2>&1
if ! grep -q "PIPELINE VERIFICATION PASSED" $RUN/verify_reselect.txt; then
  echo "STOP: verify failed after the re-selection"; grep -E "FAIL" $RUN/verify_reselect.txt | head -10 | cut -c1-260; exit 1; fi
echo "verify passed after the re-selection"
cat $RUN/reselect_best.json
$PY $G/reselect_boot.py $RUN > $RUN/reselect_boot.txt 2>&1; cat $RUN/reselect_boot.txt
echo "V15 RESELECT DONE $(date)"
