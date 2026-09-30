#!/bin/bash
# One fold of the v17 pipeline (ops/SPEC_v17_sft_pend.md, user-approved 2026-09-30): SFT warm-up -> 5 fixed GRPO updates
# (or the length-drift stop), all spec defaults, fresh run dir runs/pend_f<F>_v17; a launch stopped by GPU OUT OF MEMORY
# (another job on the GPU; judged on THIS attempt's log lines only) is retried with --resume (the SFT examples / the
# latest checkpoint are reused), up to 10 times; anything else stops. Then verify_pipeline (v17 branch) must pass.
# The test evaluation is run_v17_test.sh (it starts / releases the placeholder and every GPU we hold).
# Needs the placeholder (gpu_holder2.py) to run; start_servers5.sh starts / reuses the servers (gpt-oss-120b on 8029,
# the Planner vLLM on 8031). The training GPU is taken from the placeholder dynamically and handed back after each
# attempt (never a GPU somebody else is using). Usage: run_v17_fold.sh F   (F = 0, 1 or 2)
set -uo pipefail
F=${1:-}
case $F in 0|1|2) ;; *) echo "STOP: fold must be 0, 1 or 2"; exit 1;; esac
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v17; RUN=$G/runs/pend_f${F}_v17; H=$G/hold
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # torch GPU indices = nvidia-smi / placeholder indices (fix round 3)
export R0_BASE_URL=http://127.0.0.1:8029/v1 R0_MODEL=gpt-oss-120b R0_REASONING_EFFORT=low
export JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
export CONTROLLER_BASE_URL=http://127.0.0.1:8029/v1 OPENAI_API_KEY=local-vllm-unused
export Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
held() { cat $H/status_$1 2>/dev/null || echo 0; }
settarget() { echo $2 > $H/target_$1.tmp && mv -f $H/target_$1.tmp $H/target_$1; }
TG=""
take_train_gpu() {
  n=0; until [ -f $H/role_train ]; do [ $((n % 600)) -eq 0 ] && echo "WAITING_GPU: no training GPU with room yet ($(date +%H:%M))"; n=$((n + 1)); sleep 1; done
  TG=$(cat $H/role_train); n=0
  until [ "$(held $TG)" -ge 45 ]; do [ $((n % 300)) -eq 0 ] && echo "WAITING_GPU: training GPU $TG holds $(held $TG)/45 GiB ($(date +%H:%M))"; n=$((n + 1)); sleep 1; done
  settarget $TG 0; until [ "$(held $TG)" -le 0 ]; do sleep 1; done
  GPU=$TG
}
giveback() { [ -n "$TG" ] && settarget $TG 45; TG=""; }     # the placeholder re-takes the training GPU
trap giveback EXIT
servers() { bash $G/start_servers5.sh > $G/servers_check_f${F}_v17.log 2>&1 || { echo "STOP: servers not available"; tail -5 $G/servers_check_f${F}_v17.log; exit 1; }; }
VERIFY="$PY verify_pipeline.py --rl-dir $RUN --splits $G/splits_v1.json --fold $F --split train --arm pend --sepsim-path $G/trees/e1r_cf19400"
finished() { [ -f $RUN/final.json ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('$RUN/final.json')).get('validated') else 1)"; }
cd $C || { echo "STOP: no snapshot $C"; exit 1; }
sha256sum -c --quiet local_sha_v17.txt || { echo "STOP: pend_v17 snapshot sha mismatch"; exit 1; }
pgrep -u mzjiang -f gpu_holder2.py > /dev/null || { echo "STOP: the placeholder is not running"; exit 1; }
mkdir -p $RUN
echo "=== FOLD $F v17 start $(date)"
if finished; then
  echo "fold $F: final.json is validated already - no training (a finished v17 run never trains again)"
else
  rc=1
  for attempt in $(seq 1 10); do
    servers
    take_train_gpu
    RES=""; [ -f $RUN/run_meta.jsonl ] && RES="--resume"      # SFT cache / checkpoints of an earlier attempt are reused
    SZ=$(stat -c %s $RUN/train.log 2>/dev/null || echo 0)
    echo "=== fold $F v17 training on GPU $GPU (attempt $attempt) $RES $(date)"
    $PY train_planner_rl.py --fold $F --planner-path $Q4 --gpu $GPU --rollout-workers 4 --out $RUN $RES >> $RUN/train.log 2>&1
    rc=$?; giveback; echo "train rc=$rc $(date)"
    [ $rc -eq 0 ] && break
    if tail -c +$((SZ + 1)) $RUN/train.log | grep -cE "CUDA out of memory|OutOfMemoryError" > /dev/null; then   # this attempt's lines only; grep -c reads all input (no SIGPIPE under pipefail)
      echo "fold $F: OOM (another job on the GPU) -> retry with --resume"; sleep 60; continue; fi
    echo "STOP: training failed"; grep -v "Loading weights" $RUN/train.log | tail -8 | cut -c1-300; exit 1
  done
  [ $rc -eq 0 ] || { echo "STOP: 10 OOM retries used"; exit 1; }
fi
finished || { echo "STOP: training ended without a validated final.json"; exit 1; }
$VERIFY > $RUN/verify_train.txt 2>&1
grep -q "PIPELINE VERIFICATION PASSED" $RUN/verify_train.txt || {
  echo "STOP: verify failed after training"; grep -E "FAIL" $RUN/verify_train.txt | head -12 | cut -c1-260; exit 1; }
echo "verify passed after training"
cat $RUN/final.json
$PY -c "
import json
rows = [json.loads(l) for l in open('$RUN/sft.jsonl') if l.strip()]
sel = [r for r in rows if r['kind'] == 'selection'][-1]
print('SFT choice: epoch', sel['chosen_epoch'], '(u0 = start policy)' if sel['u0_is_start_policy'] else '', sel['candidates'])"
echo "FOLD $F v17 TRAINING DONE $(date) -- test evaluation: run_v17_test.sh $F"
