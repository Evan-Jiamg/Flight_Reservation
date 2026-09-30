#!/bin/bash
# One fold of the v17 pipeline (ops/SPEC_v17_sft_pend.md, user-approved 2026-09-30): SFT warm-up -> 5 fixed GRPO updates
# (or the length-drift stop), all spec defaults, fresh run dir runs/pend_f<F>_v17; a launch stopped by GPU OUT OF MEMORY
# (another job on the GPU; judged on THIS attempt's log lines only) is retried with --resume (the SFT examples / the
# latest checkpoint are reused), up to 10 times; anything else stops. Then verify_pipeline (v17 branch) must pass.
# The test evaluation is run_v17_test.sh (it starts / releases the placeholder and every GPU we hold).
# Needs the placeholder (gpu_holder3.py) to run; start_servers6.sh starts / reuses the servers (gpt-oss-120b on 8029,
# the Planner vLLM on 8031). The training GPU is taken from the placeholder dynamically and handed back after each
# attempt (never a GPU somebody else is using). 2026-09-30: gpu_holder3.py / start_servers6.sh (dynamic placement; a
# server start that loses the memory race is waited out and re-placed inside start_servers6.sh; only a real server
# error or the same race 3 times on one placement makes it exit 1 -> STOP here).
# Usage: run_v17_fold.sh F   (F = 0, 1 or 2)
set -uo pipefail
F=${1:-}
case $F in 0|1|2) ;; *) echo "STOP: fold must be 0, 1 or 2"; exit 1;; esac
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v17; RUN=$G/runs/pend_f${F}_v17; H=${HOLD_DIR:-$G/hold}   # HOLD_DIR: only for a cut-over next to an old placeholder
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
giveback() { [ -n "$TG" ] && settarget $TG $TN; TG=""; }     # the placeholder re-takes the training GPU
trap giveback EXIT
servers() { bash $G/start_servers6.sh > $G/servers_check_f${F}_v17.log 2>&1 || { echo "STOP: a server failed to start (not a memory race; those are waited out inside start_servers6.sh)"; tail -8 $G/servers_check_f${F}_v17.log; exit 1; }; }
VERIFY="$PY verify_pipeline.py --rl-dir $RUN --splits $G/splits_v1.json --fold $F --split train --arm pend --sepsim-path $G/trees/e1r_cf19400"
finished() { [ -f $RUN/final.json ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('$RUN/final.json')).get('validated') else 1)"; }
cd $C || { echo "STOP: no snapshot $C"; exit 1; }
sha256sum -c --quiet local_sha_v17.txt || { echo "STOP: pend_v17 snapshot sha mismatch"; exit 1; }
pgrep -u mzjiang -f "python.* .*(gpu_holder2|gpu_grab)\.py" > /dev/null && { echo "STOP: an old placeholder (gpu_holder2.py / gpu_grab.py) is running - finish the cut-over first (ops/v11ops/RESUME.md)"; exit 1; }
holder_check                             # the placeholder runs and writes $H/plan (else STOP)
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
    if tail -c +$((SZ + 1)) $RUN/train.log | grep -cE "CUDA out of memory|CUDA error: out of memory|OutOfMemoryError|GPU budget not available" > /dev/null; then   # this attempt's lines only; grep -c reads all input (no SIGPIPE under pipefail)
      echo "fold $F: OOM / GPU budget taken (another job on the GPU; exit 75 = budget) -> retry with --resume"; sleep 60; continue; fi
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
