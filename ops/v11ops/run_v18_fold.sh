#!/bin/bash
# SPEC v18 training of fold 2 (ops/SPEC_v18_multiobj_rerank.md; user decisions 2026-10-01: one combined arm, the
# recommended Q1-Q9 defaults): multi-objective GRPO on the Planner + the human-likeness reranker in the selector,
# init / KL ref = the v17 u0 (runs/pend_f2_v17/ckpt/u00000, policy sha f67d6437...), <= 5 updates with the validation-
# guarded selection (final.json may name u0). Fresh run dir runs/pend_f2_v18.
# Refuses to start unless: the pend_v18 snapshot sha list checks; the act labels exist AND the user approved them
# (<labels>.APPROVED contains the label file's sha256 -- written after reading <labels>.review.md); the validation labels
# and the reranker json (train_reranker.py fit; its gate decides the selector) exist; the v17 u0 exists.
# A launch stopped by GPU OUT OF MEMORY / "GPU budget not available" (another job; THIS attempt's log lines only) is retried
# with --resume (rollouts of the same policy are reused; adv_scales.json is re-measured only after an aborted update 1),
# up to 10 times; anything else stops. A code fix mid-run: --resume --allow-code-change (never a restart): set
# ALLOW_CODE_CHANGE=1. Then verify_pipeline (v18 branch) must pass.
# Needs the placeholder (gpu_holder3.py) running; start_servers6.sh starts / reuses gpt-oss (8029) and the Planner vLLM
# (8031); the training GPU is taken from the placeholder and handed back after each attempt (never a GPU somebody else
# uses). Same exports as run_v17_fold.sh.
# Usage: run_v18_fold.sh [F]   (F = 2: the v18 trial is fold 2 only)
set -uo pipefail
F=${1:-2}
[ "$F" = "2" ] || { echo "STOP: SPEC v18 is a fold-2 trial (the init policy sha is fold 2's v17 u0)"; exit 1; }
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v18; RUN=$G/runs/pend_f${F}_v18; H=${HOLD_DIR:-$G/hold}   # HOLD_DIR: only for a cut-over next to an old placeholder
L=$G/labels_v18
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
  n=0; until [ -f $H/role_train ]; do [ $((n % 300)) -eq 0 ] && { holder_check; echo "WAITING_GPU: no committed placement yet: $(cat $H/plan 2>/dev/null) ($(date +%H:%M))"; }; n=$((n + 1)); sleep 1; done
  TG=$(cat $H/role_train); n=0
  settarget $TG $TN
  until [ "$(held $TG)" -ge $TN ]; do [ $((n % 300)) -eq 0 ] && { holder_check; echo "WAITING_GPU: training GPU $TG holds $(held $TG)/$TN GiB ($(date +%H:%M))"; }; n=$((n + 1)); sleep 1; done
  settarget $TG 0; n=0
  until [ "$(held $TG)" -le 0 ]; do n=$((n + 1)); [ $((n % 300)) -eq 0 ] && { holder_check; echo "WAITING_GPU: GPU $TG still holds $(held $TG) GiB after the hand-over ($(date +%H:%M))"; }; sleep 1; done
  GPU=$TG
}
giveback() { [ -n "$TG" ] && settarget $TG $TN; TG=""; }
trap giveback EXIT
servers() { bash $G/start_servers6.sh > $G/servers_check_f${F}_v18.log 2>&1 || { echo "STOP: a server failed to start (not a memory race; those are waited out inside start_servers6.sh)"; tail -8 $G/servers_check_f${F}_v18.log; exit 1; }; }
V18ARGS="--spec v18 --init-adapter $INIT --act-labels $LABELS --act-labels-val $LABELS_VAL --reranker $RERANK"
VERIFY="$PY verify_pipeline.py --rl-dir $RUN --splits $G/splits_v1.json --fold $F --split train --arm pend --sepsim-path $G/trees/e1r_cf19400"
finished() { [ -f $RUN/final.json ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('$RUN/final.json')).get('validated') else 1)"; }
cd $C || { echo "STOP: no snapshot $C"; exit 1; }
sha256sum -c --quiet local_sha_v18.txt || { echo "STOP: pend_v18 snapshot sha mismatch"; exit 1; }
for f in $LABELS $LABELS_VAL $RERANK $INIT/state.json $INIT/adapter/adapter_model.safetensors; do
  [ -f $f ] || { echo "STOP: missing $f (run_v18_label.sh / run_v18_reranker.sh first)"; exit 1; }; done
LSHA=$(sha256sum $LABELS | cut -d' ' -f1)
grep -q "$LSHA" $LABELS.APPROVED 2>/dev/null || { echo "STOP: the act labels are not approved: review $LABELS.review.md, then: echo $LSHA > $LABELS.APPROVED"; exit 1; }
$PY -c "import json; g = json.load(open('$RERANK'))['gate']; print('reranker gate:', 'PASSED' if g['passed'] else 'FAILED (selector stays Borda)', {k: g.get(k) for k in ('loco_acc', 'val_acc')})"
pgrep -u mzjiang -f "python.* .*(gpu_holder2|gpu_grab)\.py" > /dev/null && { echo "STOP: an old placeholder (gpu_holder2.py / gpu_grab.py) is running - finish the cut-over first (ops/v11ops/RESUME.md)"; exit 1; }
if pgrep -u mzjiang -f "train_planner_rl.py|eval_test_rl.py|smoke_v1[78].py|bench_tf_generate.py|train_reranker.py|label_acts.py" > /dev/null; then
  echo "STOP: another job of ours is running"; exit 1; fi
holder_check
mkdir -p $RUN
echo "=== FOLD $F v18 start $(date)"
if finished; then
  echo "fold $F: final.json is validated already - no training (a finished v18 run never trains again)"
else
  rc=1
  for attempt in $(seq 1 10); do
    servers
    take_train_gpu
    RES=""; [ -f $RUN/run_meta.jsonl ] && RES="--resume"
    [ -n "$RES" ] && [ "${ALLOW_CODE_CHANGE:-0}" = "1" ] && RES="$RES --allow-code-change"
    SZ=$(stat -c %s $RUN/train.log 2>/dev/null || echo 0)
    echo "=== fold $F v18 training on GPU $GPU (attempt $attempt) $RES $(date)"
    $PY train_planner_rl.py --fold $F --planner-path $Q4 --gpu $GPU --rollout-workers 4 --out $RUN $V18ARGS $RES >> $RUN/train.log 2>&1
    rc=$?; giveback; echo "train rc=$rc $(date)"
    [ $rc -eq 0 ] && break
    if tail -c +$((SZ + 1)) $RUN/train.log | grep -cE "CUDA out of memory|CUDA error: out of memory|OutOfMemoryError|GPU budget not available" > /dev/null; then
      echo "fold $F: OOM / GPU budget taken (another job on the GPU; exit 75 = budget) -> retry with --resume"; sleep 60; continue; fi
    echo "STOP: training failed"; grep -v "Loading weights" $RUN/train.log | tail -8 | cut -c1-300
    grep -q "training refused" $RUN/train.log && echo "NOTE: SPEC v18 §4.1 refusal (a frozen r_stop / r_act / r_turn): report to the user, see $RUN/adv_scales.refused.json"
    exit 1
  done
  [ $rc -eq 0 ] || { echo "STOP: 10 OOM retries used"; exit 1; }
fi
finished || { echo "STOP: training ended without a validated final.json"; exit 1; }
$VERIFY > $RUN/verify_train.txt 2>&1
grep -q "PIPELINE VERIFICATION PASSED" $RUN/verify_train.txt || {
  echo "STOP: verify failed after training"; grep -E "FAIL" $RUN/verify_train.txt | head -12 | cut -c1-260; exit 1; }
echo "verify passed after training"
grep -E "ALARM" $RUN/train.log | tail -5
$PY -c "
import json
f = json.load(open('$RUN/final.json'))
print('final u%s (%s), last u%s, GRPO adopted %s; candidates %s, J %s, task2 check %s' % (f['final_update'], f['stop_reason'],
      f['last_update'], f['grpo_adopted'], f['selection']['candidates'], f['selection']['J'], f['selection']['task2_check']))
s = json.load(open('$RUN/adv_scales.json'))
print('scales', s['scales'], 'frozen', s['frozen'], 'kappa', s['kappa'], 'kappa2 low confidence', s['kappa2_low_confidence'])"
echo "FOLD $F v18 TRAINING DONE $(date) -- test evaluation: run_v18_test.sh $F"
