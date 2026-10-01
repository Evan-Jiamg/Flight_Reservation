#!/bin/bash
# SPEC v18 §1.1 (user 2026-10-01, Q1): our act labeller (label_acts.py: gpt-oss-120b on :8029, effort low, our L2 moves,
# 3 votes, 2048-token completions, retries, the 8-gram refusal against the benchmark codebook) on fold 2's train_all
# (the reward labels) and validation_all (§7 only) -> labels_v18/act_labels_{train,val}_f2.jsonl (+ .meta.json, a
# 20-row .review.md for the USER). The training refuses the train labels until the user writes
# labels_v18/act_labels_train_f2.jsonl.APPROVED containing the label file's sha256 (printed below).
# Needs only gpt-oss (8029): reuses the running placeholder / servers or starts them; releases ONLY what it started
# (as run_v17_smoke.sh). No GPU of ours is used by the labeller itself. Never writes into the benchmark tree.
# Usage: run_v18_label.sh [F]   (F = 2)
set -uo pipefail
F=${1:-2}
[ "$F" = "2" ] || { echo "STOP: SPEC v18 is a fold-2 trial"; exit 1; }
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v18; H=${HOLD_DIR:-$G/hold}; L=$G/labels_v18
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared CUDA_DEVICE_ORDER=PCI_BUS_ID
export OPENAI_API_KEY=local-vllm-unused E1R_TREE=$G/trees/e1r_cf19400 SEPSIM_ACT_PRIOR=nostopclobber
STARTED_HOLDER=0; STARTED_SERVERS=0
release() {
  if [ "$STARTED_HOLDER" != "1" ]; then echo "LABEL: placeholder / servers left as found $(date)"; return; fi
  touch $H/stop; sleep 3; pkill -u mzjiang -f "python.* .*gpu_holder3\.py"
  if [ "$STARTED_SERVERS" != "1" ]; then echo "LABEL: placeholder stopped; servers were already up - left running"; return; fi
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
  echo "LABEL: GPUs released $(date)"
}
cd $C || { echo "STOP: no snapshot $C"; exit 1; }
sha256sum -c --quiet local_sha_v18.txt || { echo "STOP: pend_v18 snapshot sha mismatch"; exit 1; }
pgrep -u mzjiang -f "python.* .*(gpu_holder2|gpu_grab)\.py" > /dev/null && { echo "STOP: an old placeholder is running"; exit 1; }
if pgrep -u mzjiang -f "label_acts.py|train_planner_rl.py|eval_test_rl.py|smoke_v1[78].py|train_reranker.py|bench_tf_generate.py" > /dev/null; then
  echo "STOP: label_acts.py or another job of ours is already running"; exit 1; fi
mkdir -p $L
trap release EXIT
if ! curl -s -m 5 http://127.0.0.1:8029/v1/models | grep -q gpt-oss-120b; then
  pgrep -u mzjiang -f "vllm serve" > /dev/null || STARTED_SERVERS=1
  if ! pgrep -u mzjiang -f "python.* .*gpu_holder3\.py" > /dev/null; then
    STARTED_HOLDER=1; rm -rf $H; mkdir -p $H
    PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder3.py $H >> $G/gpu_holder3.log 2>&1 < /dev/null &
    sleep 10
  fi
  # gpt-oss only (OSS_ONLY=1, start_servers6.sh of the pend_v18 bundle): no Planner vLLM for the labeller
  OSS_ONLY=1 bash $G/start_servers6.sh > $G/servers_check_label_v18.log 2>&1 || { echo "STOP: a server failed to start"; tail -8 $G/servers_check_label_v18.log; exit 1; }
  grep -q "OSS_ONLY" $G/servers_check_label_v18.log || { echo "STOP: $G/start_servers6.sh does not know OSS_ONLY (deploy the v18 one)"; exit 1; }
fi
for SPLIT in train_all validation_all; do
  case $SPLIT in train_all) OUT=$L/act_labels_train_f${F}.jsonl;; validation_all) OUT=$L/act_labels_val_f${F}.jsonl;; esac
  echo "=== labelling $SPLIT -> $OUT $(date)"
  $PY label_acts.py --fold $F --split $SPLIT --out $OUT > $OUT.log 2>&1
  rc=$?; tail -4 $OUT.log
  [ $rc -eq 0 ] || { echo "STOP: labelling $SPLIT failed (rc $rc; failure rate > 5% stops; see $OUT.failed.json)"; exit 1; }
done
LSHA=$(sha256sum $L/act_labels_train_f${F}.jsonl | cut -d' ' -f1)
echo "LABELS DONE $(date). The USER reviews $L/act_labels_train_f${F}.jsonl.review.md (20 rows); to approve:"
echo "  echo $LSHA > $L/act_labels_train_f${F}.jsonl.APPROVED"
