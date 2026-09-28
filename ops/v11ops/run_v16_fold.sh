#!/bin/bash
# One fold of the v16 pipeline (user 2026-09-28: "先跑 fold 0 和 fold 1"), the same procedure as fold 2:
#  formal GRPO run (spec defaults, fresh run dir runs/pend_f<F>_v16, chunks of 5 updates, verify after each chunk, stop
#  after 2 validations without improvement or at 30; a chunk stopped by GPU OUT OF MEMORY is retried from the latest
#  checkpoint, up to 10 times) -> 8-seed re-selection of every validated checkpoint -> verify -> paired bootstrap.
# Needs the placeholder (gpu_holder2.py) to run; start_servers5.sh starts / reuses the servers. Usage: run_v16_fold.sh F
set -uo pipefail
F=$1
case $F in 0|1) ;; *) echo "STOP: fold must be 0 or 1 (fold 2 is finished)"; exit 1;; esac
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v16; RUN=$G/runs/pend_f${F}_v16; H=$G/hold
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
  until [ "$(held $TG)" -ge 45 ]; do [ $((n % 300)) -eq 0 ] && echo "WAITING_GPU: training GPU $TG holds $(held $TG)/45 GiB ($(date +%H:%M))"; n=$((n + 1)); sleep 1; done
  settarget $TG 0; until [ "$(held $TG)" -le 0 ]; do sleep 1; done
  GPU=$TG
}
servers() { bash $G/start_servers5.sh > $G/servers_check_f$F.log 2>&1 || { echo "STOP: servers not available"; tail -5 $G/servers_check_f$F.log; exit 1; }; }
VERIFY="$PY verify_pipeline.py --rl-dir $RUN --splits $G/splits_v1.json --fold $F --split train --arm pend --sepsim-path $G/trees/e1r_cf19400"
cd $C
sha256sum -c --quiet local_sha_v16.txt || { echo "STOP: pend_v16 snapshot sha mismatch"; exit 1; }
pgrep -u mzjiang -f gpu_holder2.py > /dev/null || { echo "STOP: the placeholder is not running"; exit 1; }
mkdir -p $RUN
echo "=== FOLD $F start $(date)"
for N in 5 10 15 20 25 30; do
  for attempt in $(seq 1 10); do
    servers
    take_train_gpu
    RES=""; [ -f $RUN/ckpt/LATEST.json ] && RES="--resume"
    SZ=$(stat -c %s $RUN/train.log 2>/dev/null || echo 0)
    echo "=== fold $F chunk to update $N on GPU $GPU (attempt $attempt) $(date)"
    $PY train_planner_rl.py --fold $F --planner-path $Q4 --gpu $GPU --G 4 --scenarios-per-update 4 \
        --updates $N --val-every 5 --rollout-workers 4 --out $RUN $RES >> $RUN/train.log 2>&1
    rc=$?; settarget $TG 45; echo "train rc=$rc $(date)"
    [ $rc -eq 0 ] && break
    if tail -c +$((SZ + 1)) $RUN/train.log | grep -cE "CUDA out of memory|OutOfMemoryError" > /dev/null; then   # this attempt's lines only; grep -c reads all input (no SIGPIPE under pipefail)
      echo "fold $F: OOM (another job on the GPU) -> retry from the latest checkpoint"; sleep 60; continue; fi
    echo "STOP: training failed"; grep -v "Loading weights" $RUN/train.log | tail -6 | cut -c1-300; exit 1
  done
  [ $rc -eq 0 ] || { echo "STOP: 10 OOM retries used"; exit 1; }
  $VERIFY > $RUN/verify_u$N.txt 2>&1
  grep -q "PIPELINE VERIFICATION PASSED" $RUN/verify_u$N.txt || {
    echo "STOP: verify failed after update $N"; grep -E "FAIL" $RUN/verify_u$N.txt | head -10 | cut -c1-260; exit 1; }
  echo "verify passed after update $N"
  STOP=$(RUN=$RUN $PY - <<'PYEOF'
import json, os
O = os.environ["RUN"] + "/"
s = sorted([json.loads(l) for l in open(O + "validation.jsonl") if l.strip() and json.loads(l).get("kind") == "summary"],
           key=lambda v: v["update"])
best, stale = None, 0
for v in s:
    x = v["selection_score"]
    if x is None:
        continue                     # a withheld validation is no evaluation
    if best is None or x > best:
        best, stale = x, 0
    else:
        stale += 1
    print("VAL u%d sel %.4f" % (v["update"], x), file=__import__("sys").stderr)
print("yes" if stale >= 2 else "no")
PYEOF
)
  [ "$STOP" = "yes" ] || [ "$STOP" = "no" ] || { echo "STOP: the stop rule could not be computed ($STOP)"; exit 1; }
  echo "stop rule (2 validations without improvement): $STOP"
  [ "$STOP" = "yes" ] && { echo "early stop after update $N"; break; }
done
echo "=== fold $F training done $(date)"
cat $RUN/best.json
# 8-seed re-selection of every validated checkpoint (same procedure as fold 2)
CANDS=$(RUN=$RUN $PY -c "
import json, os
O = os.environ['RUN'] + '/'
print(' '.join(str(u) for u in sorted({json.loads(l)['update'] for l in open(O + 'validation.jsonl') if l.strip() and json.loads(l).get('kind') == 'summary'})))")
LAST=$($PY -c "import json; print(json.load(open('$RUN/ckpt/LATEST.json'))['update'])")
for attempt in $(seq 1 10); do
  servers
  take_train_gpu
  SZ=$(stat -c %s $RUN/reselect.log 2>/dev/null || echo 0)
  echo "=== fold $F re-selection of u$CANDS with seeds 0..7 on GPU $GPU (attempt $attempt) $(date)"
  $PY train_planner_rl.py --fold $F --planner-path $Q4 --gpu $GPU --G 4 --scenarios-per-update 4 --updates $LAST --val-every 5 \
      --rollout-workers 4 --out $RUN --reselect-seeds 0 1 2 3 4 5 6 7 --reselect-updates $CANDS >> $RUN/reselect.log 2>&1
  rc=$?; settarget $TG 45; echo "reselect rc=$rc $(date)"
  [ $rc -eq 0 ] && break
  if tail -c +$((SZ + 1)) $RUN/reselect.log | grep -cE "CUDA out of memory|OutOfMemoryError" > /dev/null; then
    echo "fold $F: OOM in the re-selection -> retry (finished summaries are reused)"; sleep 60; continue; fi
  echo "STOP: re-selection failed"; grep -v "Loading weights" $RUN/reselect.log | tail -8 | cut -c1-300; exit 1
done
[ $rc -eq 0 ] || { echo "STOP: 10 OOM retries used in the re-selection"; exit 1; }
$VERIFY > $RUN/verify_reselect.txt 2>&1
grep -q "PIPELINE VERIFICATION PASSED" $RUN/verify_reselect.txt || {
  echo "STOP: verify failed after the re-selection"; grep -E "FAIL" $RUN/verify_reselect.txt | head -10 | cut -c1-260; exit 1; }
echo "verify passed after the re-selection"
cat $RUN/reselect_best.json
$PY $G/reselect_boot.py $RUN > $RUN/reselect_boot.txt 2>&1
brc=$?; cat $RUN/reselect_boot.txt
[ $brc -eq 0 ] || { echo "STOP: the paired bootstrap failed (rc $brc)"; exit 1; }
echo "FOLD $F DONE $(date)"
