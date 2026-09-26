#!/bin/bash
# FORMAL GRPO continuation of runs/pend_f2_v11 after the gate (user: start the formal run directly, 2026-09-26).
# Chunks of 5 updates (--resume --updates 5, 10, ... 30); after each chunk: verify must pass; stop when the checkpoint
# selection score has not improved on the best so far for 2 consecutive validations (approved stop rule), or at 30.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v11; RUN=$G/runs/pend_f2_v11
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export R0_BASE_URL=http://127.0.0.1:8029/v1 R0_MODEL=gpt-oss-120b R0_REASONING_EFFORT=low
export JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
export CONTROLLER_BASE_URL=http://127.0.0.1:8029/v1 OPENAI_API_KEY=local-vllm-unused
export Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
pickgpu() { SG=$(cat $G/server_gpu.txt 2>/dev/null || echo -1); nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \
            awk -F', ' -v sg=$SG '$1!=sg {f=int(($2-$3)/1024); if (f>=42) print f" "$1}' | sort -nr | head -1 | cut -d" " -f2; }
H=$G/hold
held() { cat $H/status_$1 2>/dev/null || echo 0; }
settarget() { echo $2 > $H/target_$1.tmp && mv -f $H/target_$1.tmp $H/target_$1; }
take_train_gpu() {   # wait until the holder holds 45 GiB on the training GPU, then hand it over (release to 0)
  TG=$(cat $H/train_gpu); n=0
  until [ "$(held $TG)" -ge 45 ]; do [ $((n % 300)) -eq 0 ] && echo "WAITING_GPU: training GPU $TG holds $(held $TG)/45 GiB ($(date +%H:%M))"; n=$((n + 1)); sleep 2; done
  settarget $TG 0; until [ "$(held $TG)" -le 0 ]; do sleep 1; done
  GPU=$TG
}
cd $C
echo "=== waiting for the gate to finish $(date)"
while pgrep -u mzjiang -f run_v11_formal4.sh > /dev/null; do sleep 60; done
L=$G/run_v11_formal4.log
if ! grep -q "V11 GATE DONE" $L || ! grep -q "train rc=0" $L || ! grep -q "PIPELINE VERIFICATION PASSED" $L \
   || ls $RUN/ABORTED_u*.json > /dev/null 2>&1 || [ ! -d $RUN/ckpt/u00001 ]; then
  echo "STOP: the gate did not pass (see $L); not continuing"; exit 1; fi
echo "gate passed -> formal continuation"
for N in 5 10 15 20 25 30; do
  bash $G/start_servers4.sh > $G/servers_check.log 2>&1 || { echo "STOP: servers not available"; tail -5 $G/servers_check.log; exit 1; }
  take_train_gpu
  echo "=== chunk to update $N on GPU $GPU $(date)"
  $PY train_planner_rl.py --fold 2 --planner-path $Q4 --gpu $GPU --G 4 --scenarios-per-update 4 --task1-convs 4 \
      --updates $N --val-every 5 --rollout-workers 4 --out $RUN --resume >> $RUN/train.log 2>&1
  rc=$?; settarget $TG 45; echo "train rc=$rc $(date)"
  if [ $rc -ne 0 ]; then echo "STOP: training failed"; grep -v "Loading weights" $RUN/train.log | tail -6 | cut -c1-300; exit 1; fi
  $PY verify_pipeline.py --rl-dir $RUN --splits $G/splits_v1.json --fold 2 --split train --arm pend \
      --sepsim-path $G/trees/e1r_cf19400 > $RUN/verify_u$N.txt 2>&1
  if ! grep -q "PIPELINE VERIFICATION PASSED" $RUN/verify_u$N.txt; then
    echo "STOP: verify failed after update $N"; grep -E "FAIL" $RUN/verify_u$N.txt | head -10 | cut -c1-260; exit 1; fi
  echo "verify passed after update $N"
  $PY - <<'PYEOF' > $RUN/progress_u$N.txt
import json
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
for l in open(O + "updates.jsonl"):
    u = json.loads(l); a = u["train_aggregate"]; st = u["learner_stats"]
    print("U%02d %4.0fs reward %.3f shadow %.3f cov %.3f dist %.3f turns %.2f kl %s mism %s tisw %s task1_end_final %s aux_w %s cfg %s" % (
        u["update"], u["update_s"], a["reward_mean"], a["shadow_reward_mean"], a["components_mean"].get("coverage", 0),
        a["components_mean"].get("dist", 0), a["components_mean"].get("turns", 0), st.get("kl"),
        st.get("behav_mismatch_mean"), st.get("tis_w_mean"), (a.get("task1_train") or {}).get("end_at_final"),
        a.get("aux_weight"), {k: round(u["cfg_used"].get(k, 0), 4) for k in ("w_cov", "w_dist", "w_aux", "lr")}))
for l in open(O + "validation.jsonl"):
    v = json.loads(l)
    if v["kind"] == "summary":
        print("VAL u%d sel=%s withheld=%s w1=%s cov=%s sim=%s human=%s task1=%s" % (v["update"], v["selection_score"],
              v.get("selection_withheld"), (v["turn_stats"] or {}).get("turn_w1"), (v["turn_stats"] or {}).get("coverage_mean"),
              (v["turn_stats"] or {}).get("sim_turns_mean"), (v["turn_stats"] or {}).get("human_turns_mean"),
              {k: v["task1"][k] for k in ("term_f1", "premature", "k1_end_rate")} if v["task1"] else None))
PYEOF
  tail -12 $RUN/progress_u$N.txt
  STOP=$($PY - <<'PYEOF'
import json
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
s = sorted([json.loads(l) for l in open(O + "validation.jsonl") if l.strip() and json.loads(l).get("kind") == "summary"],
           key=lambda v: v["update"])
best, stale = None, 0
for v in s:
    x = v["selection_score"]
    if x is not None and (best is None or x > best):
        best, stale = x, 0
    else:
        stale += 1
print("yes" if stale >= 2 else "no")
PYEOF
)
  echo "stop rule (2 validations without improvement): $STOP"
  [ "$STOP" = "yes" ] && { echo "early stop after update $N"; break; }
done
cat $RUN/best.json 2>/dev/null
echo "V11 FORMAL DONE $(date)"
