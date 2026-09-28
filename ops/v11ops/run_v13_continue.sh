#!/bin/bash
# FORMAL GRPO continuation of runs/pend_f2_v11 with the user-approved intervention B (2026-09-27): from the next update
# w_dist = 1.0 and the controller may not lower it below 1.0 (intervention_w_dist_v13.json, applied once, recorded in
# run_meta / checkpoints, checked by verify rl.intervention). Code snapshot pend_v13 = pend_v12 + the intervention
# mechanism (train_planner_rl.py, rl_controllers.py, verify_pipeline.py, test_intervention.py).
# This script never stops anything: it waits until the v12 continuation (and its training process) has ended.
# Stop rule under B: best = best selection score over ALL validations (u5 included); a validation counts as
# "no improvement" only once the policy has had >= 5 updates under B (update >= applied_at + 4); stop after 2 such
# validations without improvement, or at 30.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v13; RUN=$G/runs/pend_f2_v11
IV=$G/intervention_w_dist_v13.json
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
sha256sum -c --quiet local_sha_v13.txt || { echo "STOP: pend_v13 snapshot sha mismatch"; exit 1; }
echo "=== waiting for the v12 continuation to end $(date)"
while pgrep -u mzjiang -f "run_v12_continue.sh|run_v12_guard.sh|train_planner_rl.py" > /dev/null; do sleep 30; done
echo "v12 continuation ended $(date): $(grep -E '^(STOP|early stop|V12 FORMAL DONE|verify)' $G/run_v12_continue.log | tail -2 | tr '\n' ' ' | cut -c1-200)"
if ls $RUN/ABORTED_u*.json > /dev/null 2>&1; then echo "STOP: an ABORTED marker exists"; exit 1; fi
LAST=$($PY -c "import json; print(json.load(open('$RUN/ckpt/LATEST.json'))['update'])")
if [ $((LAST % 5)) -eq 0 ] && ! $PY -c "
import json, sys
s = [json.loads(l) for l in open('$RUN/validation.jsonl') if l.strip()]
sys.exit(0 if any(v.get('kind') == 'summary' and v['update'] == $LAST for v in s) else 1)"; then
  echo "note: validation of u$LAST missing; the resume completes it first"; fi
$PY verify_pipeline.py --rl-dir $RUN --splits $G/splits_v1.json --fold 2 --split train --arm pend \
    --sepsim-path $G/trees/e1r_cf19400 > $RUN/verify_u${LAST}_v13.txt 2>&1
if ! grep -q "PIPELINE VERIFICATION PASSED" $RUN/verify_u${LAST}_v13.txt; then
  echo "STOP: re-verify of u1..u$LAST with pend_v13 failed"; grep -E "FAIL" $RUN/verify_u${LAST}_v13.txt | head -10 | cut -c1-260; exit 1; fi
echo "re-verify u1..u$LAST passed; intervention B applies before update $((LAST + 1))"
N=$(( (LAST / 5 + 1) * 5 ))
while [ $N -le 30 ]; do
  bash $G/start_servers5.sh > $G/servers_check.log 2>&1 || { echo "STOP: servers not available"; tail -5 $G/servers_check.log; exit 1; }
  take_train_gpu
  echo "=== chunk to update $N on GPU $GPU $(date)"
  $PY train_planner_rl.py --fold 2 --planner-path $Q4 --gpu $GPU --G 4 --scenarios-per-update 4 --task1-convs 4 \
      --updates $N --val-every 5 --rollout-workers 4 --out $RUN --resume --allow-code-change --intervention $IV >> $RUN/train.log 2>&1
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
iv = [json.loads(l)["intervention"] for l in open(O + "run_meta.jsonl") if l.strip() and json.loads(l).get("intervention")]
start = iv[0]["at_update"] + 4
s = sorted([json.loads(l) for l in open(O + "validation.jsonl") if l.strip() and json.loads(l).get("kind") == "summary"],
           key=lambda v: v["update"])
best, stale = None, 0
for v in s:
    x = v["selection_score"]
    if x is not None and (best is None or x > best):
        best, stale = x, 0
    elif v["update"] >= start:
        stale += 1
print("yes" if stale >= 2 else "no")
PYEOF
)
  echo "stop rule (2 validations under B without improvement): $STOP"
  [ "$STOP" = "yes" ] && { echo "early stop after update $N"; break; }
  N=$((N + 5))
done
cat $RUN/best.json 2>/dev/null
echo "V13 FORMAL DONE $(date)"
