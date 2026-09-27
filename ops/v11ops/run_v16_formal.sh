#!/bin/bash
# FORMAL GRPO run v16 (SPEC ops/SPEC_v16_grpo_opt.md, user 2026-09-28) on fold 2, fresh run dir runs/pend_f2_v16,
# code snapshot pend_v16. Every v16 setting is the spec default (task1_G 8, task1_convs 8, t1_trigger_margin 0.10,
# stop_sup_floor 0.5, Dr. GRPO, w_dist bound [1, 5], bal_p selection): NO setting is passed that differs from the
# spec, so no --ablation. Chunks of 5 updates (--resume after the first), verify after each chunk (with the split
# file), stop after 2 validations without improvement of the selection score, or at 30. GPUs: the all-at-once
# placeholder (gpu_holder2.py) decides the server GPU (gpt-oss 0.78 + Planner vLLM 0.15) and the training GPU (45 GiB).
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v16; RUN=$G/runs/pend_f2_v16
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
sha256sum -c --quiet local_sha_v16.txt || { echo "STOP: pend_v16 snapshot sha mismatch"; exit 1; }
mkdir -p $RUN
for N in 5 10 15 20 25 30; do
  bash $G/start_servers5.sh > $G/servers_check.log 2>&1 || { echo "STOP: servers not available"; tail -5 $G/servers_check.log; exit 1; }
  take_train_gpu
  RES=""; [ -f $RUN/ckpt/LATEST.json ] && RES="--resume"
  echo "=== chunk to update $N on GPU $GPU $(date)"
  $PY train_planner_rl.py --fold 2 --planner-path $Q4 --gpu $GPU --G 4 --scenarios-per-update 4 \
      --updates $N --val-every 5 --rollout-workers 4 --out $RUN $RES >> $RUN/train.log 2>&1
  rc=$?; settarget $TG 45; echo "train rc=$rc $(date)"
  if [ $rc -ne 0 ]; then echo "STOP: training failed"; grep -v "Loading weights" $RUN/train.log | tail -6 | cut -c1-300; exit 1; fi
  $PY verify_pipeline.py --rl-dir $RUN --splits $G/splits_v1.json --fold 2 --split train --arm pend \
      --sepsim-path $G/trees/e1r_cf19400 > $RUN/verify_u$N.txt 2>&1
  if ! grep -q "PIPELINE VERIFICATION PASSED" $RUN/verify_u$N.txt; then
    echo "STOP: verify failed after update $N"; grep -E "FAIL" $RUN/verify_u$N.txt | head -10 | cut -c1-260; exit 1; fi
  echo "verify passed after update $N"
  $PY - <<'PYEOF' > $RUN/progress_u$N.txt
import json
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/"
for l in open(O + "updates.jsonl"):
    u = json.loads(l); a = u["train_aggregate"]; st = u["learner_stats"]; t1 = a.get("task1_train") or {}
    print("U%02d %4.0fs reward %.3f cov %.3f dist %.3f turns %.2f kl %s mism %s rl_gn %s aux_gn %s adv %s aux_w %s "
          "t1 fin/early %s/%s refill %s/%s" % (
        u["update"], u["update_s"], a["reward_mean"], a["components_mean"].get("coverage", 0),
        a["components_mean"].get("dist", 0), a["components_mean"].get("turns", 0), st.get("kl"),
        st.get("behav_mismatch_mean"), st.get("rl_grad_norm"), st.get("aux_grad_norm"), st.get("adv_abs_mean"),
        a.get("aux_weight"), t1.get("end_at_final"), t1.get("end_at_nonfinal"), t1.get("n_refill_groups"), t1.get("refill_stop")))
for l in open(O + "validation.jsonl"):
    v = json.loads(l)
    if v["kind"] == "summary":
        ts, t1 = v["turn_stats"] or {}, v["task1"] or {}
        print("VAL u%d sel=%s withheld=%s w1=%s cov=%s sim=%s human=%s bal_p=%s auc=%s term_f1=%s d2=%s" % (
            v["update"], v["selection_score"], v.get("selection_withheld"), ts.get("turn_w1"), ts.get("coverage_mean"),
            ts.get("sim_turns_mean"), ts.get("human_turns_mean"), t1.get("bal_p"), t1.get("auc"), t1.get("term_f1"), v.get("d2")))
PYEOF
  tail -12 $RUN/progress_u$N.txt
  STOP=$($PY - <<'PYEOF'
import json
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/"
s = sorted([json.loads(l) for l in open(O + "validation.jsonl") if l.strip() and json.loads(l).get("kind") == "summary"],
           key=lambda v: v["update"])
best, stale = None, 0
for v in s:
    x = v["selection_score"]
    if x is None:
        continue                     # a withheld validation (unclean episodes after re-runs) is no evaluation
    if best is None or x > best:
        best, stale = x, 0
    else:
        stale += 1
print("yes" if stale >= 2 else "no")
PYEOF
)
  [ "$STOP" = "yes" ] || [ "$STOP" = "no" ] || { echo "STOP: the stop rule could not be computed ($STOP)"; exit 1; }
  echo "stop rule (2 validations without improvement): $STOP"
  [ "$STOP" = "yes" ] && { echo "early stop after update $N"; break; }
done
cat $RUN/best.json 2>/dev/null
echo "V16 FORMAL DONE $(date)"
