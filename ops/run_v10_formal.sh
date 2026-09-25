#!/bin/bash
# FORMAL GRPO run, fold 2, pend v10 -- gate stage: --updates 1 (validation at u0 + update 1), then STOP.
# Continued later with --resume --updates N only after the gate checks pass and the user agrees.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v10; RUN=$G/runs/pend_f2_v10
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export R0_BASE_URL=http://127.0.0.1:8029/v1 R0_MODEL=gpt-oss-120b R0_REASONING_EFFORT=low
export JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
export CONTROLLER_BASE_URL=http://127.0.0.1:8029/v1 OPENAI_API_KEY=local-vllm-unused
Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
pickgpu() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \
            awk -F', ' '{f=int(($2-$3)/1024); if (f>=42) print f" "$1}' | sort -nr | head -1 | cut -d" " -f2; }
cd $C
echo "=== 0 preflight $(date)"
if ! curl -s -m 15 http://127.0.0.1:8029/v1/models | grep -q gpt-oss-120b; then
  echo "ABORT: the gpt-oss-120b server on port 8029 is not answering (not restarting it without the user)"; exit 1; fi
T=""; for f in test_pend_generate.py test_implicit_profile.py test_batching.py test_e16_port.py test_pend.py test_rl_reward.py \
    test_rl_controllers.py test_llm4_controller.py test_rl_advantages.py test_fit_prompts.py test_task2_episode.py \
    test_verify_pipeline.py test_planner_prompt_v3.py test_audit4.py test_judge_wrapper.py; do
    if [ -f $f ]; then T="$T $f"; else echo "missing in snapshot: $f"; fi; done
E1R_TREE=$G/trees/e1r_cf19400 V2FIX_TREE=/home/mzjiang/Sep-Simulator $PY -m pytest -q -p no:cacheprovider $T > $G/v10_unit.txt 2>&1
rc=$?; tail -4 $G/v10_unit.txt
if [ $rc -ne 0 ]; then echo "ABORT: unit tests failed (see $G/v10_unit.txt)"; exit 1; fi
echo "=== 1 waiting for the v9 smoke to finish and for a free GPU $(date)"
while pgrep -u mzjiang -f run_v9_smoke.sh > /dev/null; do sleep 60; done
GPU=""; while [ -z "$GPU" ]; do GPU=$(pickgpu); [ -z "$GPU" ] && sleep 60; done
echo "=== 2 FORMAL GRPO gate run on GPU $GPU $(date)"
if [ -e $RUN/ckpt/LATEST.json ]; then echo "ABORT: $RUN already has checkpoints"; exit 1; fi
mkdir -p $RUN
$PY train_planner_rl.py --fold 2 --planner-path $Q4 --gpu $GPU --G 4 --scenarios-per-update 4 --task1-convs 4 \
    --updates 1 --val-every 5 --rollout-workers 4 --out $RUN > $RUN/train.log 2>&1
echo "train rc=$? $(date)"; grep -v "Loading weights" $RUN/train.log | tail -4 | cut -c1-300
echo "=== 3 verify"
$PY verify_pipeline.py --rl-dir $RUN --splits $G/splits_v1.json --fold 2 --split train --arm pend \
    --sepsim-path $G/trees/e1r_cf19400 > $RUN/verify.txt 2>&1
echo "verify rc=$?"; grep -E "FAIL|WARN|PASSED|FAILED" $RUN/verify.txt | cut -c1-260 | head -30
echo "=== 4 gate metrics"
$PY - <<'PYEOF'
import json, collections
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v10/"
for l in open(O + "updates.jsonl"):
    u = json.loads(l); st = u["learner_stats"]; a = u["train_aggregate"]
    print("UPDATE %d update_s %d timing %s" % (u["update"], u["update_s"], u.get("timing")))
    print("  rl_grad_norm %s aux_grad_norm %s grad_norm(total) %s aux_loss %s aux_p_end %s optimizer_steps %s" % (
          st.get("rl_grad_norm"), st.get("aux_grad_norm"), st.get("grad_norm"), st.get("aux_loss"),
          st.get("aux_p_correct_end"), st.get("optimizer_steps")))
    print("  ratio_init_maxdev %s kl %s n_samples %s n_tokens %s" % (st.get("ratio_init_maxdev"), st.get("kl"), u["n_samples"], st.get("n_tokens")))
    print("  reward %.3f shadow %.3f comps %s" % (a["reward_mean"], a["shadow_reward_mean"], a["components_mean"]))
    print("  turn_hist %s unclean %s singletons %s aux_w %s task1 %s" % (a["turn_hist"], a["n_unclean_episodes"],
          a["n_dropped_singleton_episodes"], a["aux_weight"], a["task1_train"]))
    print("  q %s p_h %s" % ([round(x, 3) for x in u["reward_ctx"]["q"]], [round(x, 3) for x in u["reward_ctx"]["p_h"]]))
tot = collections.Counter(); n = collections.Counter(); inc = collections.Counter(); eps = 0
for l in open(O + "rollouts.jsonl"):
    e = json.loads(l)["episode"]; eps += 1
    for k, v in (e.get("episode_counters") or {}).items():
        inc[k] += v
    for s in e["trace"]:
        for k in ("planner_s", "speaker_s", "r0_s", "ledger_s"):
            if s.get(k) is not None:
                tot[k] += s[k]; n[k] += 1
print("EPISODES %d incidents %s" % (eps, dict(inc)))
print("PER-STEP MEAN SECONDS (4 workers, wall incl. waiting):", {k: round(tot[k] / n[k], 1) for k in tot}, "steps", dict(n))
for l in open(O + "validation.jsonl"):
    v = json.loads(l)
    if v["kind"] == "summary":
        print("VAL u%d validation_s %s n=%d unclean=%s withheld=%s sel=%s turn_stats=%s task1=%s" % (v["update"], v.get("validation_s"),
              v["n_episodes"], v["n_unclean_episodes"], v.get("selection_withheld"), v["selection_score"], v["turn_stats"],
              {k: v["task1"][k] for k in ("term_f1", "premature", "k1_end_rate", "n_emitted_capped_turns")} if v["task1"] else None))
PYEOF
ls $RUN/ckpt/
echo "V10 GATE DONE $(date)"
