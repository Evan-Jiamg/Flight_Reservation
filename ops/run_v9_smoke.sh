#!/bin/bash
# v9 smoke (pend + GRPO): unit tests, live controller with w_aux, Task 1 (limit-aware verify), GRPO with
# 4 rollout workers + per-step / per-update timing, GPU tests. NOT a formal run.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v9; O=$G/v9_smoke
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export R0_BASE_URL=http://127.0.0.1:8029/v1 R0_MODEL=gpt-oss-120b R0_REASONING_EFFORT=low
export JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
export CONTROLLER_BASE_URL=http://127.0.0.1:8029/v1 OPENAI_API_KEY=local-vllm-unused
Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
pickgpu() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \
            awk -F', ' '{f=int(($2-$3)/1024); if (f>=42) print f" "$1}' | sort -nr | head -1 | cut -d" " -f2; }
rm -rf $O; mkdir -p $O; cd $C
echo "=== 0 R0 server"; curl -s -m 10 http://127.0.0.1:8029/v1/models | head -c 120; echo
echo "=== 1 unit tests (CPU; only files present in the snapshot)"
T=""; for f in test_pend_generate.py test_implicit_profile.py test_batching.py test_e16_port.py test_pend.py test_rl_reward.py \
    test_rl_controllers.py test_llm4_controller.py test_rl_advantages.py test_fit_prompts.py test_task2_episode.py \
    test_verify_pipeline.py test_planner_prompt_v3.py; do if [ -f $f ]; then T="$T $f"; else echo "missing in snapshot: $f"; fi; done
E1R_TREE=$G/trees/e1r_cf19400 V2FIX_TREE=/home/mzjiang/Sep-Simulator $PY -m pytest -q -p no:cacheprovider $T 2>&1 | tail -6
echo "=== 2 v4 controller against the live gpt-oss, with w_aux and aux stats"
$PY - <<'PYEOF' 2>&1 | tail -8
import json, sys
sys.path.insert(0, "/tmp2/mzjiang_usersim/grpo_planner/code_snapshots/pend_v9")
import rl_controllers as RC
cfg0 = RC.initial_cfg(version="v4", lambda_unparsed=1.0, lambda_hit_max_new=1.0, w_aux=1.0)
c = RC.make_controller("llm", cfg0, log_path="/tmp2/mzjiang_usersim/grpo_planner/v9_smoke/ctrl_probe.jsonl", every=1)
h = [{"update": 1, "split": "train", "reward_version": "v4", "reward_mean": 0.34, "shadow_reward_mean": 0.34,
      "components_mean": {"coverage": 0.97, "dist": -0.62, "turns": 6.5, "rate_unparsed": 0.0, "rate_hit_max_new": 0.0},
      "turn_hist": [0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 1], "p_h": [0.02, 0.2, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03],
      "kl": 0.0, "grad_norm": 0.022, "aux_weight": 1.0,
      "aux_stats": {"aux_n": 2, "aux_loss": 7.25, "aux_grad_norm": 30.3, "aux_p_correct_before": 0.5, "aux_p_correct_end": 5e-7},
      "task1_train": {"acc": 0.5, "end_at_final": 0.0, "end_at_nonfinal": 0.0}}]
cfg = c.propose(h)
print(type(c).__name__, "->", {k: cfg[k] for k in ("w_cov", "w_dist", "w_aux")}, "failures", c.n_failures)
rec = [json.loads(l) for l in open("/tmp2/mzjiang_usersim/grpo_planner/v9_smoke/ctrl_probe.jsonl")][-1]
print("finish_reason", rec.get("finish_reason"), "ok", rec.get("ok"), "error", rec.get("error"))
print("applied", rec.get("applied")); print("rationale", rec.get("rationale"))
PYEOF
GPU=""; while [ -z "$GPU" ]; do GPU=$(pickgpu); [ -z "$GPU" ] && sleep 60; done
echo "=== using GPU $GPU ($(date))"
echo "=== 3 Task 1 smoke (fold 2 validation, limit 3; limit-aware verify)"
$PY task1_v4.py --sessions fold-validation --fold 2 --limit 3 --planner-path $Q4 --gpu $GPU --workers 3 \
    --out $O/t1.jsonl > $O/t1.log 2>&1
echo "task1 rc=$?"; $PY verify_task1.py --gen $O/t1.jsonl --fold 2 2>&1 | tail -6
echo "=== 4 GRPO smoke (1 update, 4 rollout workers, validation at u0 and u1) $(date)"
$PY train_planner_rl.py --fold 2 --planner-path $Q4 --gpu $GPU --G 2 --scenarios-per-update 2 --updates 1 \
    --task1-convs 1 --val-every 1 --val-seeds 0 --rollout-workers 4 --out $O/rl > $O/rl.log 2>&1
echo "grpo rc=$? $(date)"; grep -v "Loading weights" $O/rl.log | tail -4 | cut -c1-300
$PY verify_pipeline.py --rl-dir $O/rl --splits $G/splits_v1.json --fold 2 --split train --arm pend \
    --sepsim-path $G/trees/e1r_cf19400 > $O/rl_verify.txt 2>&1
echo "verify rl rc=$?"; grep -E "FAIL|WARN|PASSED|FAILED" $O/rl_verify.txt | cut -c1-240 | head -20
$PY - <<'PYEOF'
import json, collections
O = "/tmp2/mzjiang_usersim/grpo_planner/v9_smoke/rl/"
for l in open(O + "updates.jsonl"):
    u = json.loads(l); st = u["learner_stats"]; a = u["train_aggregate"]
    print("UPDATE %d update_s %d timing %s" % (u["update"], u["update_s"], u.get("timing")))
    print("  grad_norm %s aux_grad_norm %s aux_loss %s aux_p_end %s w_aux %s effective %s" % (st.get("grad_norm"),
          st.get("aux_grad_norm"), st.get("aux_loss"), st.get("aux_p_correct_end"), u["cfg_used"].get("w_aux"), a.get("aux_weight")))
    print("  reward %.3f shadow %.3f comps %s" % (a["reward_mean"], a["shadow_reward_mean"], a["components_mean"]))
tot = collections.Counter(); n = collections.Counter()
for l in open(O + "rollouts.jsonl"):
    e = json.loads(l)["episode"]
    for s in e["trace"]:
        for k in ("planner_s", "speaker_s", "r0_s", "ledger_s"):
            if s.get(k) is not None:
                tot[k] += s[k]; n[k] += 1
print("PER-STEP MEAN SECONDS (train rollouts, 4 workers):", {k: round(tot[k] / n[k], 1) for k in tot}, "steps", dict(n))
for l in open(O + "validation.jsonl"):
    v = json.loads(l)
    if v["kind"] == "summary":
        print("VAL u%d validation_s %s n=%d sel=%s w1=%s cov=%s task1=%s" % (v["update"], v.get("validation_s"), v["n_episodes"],
              v["selection_score"], (v["turn_stats"] or {}).get("turn_w1"), (v["turn_stats"] or {}).get("coverage_mean"),
              {k: v["task1"][k] for k in ("term_f1", "premature", "k1_end_rate")} if v["task1"] else None))
PYEOF
echo "=== 5 GPU tests $(date)"
GPU=$GPU $PY test_rl_algos_server.py 2>&1 | tail -3
GPU=$GPU $PY test_batching_server.py 2>&1 | grep -v "Loading weights" | tail -6
echo "V9 SMOKE DONE $(date)"
