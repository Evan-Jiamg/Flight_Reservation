G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v9
[ -d $C ] || { echo "pend_v9 missing: run part a first"; exit 1; }
cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - || exit 1
cat > $C/LOCAL_SHA256 <<'SHAEOF'
9c1a9dca8f18d7133663024e91cc598ac16275ea7d8a5d48b2cf8e2d6f3a0597  implicit_profile.py
76c2a5a663b6aa90df2befb3485e9e86fd8da16956efaafd0552ccd77cb295aa  planner_prompt_v3.py
064b5596bde7fe97ac225469200dff96c479e29a95530f4d7c72d9193ba1009b  rl_algos.py
8ea7316472e3b87c58f834e5ce49c85f85c764966d379f2a202e2df4da92a94d  rl_controllers.py
67b8f5a719b729b5bac82ee3dbd1334005bbf87d30fa8895827851f3c0a633ee  rl_reward.py
a2ea1892105d836b794efb19752199096aeef4e72fc5df5478a2004f73340e6c  rollout_v4.py
8a2b6695e2e60cf06d0e96e711cb35663f88f242b4ee4a9ab462ef92d6750620  task1_stop.py
6516546cec2f786342ff53c59ae8dfec935643715abc5b7a4a88f8515f80542f  task1_v4.py
b5ad8087c059db36e592c289db60261e85c4e5d2ff9d18fa690220be1c46c564  task2_env.py
86ce6abbe9b0375d91a95b172786daee28daf26574052aee09c6e197339040cf  test_batching_server.py
dbd071fb17d367a7918aec878e3c915923671931371170ec484274fa2a1baae4  test_llm4_controller.py
373cde1510c6e6e30d9cdc0682e346487579a95107c96b76755d708c917fcc41  test_pend.py
a79a8b23067c45556608d7c2cafd1081c8a3f8181266bfd16a273ccc8b99e5e2  test_rl_advantages.py
3513633498813e60e117fbee189684a7bfaf5d303d1d6ab228301af8465f618b  test_verify_pipeline.py
eaff6c7fec9c5c7118b2239cac77f7f2d7ee5ca288985ff212418aa555ca9ebe  train_planner_rl.py
8864591f642903d07c67a5a4298cca696816c47d8f48a0dcd249b58cb5b62c16  verify_pipeline.py
bc46a17cf0851edc53b97bc918905fd6997d1e8ce6e367e9e82be2043295264d  verify_task1.py
83067f72ddbdeb4306b5058cdad920eb65be61bd91b5f5302a4201fd9241ddfc  style_select.py
c18fb651c6ad534d939b8c966357f033f9ff688b482134d57cd84f93edf18ceb  ditto_e16.py
3f97f0b20dc102cc4b3bb291008b49cc1f2fdf1350ee9783991ee5e0d3a7d365  fit_prompts.py
f7ca29f24ae62c6c2606ba48207bfff9685e3335349b2009907d988b32d0f555  batching.py
8688ee60a4c29456f080a78efd3405c42623df6155c2721b98d42d0196e10ed8  task2_episode.py
6d70c0c11dc789c6a762f8a67acc604dab4c7d256068763240b9c69cd21a2b46  diag_t1.py
4cb6fbf8ce67fe09533ad2de87bd622986e8221d79b662be2b0142b13733c236  test_rl_algos_server.py
SHAEOF
if ! sha256sum -c --quiet LOCAL_SHA256; then echo "SHA MISMATCH vs local worktree: not launching"; exit 1; fi
sha256sum *.py > SHA256SUMS && chmod a-w *.py
echo "part b ok; local sha check passed"
cat > $G/run_v9_smoke.sh <<'SMOKEEOF'
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
SMOKEEOF
setsid nohup bash $G/run_v9_smoke.sh > $G/run_v9_smoke.log 2>&1 < /dev/null &
echo "smoke launched"
