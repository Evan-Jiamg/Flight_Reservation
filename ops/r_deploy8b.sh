G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v8
[ -d $C ] || { echo "pend_v8 missing: run part a first"; exit 1; }
cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - || exit 1
cat > $C/LOCAL_SHA256 <<'SHAEOF'
9c1a9dca8f18d7133663024e91cc598ac16275ea7d8a5d48b2cf8e2d6f3a0597  implicit_profile.py
76c2a5a663b6aa90df2befb3485e9e86fd8da16956efaafd0552ccd77cb295aa  planner_prompt_v3.py
4632bc96a647330ae4142a9b4289f4bca82913ee083cbd3ec3ae069b0cb64d7b  rl_algos.py
4dda38d41da76b74379aaf79da782201ff4c6508733abfb3a52428f4e915bc4e  rl_controllers.py
67b8f5a719b729b5bac82ee3dbd1334005bbf87d30fa8895827851f3c0a633ee  rl_reward.py
a2ea1892105d836b794efb19752199096aeef4e72fc5df5478a2004f73340e6c  rollout_v4.py
8a2b6695e2e60cf06d0e96e711cb35663f88f242b4ee4a9ab462ef92d6750620  task1_stop.py
1a843f6d16ed15b5fbd9da5b444dfd4ebcdebd3d5f6aecc9dfe48cb2052963ed  task1_v4.py
c358847b8163bd93a3272cb2679d8ecd13585d155edd17043c07e108da681db6  task2_env.py
86ce6abbe9b0375d91a95b172786daee28daf26574052aee09c6e197339040cf  test_batching_server.py
912ac4df75d23f02224a98e7b0265529320d2302ed2b1a05cecc6d80ff57d6e3  test_llm4_controller.py
f6faa3895506a477441abdbb7eabb3a006c473508382901e3cfb154716adc84f  test_pend.py
a79a8b23067c45556608d7c2cafd1081c8a3f8181266bfd16a273ccc8b99e5e2  test_rl_advantages.py
3513633498813e60e117fbee189684a7bfaf5d303d1d6ab228301af8465f618b  test_verify_pipeline.py
f0e251a16e0a311714761064a49380d67fa6227489d3d22ed1b749116b52d74a  train_planner_rl.py
8864591f642903d07c67a5a4298cca696816c47d8f48a0dcd249b58cb5b62c16  verify_pipeline.py
732074a2e9d4bff5a290169ae82ccbaad0ea7c4cdde635466a204fe35fa9bd97  verify_task1.py
83067f72ddbdeb4306b5058cdad920eb65be61bd91b5f5302a4201fd9241ddfc  style_select.py
c18fb651c6ad534d939b8c966357f033f9ff688b482134d57cd84f93edf18ceb  ditto_e16.py
3f97f0b20dc102cc4b3bb291008b49cc1f2fdf1350ee9783991ee5e0d3a7d365  fit_prompts.py
f7ca29f24ae62c6c2606ba48207bfff9685e3335349b2009907d988b32d0f555  batching.py
8688ee60a4c29456f080a78efd3405c42623df6155c2721b98d42d0196e10ed8  task2_episode.py
6d70c0c11dc789c6a762f8a67acc604dab4c7d256068763240b9c69cd21a2b46  diag_t1.py
SHAEOF
if ! sha256sum -c --quiet LOCAL_SHA256; then echo "SHA MISMATCH vs local worktree: not launching"; exit 1; fi
sha256sum *.py > SHA256SUMS && chmod a-w *.py
echo "part b ok; local sha check passed"
cat > $G/run_v8_smoke.sh <<'EOF'
#!/bin/bash
# v8 smoke (pend + GRPO). NOT a formal run: small limits, outputs under v8_smoke/.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v8; O=$G/v8_smoke
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
BENCH=/tmp2/hchsu/trec2026-usersim-benchmark
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export R0_BASE_URL=http://127.0.0.1:8029/v1 R0_MODEL=gpt-oss-120b R0_REASONING_EFFORT=low
export JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
export CONTROLLER_BASE_URL=http://127.0.0.1:8029/v1 OPENAI_API_KEY=local-vllm-unused
Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
pickgpu() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \
            awk -F', ' '{f=int(($2-$3)/1024); if (f>=42) print f" "$1}' | sort -nr | head -1 | cut -d" " -f2; }
rm -rf $O; mkdir -p $O; cd $C
echo "=== 0 R0 server"; curl -s -m 10 http://127.0.0.1:8029/v1/models | head -c 200; echo
echo "=== 1 benchmark internals (judge cache key, R0 history handling, Ledger threading)"
ls $BENCH/tools/metrics/judge.py $BENCH/tools/r0_client.py
grep -n "cache\|max_tokens\|def chat\|sha\|hash" $BENCH/tools/metrics/judge.py | head -40
grep -n "Thread\|Pool\|concurrent\|asyncio\|def update\|def reply\|trim\|truncat\|max_tokens\|messages\[" $BENCH/tools/r0_client.py | head -50
echo "=== 1b finished sessions in fold 2"
$PY - <<'PYEOF'
import json
sp = [f for f in json.load(open("/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json"))["folds"] if f["fold"] == 2][0]
recs = {}
for l in open("/home/mzjiang/v5-latency/data.jsonl"):
    if l.strip():
        r = json.loads(l); recs[r["conversation_id"]] = r
for k in ("train", "train_all", "validation", "validation_all", "test", "test_all"):
    ids = sp[k]
    fin = [c for c in ids if any(m.get("is_final") is True for m in recs[c].get("chat_messages", []))]
    print(k, len(ids), "finished", len(fin), "unfinished", [c[:10] for c in ids if c not in fin])
PYEOF
echo "=== 2 unit tests (CPU)"
E1R_TREE=$G/trees/e1r_cf19400 V2FIX_TREE=/home/mzjiang/Sep-Simulator $PY -m pytest -q -p no:cacheprovider \
  test_pend_generate.py test_implicit_profile.py test_batching.py test_e16_port.py test_pend.py test_rl_reward.py \
  test_rl_controllers.py test_llm4_controller.py test_rl_advantages.py test_fit_prompts.py test_task2_episode.py \
  test_verify_pipeline.py 2>&1 | tail -3
echo "=== 3 v4 controller against the live gpt-oss (one decision)"
$PY - <<'PYEOF' 2>&1 | tail -8
import json, sys
sys.path.insert(0, "/tmp2/mzjiang_usersim/grpo_planner/code_snapshots/pend_v8")
import rl_controllers as RC
cfg0 = RC.initial_cfg(version="v4", lambda_unparsed=1.0, lambda_hit_max_new=1.0)
c = RC.make_controller("llm", cfg0, log_path="/tmp2/mzjiang_usersim/grpo_planner/v8_smoke/ctrl_probe.jsonl", every=1)
h = [{"update": 1, "split": "train", "reward_version": "v4", "reward_mean": -0.4, "shadow_reward_mean": -0.4,
      "components_mean": {"coverage": 0.55, "dist": -1.2, "turns": 7.5, "rate_unparsed": 0.0, "rate_hit_max_new": 0.0},
      "turn_hist": [0, 0, 1, 1, 1, 2, 3, 2, 3, 1, 2], "p_h": [0.02, 0.2, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03], "kl": 0.0}]
print(type(c).__name__, "->", {k: c.propose(h)[k] for k in ("w_cov", "w_dist")}, "failures", c.n_failures)
rec = [json.loads(l) for l in open("/tmp2/mzjiang_usersim/grpo_planner/v8_smoke/ctrl_probe.jsonl")][-1]
print("finish_reason", rec.get("finish_reason"), "ok", rec.get("ok"), "error", rec.get("error"), "rationale", rec.get("rationale"))
PYEOF
GPU=""; while [ -z "$GPU" ]; do GPU=$(pickgpu); [ -z "$GPU" ] && sleep 60; done
echo "=== using GPU $GPU ($(date))"
echo "=== 4 Task 1 smoke (fold 2 validation, 3 sessions, spec defaults)"
$PY task1_v4.py --sessions fold-validation --fold 2 --limit 3 --planner-path $Q4 --gpu $GPU --workers 3 \
    --out $O/t1.jsonl > $O/t1.log 2>&1
echo "task1 rc=$? rows $(wc -l < $O/t1.jsonl) k1 $(wc -l < $O/t1.jsonl.k1.jsonl)"
$PY verify_task1.py --gen $O/t1.jsonl --fold 2 2>&1 | tail -12
$PY diag_t1.py $O/t1.jsonl 2>&1 | tail -6
$PY - <<'PYEOF'
import json
O = "/tmp2/mzjiang_usersim/grpo_planner/v8_smoke/"
rows = [json.loads(l) for l in open(O + "t1.jsonl")]
k1 = [json.loads(l) for l in open(O + "t1.jsonl.k1.jsonl")]
for r in rows:
    print("T1 %s t%d end_dec=%s greedy_ended=%s prev=%s blank=%s sel=%s dup=%s reasons=%s" % (
        r["conversation_id"][:8], r["turn_index"], r["planner_ends_session"], r["greedy_ended"], r["ended_by_prev_decision"],
        r["speaker_ended"], r["selected_index"], r["dup_redraws"], r["guard_reasons"]))
    print("   entry:", (r.get("profile_entry") or "")[:220])
    print("   greedy:", (r["greedy"] or "")[:160])
for k in k1:
    print("K1 %s n=%d ended=%s by_decision_at_n=%s blank=%s k1_own_decision=%s" % (
        k["conversation_id"][:8], k["gold_k"], k["ended"], k["ended_by_decision_at_n"], k["ended_speaker"], k["ended_planner_k1"]))
PYEOF
echo "=== 5 Task 2 smoke (fold 2 validation, 2 episodes, spec defaults: sampled Planner T=0.7)"
$PY rollout_v4.py --arm pend --fold 2 --split validation --planner-path $Q4 --gpu $GPU --out-dir $O/t2_smoke --limit 2 --smoke \
    --workers 2 > $O/t2.log 2>&1
echo "task2 rc=$?"; grep -E "emitted|DONE|ERROR" $O/t2.log | tail -4
$PY verify_pipeline.py --episodes $O/t2_smoke/pend.jsonl --meta $O/t2_smoke/run_meta_pend_rep0.json \
    --splits $G/splits_v1.json --fold 2 --split validation --arm pend --sepsim-path $G/trees/e1r_cf19400 > $O/t2_verify.txt 2>&1
echo "verify t2 rc=$?"; grep -E "FAIL|WARN|PASSED|FAILED" $O/t2_verify.txt | head -30
$PY - <<'PYEOF'
import json
for l in open("/tmp2/mzjiang_usersim/grpo_planner/v8_smoke/t2_smoke/pend.jsonl"):
    e = json.loads(l)
    print("T2 %s s%d emitted=%d human=%d end=%s cov=%.3f clean=%s counters=%s" % (e["conversation_id"][:8], e["seed"],
          e["emitted_user_turns"], e["human_turns"], e["end_kind"], e["coverage"], e["clean"], e["episode_counters"]))
    for s in e["trace"]:
        print("   t%d act=%s/%s end=%s goal_met=%s entry=%s" % (s["t"], s.get("move"), s.get("act"), s.get("ended_planner"),
              s.get("goal_met"), (s.get("profile_entry") or "")[:120]))
        print("      user:", (s.get("user") or "")[:150])
PYEOF
echo "=== 6 GRPO smoke (1 update, validation at u0 and u1)"
$PY train_planner_rl.py --fold 2 --planner-path $Q4 --gpu $GPU --G 2 --scenarios-per-update 2 --updates 1 \
    --task1-convs 1 --val-every 1 --val-seeds 0 --rollout-workers 2 --out $O/rl > $O/rl.log 2>&1
echo "grpo rc=$?"; tail -5 $O/rl.log
$PY verify_pipeline.py --rl-dir $O/rl --splits $G/splits_v1.json --fold 2 --split train \
    --arm pend --sepsim-path $G/trees/e1r_cf19400 > $O/rl_verify.txt 2>&1
echo "verify rl rc=$?"; grep -E "FAIL|WARN|PASSED|FAILED" $O/rl_verify.txt | head -30
$PY - <<'PYEOF'
import json
O = "/tmp2/mzjiang_usersim/grpo_planner/v8_smoke/rl/"
for l in open(O + "updates.jsonl"):
    u = json.loads(l); a = u["train_aggregate"]
    print("U%d reward %.3f shadow %.3f comps %s" % (u["update"], a["reward_mean"], a["shadow_reward_mean"], a["components_mean"]))
    print("   turn_hist %s unclean %s singletons %s aux_w %s task1 %s" % (a["turn_hist"], a["n_unclean_episodes"],
          a["n_dropped_singleton_episodes"], a["aux_weight"], a["task1_train"]))
    print("   learner", {k: u["learner_stats"].get(k) for k in ("loss", "kl", "ratio_init_maxdev", "n_tokens", "aux_n", "aux_loss", "aux_p_correct_before", "optimizer_steps")})
    print("   q", [round(x, 3) for x in (u.get("reward_ctx") or {}).get("q", [])], "controller_failures", u.get("controller_failures"))
for l in open(O + "validation.jsonl"):
    v = json.loads(l)
    if v["kind"] == "summary":
        print("VAL u%d n=%d unclean=%s score=%.3f sel=%.3f turn_stats=%s task1=%s anneal=%s" % (v["update"], v["n_episodes"],
              v["n_unclean_episodes"], v["mean_reward_selection"], v["selection_score"], v["turn_stats"],
              {k: v["task1"][k] for k in ("term_f1", "premature", "k1_end_rate", "decision_f1")} if v["task1"] else None, v["aux_anneal_start"]))
PYEOF
ls $O/rl/ckpt/*/rl_manifest.json
echo "=== 7 GPU tests"
GPU=$GPU $PY test_rl_algos_server.py 2>&1 | tail -6
GPU=$GPU $PY test_batching_server.py 2>&1 | tail -6
echo "V8 SMOKE DONE $(date)"
EOF
setsid nohup bash $G/run_v8_smoke.sh > $G/run_v8_smoke.log 2>&1 < /dev/null &
echo "smoke launched"
