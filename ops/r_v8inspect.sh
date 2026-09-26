G=/tmp2/mzjiang_usersim/grpo_planner; O=$G/v8_smoke; PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
echo "=== t2 verify"; grep -E "FAIL|WARN|PASSED|FAILED" $O/t2_verify.txt | cut -c1-260 | head -30
echo "=== main log: stage 5 lines"; grep -n "=== 5" -A6 $G/run_v8_smoke.log | cut -c1-200 | head -10
echo "=== unit test stage (full)"; sed -n '/=== 2 unit/,/=== 3 v4/p' $G/run_v8_smoke.log | head -20
echo "=== batching test context"; grep -n "=== 7" -A40 $G/run_v8_smoke.log | grep -v "Loading weights" | cut -c1-220 | tail -30
export PYTHONNOUSERSITE=1
$PY - <<'PYEOF'
import json, collections
O = "/tmp2/mzjiang_usersim/grpo_planner/v8_smoke/rl/"
for l in open(O + "updates.jsonl"):
    u = json.loads(l); st = u["learner_stats"]
    print("UPDATE", u["update"], "update_s", round(u["update_s"]), "grad_norm", st.get("grad_norm"), "aux_grad_norm", st.get("aux_grad_norm"),
          "aux_loss", st.get("aux_loss"), "aux_p_end", st.get("aux_p_correct_end"), "loss", st.get("loss"), "n_samples", u["n_samples"])
    print("  cfg w_aux", u["cfg_used"].get("w_aux"), "version", u["cfg_used"]["version"], "stop_credit comps", u["train_aggregate"]["components_mean"])
ro = [json.loads(l) for l in open(O + "rollouts.jsonl")]
for r in ro:
    e = r["episode"]
    print("ROLL %s g%d s=%d emitted=%d human=%d end=%s cov=%.3f clean=%s rollout_s=%d" % (e["conversation_id"][:8], r["replicate"], r["slot"],
          e["emitted_user_turns"], e["human_turns"], e["end_kind"], e["coverage"], e["clean"], r["rollout_s"]))
    for s in e["trace"]:
        g = s.get("planner_gen") or {}
        sm, nm = g.get("stop_mask"), g.get("note_mask")
        print("   t%d end=%s stop_mask=%s note_mask=%s gen=%d" % (s["t"], s.get("ended_planner"), sum(sm) if sm else None,
              sum(nm) if nm else None, len(g.get("gen_ids") or [])))
t1 = [json.loads(l) for l in open(O + "rollouts_task1.jsonl")]
for r in t1:
    print("T1GROUP %s t%d n=%d final=%s rewards=%s ends=%s valid=%s aux=%s" % (r["conversation_id"][:8], r["t"], r["n_real"], r["real_final"],
          [x["reward"] for x in r["samples"]], [x["ended_planner"] for x in r["samples"]], [x["decision_valid"] for x in r["samples"]],
          bool(r["samples"][0].get("aux"))))
times = collections.defaultdict(float)
for l in open(O + "validation.jsonl"):
    v = json.loads(l)
    if v["kind"] == "episode":
        e = v["episode"]
        print("VALEP u%d %s emitted=%d human=%d end=%s cov=%.3f clean=%s" % (v["update"], e["conversation_id"][:8], e["emitted_user_turns"],
              e["human_turns"], e["end_kind"], e["coverage"], e["clean"]))
PYEOF
ls -la $O/rl/llm_controller.jsonl 2>&1 | tail -1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
