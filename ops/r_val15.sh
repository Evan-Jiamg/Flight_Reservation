python3 - <<'PYEOF'
import json
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
for l in open(O + "validation.jsonl"):
    v = json.loads(l)
    if v.get("kind") == "summary":
        ts = v["turn_stats"] or {}
        print("VAL u%d sel=%s sim=%.2f human=%.2f w1=%s cov=%.3f term_f1=%s %ss" % (v["update"], v["selection_score"],
              ts.get("sim_turns_mean", 0), ts.get("human_turns_mean", 0), ts.get("turn_w1"), ts.get("coverage_mean", 0),
              (v.get("task1") or {}).get("term_f1"), v.get("validation_s")))
PYEOF
