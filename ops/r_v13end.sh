G=/tmp2/mzjiang_usersim/grpo_planner
grep -E "^(verify|stop rule|early stop|V13|STOP)" $G/run_v13_continue.log | tail -6
tail -2 $G/run_v13_guard.log | cut -c1-200
cat $G/run_v14_reselect.log | cut -c1-200
bash /tmp2/mzjiang_usersim/.cc_val15.sh 2>/dev/null
python3 - <<'PYEOF'
import json
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
for l in open(O + "validation.jsonl"):
    v = json.loads(l)
    if v.get("kind") == "summary" and v["update"] >= 20:
        ts = v["turn_stats"] or {}
        print("VAL u%d sel=%s sim=%.2f human=%.2f w1=%s cov=%.3f term_f1=%s" % (v["update"], v["selection_score"],
              ts.get("sim_turns_mean", 0), ts.get("human_turns_mean", 0), ts.get("turn_w1"), ts.get("coverage_mean", 0),
              (v.get("task1") or {}).get("term_f1")))
import os
p = O + "reselect.jsonl"
if os.path.exists(p):
    rows = [json.loads(l) for l in open(p)]
    print("reselect rows", len(rows), "summaries", [r["update"] for r in rows if r.get("kind") == "summary"])
PYEOF
tail -3 $G/runs/pend_f2_v11/reselect.log | grep -v "Loading weights" | cut -c1-200
