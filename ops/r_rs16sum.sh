python3 - <<'PYEOF'
import json
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/"
for fn, tag in (("validation.jsonl", "2-seed"), ("reselect.jsonl", "8-seed")):
    try:
        rows = [json.loads(l) for l in open(O + fn) if l.strip()]
    except FileNotFoundError:
        continue
    for v in rows:
        if v.get("kind") == "summary":
            ts, t1 = v["turn_stats"] or {}, v["task1"] or {}
            print("%s u%-2d sel=%s n=%s sim %.2f human %.2f w1 %.3f cov %.3f bal_p %.3f auc %s logloss %.2f term_f1 %s" % (
                tag, v["update"], None if v["selection_score"] is None else round(v["selection_score"], 4), v.get("n_episodes"),
                ts.get("sim_turns_mean", 0), ts.get("human_turns_mean", 0), ts.get("turn_w1") or 0, ts.get("coverage_mean", 0),
                t1.get("bal_p", -1), None if t1.get("auc") is None else round(t1["auc"], 3), t1.get("logloss", -1), t1.get("term_f1")))
PYEOF
