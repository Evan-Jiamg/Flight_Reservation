python3 - <<'PYEOF'
import json
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/"
for l in open(O + "validation.jsonl"):
    v = json.loads(l)
    if v.get("kind") == "summary":
        ts, t1 = v["turn_stats"] or {}, v["task1"] or {}
        print("VAL u%d sel=%s withheld=%s metric=%s | sim %.2f human %.2f w1 %s cov %.3f | bal_p %.3f auc %s logloss %.2f term_f1 %s n_points %s invalid %s | d2 %s | %ss" % (
            v["update"], v["selection_score"], v.get("selection_withheld"), v.get("selection_task1_metric"),
            ts.get("sim_turns_mean", 0), ts.get("human_turns_mean", 0), ts.get("turn_w1"), ts.get("coverage_mean", 0),
            t1.get("bal_p", -1), t1.get("auc"), t1.get("logloss", -1), t1.get("term_f1"), t1.get("n_points"), t1.get("n_invalid"),
            v.get("d2"), v.get("validation_s")))
m = json.loads(open(O + "run_meta.jsonl").readline())
e = m["config"].get("env", {})
print("bench:", e.get("bench_pin"), e.get("bench_tools"), {k: v[:12] for k, v in (e.get("bench_data_sha256") or {}).items()})
print("args:", {k: m["config"]["args"][k] for k in ("task1_G", "task1_convs", "stop_sup_floor", "t1_trigger_margin", "G", "ablation")})
PYEOF
