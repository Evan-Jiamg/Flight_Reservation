python3 - <<'PYEOF'
import json
R = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/"
rows = [json.loads(l) for l in open(R + "updates.jsonl")]
print("train_aggregate keys:", sorted(rows[0]["train_aggregate"].keys()))
for r in rows:
    ta = r["train_aggregate"]
    cm = ta.get("components_mean") or {}
    t1 = ta.get("task1") or ta.get("task1_train") or {}
    print(json.dumps({"u": r["update"], "n_ep": ta.get("n_episodes"), "groups": ta.get("n_groups"), "skip0": ta.get("n_groups_skipped_zero_std"),
                      "R_mean": round(ta.get("reward_mean", 0), 3), "R_std": round(ta.get("reward_std", 0), 3),
                      "cov": round(cm.get("coverage", -1), 3), "dist": round(cm.get("dist", -9), 3), "turns": cm.get("turns"),
                      "end_kind": ta.get("end_kind_frac"),
                      "t1": {k: t1.get(k) for k in ("n", "acc", "end_at_final", "end_at_nonfinal", "groups_skipped_zero_std", "n_base_groups", "n_informative_groups", "n_refill_groups", "refill_stop")},
                      "aux": ta.get("aux_stats"), "aux_w": ta.get("aux_weight")}))
print("##### real turn-count distribution p_h (train_all) and q of u1 / u10")
print("p_h", [round(x, 3) for x in rows[0]["reward_ctx"]["p_h"]])
print("q u1", [round(x, 3) for x in rows[0]["reward_ctx"]["q"]])
print("q u10", [round(x, 3) for x in rows[-1]["reward_ctx"]["q"]])
# Task 1 stop groups: rewards per position type from rollouts_task1.jsonl
by = {}
for l in open(R + "rollouts_task1.jsonl"):
    r = json.loads(l)
    key = (r.get("update"), bool(r.get("real_final")))
    s = r.get("samples") or []
    ends = [x.get("end") for x in s if isinstance(x, dict)]
    rw = r.get("rewards") or [x.get("reward") for x in s if isinstance(x, dict)]
    d = by.setdefault(key, [0, 0, 0])
    d[0] += 1
    d[1] += sum(1 for e in ends if e)
    d[2] += len(ends)
print("##### Task 1 stop groups: (update, real_final) -> groups, sampled ends / samples")
for k in sorted(by, key=lambda k: (k[0] or 0, k[1])):
    print(k, by[k])
print("rollouts_task1 row keys:", sorted(json.loads(open(R + "rollouts_task1.jsonl").readline()).keys()))
PYEOF
