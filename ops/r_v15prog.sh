python3 - <<'PYEOF'
import json, time
rows = [json.loads(l) for l in open("/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/reselect.jsonl") if l.strip()]
t0 = time.mktime(time.strptime("2026-09-27 14:11:32", "%Y-%m-%d %H:%M:%S"))
for u in (5, 25):
    eps = [r for r in rows if r.get("kind") == "episode" and r["update"] == u and r["time"] >= t0]
    t1 = [r for r in rows if r.get("kind") == "task1" and r["update"] == u and r["time"] >= t0]
    sm = [r for r in rows if r.get("kind") == "summary" and r["update"] == u]
    print("u%d: episodes %d/32 (clean %d), task1 %d/4, summary %s" % (u, len(eps), sum(1 for r in eps if r["episode"]["clean"]),
          len(t1), "done" if sm else "not yet"))
PYEOF
