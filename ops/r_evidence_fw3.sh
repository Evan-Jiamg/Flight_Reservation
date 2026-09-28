G=/tmp2/mzjiang_usersim/grpo_planner
python3 - <<'PYEOF'
import json, glob
A = glob.glob("/tmp2/mzjiang_usersim/grpo_planner/archive/pend_f2_v11_u25*/run/reselect.jsonl")[0]
print("##### v11 archive run/reselect.jsonl summaries")
for l in open(A):
    r = json.loads(l)
    if r.get("kind") == "summary" or "selection_score" in r:
        print({k: r.get(k) for k in ("kind", "update", "selection_score", "val_seeds", "reselect", "selection_formula", "selection_task1_metric")})
R = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/"
print("##### v16 fold 2 run_meta start/resume times and updates timing")
import datetime
for l in open(R + "run_meta.jsonl"):
    r = json.loads(l)
    print("run_meta", r.get("kind"), datetime.datetime.fromtimestamp(r["time"]).strftime("%m-%d %H:%M"))
last = None
for l in open(R + "updates.jsonl"):
    r = json.loads(l)
    t = r.get("time") or (r.get("timing") or {}).get("end")
    print("update", r.get("update"), "wall_s", r.get("wall_s"), "time", datetime.datetime.fromtimestamp(t).strftime("%m-%d %H:%M") if t else None)
for l in open(R + "validation.jsonl"):
    r = json.loads(l)
    if r.get("kind") == "summary":
        print("validation summary u%d at %s (validation_s %s)" % (r["update"], datetime.datetime.fromtimestamp(r["time"]).strftime("%m-%d %H:%M"), r.get("validation_s")))
PYEOF
