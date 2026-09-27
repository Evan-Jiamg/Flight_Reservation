python3 - <<'PYEOF'
import json, collections
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
V = [json.loads(l) for l in open(O + "validation.jsonl") if l.strip()]
print("=== validation Task 1 (greedy, real conversations): goal_met / ended per turn")
for v in V:
    if v.get("kind") == "task1" and v["update"] in (0, 5, 25):
        t = v["task1"]
        print("u%-2d %s n=%d  %s" % (v["update"], v["conversation_id"][:8], t["n_real"],
              " ".join("t%d:%s%s%s" % (x["t"], {True: "G", False: "g", None: "?"}.get(x.get("goal_met"), str(x.get("goal_met"))[:3]),
                                       "E" if x["ended_planner"] else "-", "*" if x["real_final"] else "") for x in t["turns"])))
d0 = [x for v in V if v.get("kind") == "task1" for x in v["task1"]["turns"] if x["t"] >= 2]
print("sample turn keys:", sorted(d0[0].keys()))
print("sample planner_diag keys:", sorted((d0[0].get("planner_diag") or {}).keys())[:40])
tab = collections.Counter((x.get("goal_met"), x["real_final"]) for x in d0)
print("goal_met x real_final (all validation updates, t>=2):", dict(tab))
print("=== training Task 1 samples: planner_diag keys")
T1 = [json.loads(l) for l in open(O + "rollouts_task1.jsonl") if l.strip()]
s = T1[0]["samples"][0]
print(sorted((s.get("planner_diag") or {}).keys())[:40])
gm = collections.Counter()
for r in T1:
    for x in r["samples"]:
        dg = x.get("planner_diag") or {}
        gm[(dg.get("goal_met", dg.get("goal_status")), r["real_final"], x["ended_planner"])] += 1
print("train (goal_met, real_final, ended) counts:", sorted(gm.items(), key=lambda kv: -kv[1])[:14])
print("=== Task 2 rollouts: per step goal fields")
R = [json.loads(l) for l in open(O + "rollouts.jsonl")][-16:]
st = R[0]["episode"]["trace"][1]
print(sorted(st.keys())[:60])
PYEOF
