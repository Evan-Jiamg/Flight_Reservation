G=/tmp2/mzjiang_usersim/grpo_planner
python3 - <<'PYEOF'
import json, collections, glob, os
R = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/rollouts.jsonl"
by = collections.defaultdict(list)
for l in open(R):
    r = json.loads(l)
    ep = r.get("episode") or r
    u = r.get("update", r.get("policy_version"))
    t = ep.get("emitted_user_turns")
    if t is not None and (ep.get("clean", True)):
        by[u].append(t)
print("##### v16 fold 2 TRAIN rollouts: mean emitted user turns per update (clean episodes)")
for u in sorted(k for k in by if k is not None):
    print("u%s n=%d mean=%.2f" % (u, len(by[u]), sum(by[u]) / len(by[u])))
A = glob.glob("/tmp2/mzjiang_usersim/grpo_planner/archive/pend_f2_v11_u25*/")[0]
print("##### v11 archive reselect files")
for fn in glob.glob(os.path.join(A, "**", "reselect*.jsonl"), recursive=True) + glob.glob(os.path.join(A, "**", "*reselect*", "*.jsonl"), recursive=True):
    for l in open(fn):
        r = json.loads(l)
        if r.get("kind") == "summary":
            print(os.path.relpath(fn, A), "u%d" % r["update"], "sel", r["selection_score"], "seeds", r.get("val_seeds"))
PYEOF
grep -rl "0.176" $G/archive/pend_f2_v11_u25*/ 2>/dev/null | head -5
grep -rh "0\.17[56]" $G/archive/pend_f2_v11_u25*/RESUME.md 2>/dev/null | head -3 | cut -c1-200
