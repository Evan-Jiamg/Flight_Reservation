python3 - <<'PYEOF'
import json, collections
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
sp = json.load(open("/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json"))
f = [x for x in sp["folds"] if x["fold"] == 2][0]
print("fold2 sizes:", {k: len(v) for k, v in f.items() if isinstance(v, list)})
print("--- per update: Task1 train (RL groups) + aux")
for l in open(O + "updates.jsonl"):
    u = json.loads(l); a = u["train_aggregate"]; st = u["learner_stats"]
    t1 = a.get("task1_train") or {}
    ax = u.get("aux_stats") or {k: st.get(k) for k in ("aux_n", "aux_loss", "aux_p_correct_before", "aux_p_correct_end")}
    print("U%02d t1 %s | aux_w %s aux_n %s p_correct %s p_correct_end %s | rl_gn %s aux_gn %s" % (
        u["update"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t1.items()},
        a.get("aux_weight"), st.get("aux_n"), None if st.get("aux_p_correct_before") is None else round(st["aux_p_correct_before"], 3),
        None if st.get("aux_p_correct_end") is None else round(st["aux_p_correct_end"], 3),
        None if st.get("rl_grad_norm") is None else round(st["rl_grad_norm"], 4),
        None if st.get("aux_grad_norm") is None else round(st["aux_grad_norm"], 4)))
print("--- Task1 RL rows: per update, P(end) sampled at the real final msg vs at an earlier msg")
rows = [json.loads(l) for l in open(O + "rollouts_task1.jsonl") if l.strip()]
k0 = rows[0]["samples"][0].keys() if rows and rows[0]["samples"] else []
print("sample keys:", sorted(k0))
by = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in rows:
    ends = [1 if s.get("end") or s.get("end_session") or s.get("pred_end") else 0 for s in r["samples"]]
    b = by[r["update"]]
    if r["real_final"]:
        b[0] += sum(ends); b[1] += len(ends)
    else:
        b[2] += sum(ends); b[3] += len(ends)
for u in sorted(by):
    b = by[u]
    print("U%02d P(end|final) %d/%d  P(end|earlier) %d/%d" % (u, b[0], b[1], b[2], b[3]))
print("--- validation Task 1")
for l in open(O + "validation.jsonl"):
    v = json.loads(l)
    if v.get("kind") == "summary":
        print("VAL u%d task1 %s" % (v["update"], v.get("task1")))
    if v.get("kind") == "task1" and v["update"] in (0, 25):
        t = v["task1"]
        print("   u%d %s keys %s" % (v["update"], v["conversation_id"][:8], sorted(t)[:12] if isinstance(t, dict) else type(t)))
PYEOF
