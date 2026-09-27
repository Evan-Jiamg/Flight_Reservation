python3 - <<'PYEOF'
import json, collections, statistics
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
print("U | aux_w | aux p_correct(end) | T1 train end@final end@earlier | T2 train turns cov dist | w_dist lam_unp | rl_gn aux_gn | kl")
for l in open(O + "updates.jsonl"):
    u = json.loads(l); a = u["train_aggregate"]; st = u["learner_stats"]; c = a["components_mean"]; t1 = a.get("task1_train") or {}
    f = lambda x, n=3: "-" if x is None else round(x, n)
    print("U%02d | %s | %s(%s) | %s %s | %s %s %s | %s %s | %s %s | %s" % (
        u["update"], f(a.get("aux_weight"), 2), f(st.get("aux_p_correct_before")), f(st.get("aux_p_correct_end")),
        f(t1.get("end_at_final")), f(t1.get("end_at_nonfinal")), f(c.get("turns"), 2), f(c.get("coverage")), f(c.get("dist")),
        f(u["cfg_used"]["w_dist"], 2), f(u["cfg_used"]["lambda_unparsed"], 2), f(st.get("rl_grad_norm"), 4), f(st.get("aux_grad_norm"), 4),
        f(st.get("kl"), 4)))
print("--- point 5: can a person's length be predicted from what the Planner sees?")
rows = [json.loads(l) for l in open(O + "rollouts.jsonl")]
per = {}
for r in rows:
    e = r["episode"]; per[e["conversation_id"]] = (e["n_req"], e["human_turns"])
V = [json.loads(l) for l in open(O + "validation.jsonl") if l.strip()]
for v in V:
    if v.get("kind") == "episode":
        e = v["episode"]; per["VAL:" + e["conversation_id"]] = (e["n_req"], e["human_turns"])
xs = [(k[:12], n, h) for k, (n, h) in sorted(per.items(), key=lambda kv: kv[1])]
for k, n, h in xs:
    print("   %-12s n_req %2d human_turns %2d" % (k, n, h))
tr = [(n, h) for k, (n, h) in per.items() if not k.startswith("VAL:")]
n_, h_ = [x for x, _ in tr], [y for _, y in tr]
mn, mh = statistics.mean(n_), statistics.mean(h_)
cov = sum((a - mn) * (b - mh) for a, b in zip(n_, h_))
r = cov / ((sum((a - mn) ** 2 for a in n_) * sum((b - mh) ** 2 for b in h_)) ** 0.5)
print("train scenarios %d: corr(n_req, human_turns) = %.3f; human_turns mean %.2f sd %.2f" % (len(tr), r, mh, statistics.pstdev(h_)))
sp = json.load(open("/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json"))
f2 = [x for x in sp["folds"] if x["fold"] == 2][0]
print("train_all (Task 1 pool) human turns:", sorted(set(f2["train_all"]) - set(f2["train"])))
PYEOF
