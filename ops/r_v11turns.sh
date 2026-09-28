python3 - <<'PYEOF'
import json, collections
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
rows = [json.loads(l) for l in open(O + "rollouts.jsonl")]
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    e = r["episode"]
    by[r["update"]][r["conversation_id"][:8]].append((e["emitted_user_turns"], e["human_turns"], e["end_kind"]))
for u in sorted(by):
    print("U%d" % u)
    for cid, eps in sorted(by[u].items()):
        print("   %s human=%d sim=%s ends=%s" % (cid, eps[0][1], [x[0] for x in eps], collections.Counter(x[2] for x in eps)))
    allsim = [x[0] for v in by[u].values() for x in v]; allh = [x[1] for v in by[u].values() for x in v]
    print("   mean sim %.2f  mean human %.2f  mean(sim-human) %.2f  n=%d" % (sum(allsim) / len(allsim), sum(allh) / len(allh),
          sum(a - b for a, b in zip(allsim, allh)) / len(allsim), len(allsim)))
for l in open(O + "updates.jsonl"):
    u = json.loads(l); a = u["train_aggregate"]
    print("UPD %d dist=%.3f cov=%.3f task1 end_at_final=%s end_at_nonfinal=%s kl=%s" % (u["update"], a["components_mean"]["dist"],
          a["components_mean"]["coverage"], a["task1_train"]["end_at_final"], a["task1_train"]["end_at_nonfinal"], u["learner_stats"].get("kl")))
PYEOF
