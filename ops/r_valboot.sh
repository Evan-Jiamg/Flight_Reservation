python3 - <<'PYEOF'
import json, random, collections
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
V = [json.loads(l) for l in open(O + "validation.jsonl") if l.strip()]
eps = collections.defaultdict(dict)       # update -> (cid, seed) -> episode (last clean attempt)
for v in V:
    if v.get("kind") == "episode" and v["episode"].get("clean"):
        e = v["episode"]; eps[v["update"]][(e["conversation_id"], e.get("seed"))] = e
summ = {v["update"]: v for v in V if v.get("kind") == "summary"}
print("validation updates", sorted(eps), "episodes per update", {u: len(x) for u, x in eps.items()})
print("conversations", sorted({k[0][:8] for u in eps for k in eps[u]}))
def w1(sim, hum):
    hi = max(max(sim), max(hum)); w = cs = ch = 0.0
    for k in range(hi + 1):
        cs += sim.count(k) / len(sim); ch += hum.count(k) / len(hum); w += abs(cs - ch)
    return w
def stats(E, keys):
    sim = [E[k]["emitted_user_turns"] for k in keys]; hum = [E[k]["human_turns"] for k in keys]
    return {"w1": w1(sim, hum), "abs_err": sum(abs(a - b) for a, b in zip(sim, hum)) / len(keys),
            "cov": sum(E[k]["coverage"] for k in keys) / len(keys), "sim": sum(sim) / len(sim)}
for a, b in ((0, 5), (5, 10), (0, 10)):
    if a not in eps or b not in eps:
        continue
    keys = sorted(set(eps[a]) & set(eps[b]))
    convs = sorted({k[0] for k in keys})
    sa, sb = stats(eps[a], keys), stats(eps[b], keys)
    print("u%d vs u%d  paired episodes %d on %d conversations" % (a, b, len(keys), len(convs)))
    for m in ("w1", "abs_err", "cov", "sim"):
        print("   %-8s u%d %.3f  u%d %.3f" % (m, a, sa[m], b, sb[m]))
    rng = random.Random(0); diffs = collections.defaultdict(list)
    for _ in range(5000):
        pick = [rng.choice(convs) for _ in convs]
        ks = [k for c in pick for k in keys if k[0] == c]
        x, y = stats(eps[a], ks), stats(eps[b], ks)
        for m in ("w1", "abs_err", "cov"):
            diffs[m].append(y[m] - x[m])
    for m in ("w1", "abs_err", "cov"):
        d = sorted(diffs[m]); lo, hi = d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]
        better = sum(1 for x in d if (x < 0 if m != "cov" else x > 0)) / len(d)
        print("   diff u%d-u%d %-8s mean %.3f  95%% CI [%.3f, %.3f]  P(better) %.3f" % (b, a, m, sum(d) / len(d), lo, hi, better))
for u in sorted(summ):
    s = summ[u]; print("summary u%d sel=%s task1=%s" % (u, s["selection_score"],
          {k: s["task1"][k] for k in ("term_f1", "premature", "k1_end_rate")} if s.get("task1") else None))
PYEOF
