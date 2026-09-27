"""Paired bootstrap over validation conversations for the checkpoint re-selection (reselect.jsonl).
For every candidate vs u0 (untrained) and vs the re-selected best: selection score (Task 2 parts resampled, Task 1
term_f1 of each checkpoint fixed: greedy, deterministic), turn W1, mean |sim - human| turns and coverage.
Usage: python reselect_boot.py RUN_DIR"""
import collections
import json
import os
import random
import sys

run = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(run, "reselect.jsonl")) if l.strip()]
summ = {r["update"]: r for r in rows if r.get("kind") == "summary"}
eps = collections.defaultdict(dict)
for r in rows:
    if r.get("kind") == "episode":
        eps[r["update"]][(r["conversation_id"], r["seed"])] = r["episode"]      # later attempts overwrite
best = json.load(open(os.path.join(run, "reselect_best.json")))
bu = (best.get("best") or {}).get("update")
s0 = next(iter(summ.values()))
W_COV, W_W1, W_T1 = s0["w_sel_cov"], s0["w_sel_w1"], s0["w_sel_task1"]
meta0 = json.loads(open(os.path.join(run, "run_meta.jsonl")).readline())
T_MAX = int(meta0["config"]["selection_cfg"]["t_max"])       # validate() caps human turns at t_max for W1


def w1(sim, hum):
    hi = max(max(sim), max(hum))
    w = cs = ch = 0.0
    for k in range(hi + 1):
        cs += sim.count(k) / len(sim)
        ch += hum.count(k) / len(hum)
        w += abs(cs - ch)
    return w


def stats(u, keys):
    E = eps[u]
    ks = [k for k in keys if E[k].get("clean")]
    sim = [E[k]["emitted_user_turns"] for k in ks]
    hum = [min(int(E[k]["human_turns"]), T_MAX) for k in ks]
    cov = sum(float(E[k]["coverage"]) for k in ks) / len(ks)
    t1 = (summ[u].get("task1") or {}).get("term_f1") or 0.0
    W = w1(sim, hum)
    return {"sel": W_COV * cov - W_W1 * W + W_T1 * t1, "w1": W, "cov": cov,
            "abs_err": sum(abs(a - b) for a, b in zip(sim, hum)) / len(ks), "sim": sum(sim) / len(sim),
            "hum": sum(hum) / len(hum)}


print("candidates", sorted(summ), "re-selected best u%s" % bu, "seeds", best.get("seeds"))
for u in sorted(summ):
    s = summ[u]
    ts = s.get("turn_stats") or {}
    print("u%-3d sel=%s n=%s unclean=%s sim=%.2f human=%.2f w1=%s cov=%s term_f1=%s" % (
        u, None if s["selection_score"] is None else round(s["selection_score"], 4), s["n_episodes"],
        s["n_unclean_episodes"], ts.get("sim_turns_mean") or 0, ts.get("human_turns_mean") or 0,
        None if ts.get("turn_w1") is None else round(ts["turn_w1"], 3),
        None if ts.get("coverage_mean") is None else round(ts["coverage_mean"], 3), (s.get("task1") or {}).get("term_f1")))
# self-check: the full-sample recomputation must reproduce each summary's selection score
for u in sorted(summ):
    if summ[u]["selection_score"] is not None and u in eps:
        full = stats(u, sorted(eps[u]))["sel"]
        if abs(full - summ[u]["selection_score"]) > 1e-9:
            print("ERROR: recomputed selection u%d %.6f != summary %.6f" % (u, full, summ[u]["selection_score"]))
            sys.exit(1)
print("self-check: recomputed selection scores match the summaries")
pairs = [(0, u) for u in sorted(summ) if u != 0] + [(bu, u) for u in sorted(summ) if bu is not None and u not in (0, bu)]
for a, b in pairs:
    # only checkpoints with a finished re-selection summary (a stopped run may leave partial episodes)
    if a not in summ or b not in summ or a not in eps or b not in eps:
        continue
    keys = sorted(set(eps[a]) & set(eps[b]))
    convs = sorted({k[0] for k in keys})
    by_c = {c: [k for k in keys if k[0] == c] for c in convs}
    rng = random.Random(0)
    diffs = collections.defaultdict(list)
    for _ in range(10000):
        ks = [k for c in (rng.choice(convs) for _ in convs) for k in by_c[c]]
        try:
            x, y = stats(a, ks), stats(b, ks)
        except ZeroDivisionError:
            continue
        for m in ("sel", "w1", "abs_err", "cov"):
            diffs[m].append(y[m] - x[m])
    full_a, full_b = stats(a, keys), stats(b, keys)
    print("u%d vs u%d: %d paired episodes on %d conversations" % (b, a, len(keys), len(convs)))
    for m in ("sel", "w1", "abs_err", "cov"):
        d = sorted(diffs[m])
        lo, hi = d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]
        better = sum(1 for x in d if (x > 0 if m in ("sel", "cov") else x < 0)) / len(d)
        print("   %-7s u%d %.3f -> u%d %.3f  diff %.3f  95%% CI [%.3f, %.3f]  P(u%d better) %.3f" % (
            m, a, full_a[m], b, full_b[m], full_b[m] - full_a[m], lo, hi, b, better))
