"""Check + paired bootstrap of a test evaluation (eval_test_rl.py -> RUN/test.jsonl [+ RUN/test_base.jsonl]), SPEC v17 §5
(user 2026-09-30): SFT (u0) vs final (GRPO) = the final update of final.json [vs base = the start policy, --include-base].
Check (exit 1 on any failure): the ids are exactly splits[fold].test (Task 2) and test_all (Task 1), in no training /
validation list; every (conversation, seed) of every evaluated policy is present, with the checkpoint's policy sha; every
Planner generation (Task 2 steps, Task 1 turns, Task 1 probes) was served by that checkpoint's vLLM adapter; the
unscored-point rule and sum(n - 1) decision points hold; every summary number (all seeds and the seeds 0/1 subset: turn
stats; term_f1, bal_p, auc, nll) is recomputed from the rows; the meta matches the summaries and final.json (the tested
updates are exactly {0, final}).
Bootstrap: 10000 resamples of the test conversations (seed 0), paired on the (conversation, seed) pairs clean in BOTH:
Task 2 (turn W1, mean |sim - human| turns, coverage, sim turns) over test, with all seeds and with seeds 0/1; Task 1
(term_f1, bal_p, auc) over test_all. Pairs: final vs SFT (u0); with a base evaluation also SFT (u0) vs base and final vs
base. Report wording (§7): base, SFT (u0), final (GRPO).
Usage: python eval_test_boot.py RUN_DIR"""
import collections
import json
import os
import random
import sys

import task1_stop as T1
import train_planner_rl as TP
import verify_pipeline as VP

run = sys.argv[1]
meta = TP.read_jsonl(os.path.join(run, "test_meta.jsonl"))[-1]
first = TP.read_jsonl(os.path.join(run, "run_meta.jsonl"))[0]
fails = []


def ok(cond, msg):
    if not cond:
        fails.append(msg)
        print("FAIL", msg)
    return cond


# ---- ids against the splits file of the run
args = first["config"]["args"]
ok(TP.sha_file(args["splits"]) == first["splits_sha256"] == meta["splits_sha256"], "splits file sha differs from the run's")
fold = {int(f["fold"]): f for f in json.load(open(args["splits"], encoding="utf-8"))["folds"]}[int(args["fold"])]
test, test_all = sorted(fold["test"]), sorted(fold["test_all"])
ok(meta["task2_ids"] == test and meta["task1_ids"] == test_all, "meta ids differ from splits test / test_all")
ok(set(test) <= set(test_all) <= set(fold["forbidden_for_training"]), "test ids not inside forbidden_for_training")
for k in ("train", "train_all", "validation", "validation_all"):
    ok(not set(test_all) & set(fold.get(k) or []), "test_all intersects %s" % k)
HERE = os.path.dirname(os.path.abspath(__file__))
for f in ("eval_test_rl.py", "eval_test_boot.py"):
    ok((meta.get("eval_code_sha256") or {}).get(f) == TP.sha_file(os.path.join(HERE, f)), "%s differs from the one that ran" % f)
fp = os.path.join(run, "final.json")
ok(os.path.exists(fp) and meta.get("final_json_sha256") == TP.sha_file(fp), "final.json missing or changed after the test")
fu = json.load(open(fp))["final_update"] if os.path.exists(fp) else None
ok(sorted(meta["updates"]) == sorted({0, fu}), "tested updates %s are not u0 + the final u%s" % (meta["updates"], fu))
T_MAX = int(first["config"]["selection_cfg"]["t_max"])
rows = [r for r in TP.read_jsonl(os.path.join(run, "test.jsonl"))]
with_base = bool(meta.get("include_base"))
# fix round 3 (A N2): a v17 test evaluation always includes the base re-scored under the v17 definitions (every fold)
if (first.get("config") or {}).get("spec_version") == "v17":
    ok(with_base and os.path.exists(os.path.join(run, "test_base.jsonl")),
       "a v17 test run without the base (eval_test_rl.py --include-base -> test_base.jsonl)")
if with_base:
    rows += TP.read_jsonl(os.path.join(run, "test_base.jsonl"))
summ = {r["update"]: r for r in rows if r.get("kind") == "summary"}
want_keys = sorted(meta["updates"], key=str) + (["base"] if with_base else [])
ok(sorted(summ, key=str) == sorted(want_keys, key=str), "summaries %s != planned %s" % (sorted(summ, key=str), want_keys))
seeds = meta["seeds"]
if fails:                                        # nothing below is meaningful without the plan
    print("TEST CHECK FAILED (%d)" % len(fails))
    sys.exit(1)

LABEL = {0: "SFT (u0)", fu: "final (GRPO) u%s" % fu, "base": "base"}


def ckpt_of(u):
    return os.path.join(run, "ckpt", "sft_e0" if u == "base" else "u%05d" % u)


eps, t1 = collections.defaultdict(dict), collections.defaultdict(dict)
for u in want_keys:
    psha = json.load(open(os.path.join(ckpt_of(u), "state.json")))["policy_sha"]
    # the served-adapter checks apply to a real vLLM run (a dry run's stub adapter directory is never served)
    real_vllm = args.get("planner_backend") == "vllm" and not args.get("dry_run")
    want_ad = VP.adapter_name(run, "sft_e0" if u == "base" else u) if real_vllm else None
    s = summ[u]
    ok(s["policy_sha"] == psha, "%s summary policy sha != checkpoint" % LABEL[u])
    ok(s["seeds"] == seeds and s["task2_ids"] == test and s["task1_ids"] == test_all, "%s summary seeds / ids != meta" % LABEL[u])
    for r in rows:
        if r.get("update") != u or r.get("kind") not in ("episode", "task1"):
            continue
        ok(r["split"] == "test", "%s row with split %r" % (LABEL[u], r["split"]))
        ok(r["policy_sha"] == psha, "%s row of another policy sha (%s)" % (LABEL[u], str(r["policy_sha"])[:12]))
        if r["kind"] == "episode":
            ok(r["conversation_id"] in test and r["seed"] in seeds, "%s episode id/seed outside the test plan" % LABEL[u])
            eps[u][(r["conversation_id"], r["seed"])] = r["episode"]       # latest attempt
            fits = [st.get("planner_fit") for st in r["episode"].get("trace") or [] if st.get("planner_fit")]
            ok(all(f.get("gen_adapter") == want_ad for f in fits) and (fits or want_ad is None),
               "%s %s s%s: Task 2 Planner steps not served by adapter %s" % (LABEL[u], r["conversation_id"], r["seed"], want_ad))
        else:
            ok(r["conversation_id"] in test_all, "%s Task 1 id outside test_all" % LABEL[u])
            t1[u][r["conversation_id"]] = r
            fits = [x.get("planner_fit") for x in r["task1"]["turns"] if x.get("planner_fit")]
            ok(all(f.get("gen_adapter") == want_ad for f in fits) and (fits or want_ad is None),
               "%s %s: Task 1 turns not served by adapter %s" % (LABEL[u], r["conversation_id"], want_ad))
    ok(sorted(eps[u]) == sorted((c, s_) for c in test for s_ in seeds), "%s: Task 2 episodes incomplete" % LABEL[u])
    ok(sorted(t1[u]) == test_all, "%s: Task 1 conversations != test_all" % LABEL[u])
    n_unclean = sum(1 for e in eps[u].values() if not e["clean"])
    ok(s["n_unclean_episodes"] == n_unclean and s["n_episodes"] == len(eps[u]) - n_unclean, "%s episode counts" % LABEL[u])
    pts = [p for r in t1[u].values() for p in r["end_probs"]]
    ok(all(p.get("gen_adapter") == want_ad for p in pts), "%s: probes not generated by adapter %s" % (LABEL[u], want_ad))
    ok(len(pts) == sum(len(r["task1"]["turns"]) - 1 for r in t1[u].values()), "%s: decision points != sum(n - 1)" % LABEL[u])
    for p in pts:
        if not p["valid"]:
            ok(float(p["p_end"]) == (1.0 if (p["decision_valid"] and p["greedy_end"]) else 0.0),
               "%s t%s: unscored point p_end %r" % (LABEL[u], p["t"], p["p_end"]))


def t2stats(u, keys):
    E = [eps[u][k] for k in keys if eps[u][k]["clean"]]
    sim = [e["emitted_user_turns"] for e in E]
    hum = [min(int(e["human_turns"]), T_MAX) for e in E]            # capped for W1, as validate()
    # human turns capped at t_max everywhere (fix round 1, D-N5; TP.turn_stats_of); the uncapped mean is "hum_uncapped"
    return {"w1": TP.turn_w1(sim, hum), "abs_err": sum(abs(a_ - b_) for a_, b_ in zip(sim, hum)) / len(E),
            "cov": sum(float(e["coverage"]) for e in E) / len(E), "sim": sum(sim) / len(sim),
            "hum": sum(hum) / len(hum), "hum_uncapped": sum(e["human_turns"] for e in E) / len(E)}


def t1stats(u, convs):
    rs = [t1[u][c] for c in convs]
    m = T1.task1_stop_metrics([r["task1"] for r in rs])
    m.update(T1.task1_prob_metrics([p for r in rs for p in r["end_probs"]]))
    return m


def close(a_, b_):
    return a_ == b_ or (a_ is not None and b_ is not None and abs(a_ - b_) < 1e-9)


# ---- every summary number recomputed from the rows
for u in want_keys:
    s, full1 = summ[u], t1stats(u, test_all)
    for tag, sd in (("turn_stats", seeds), ("turn_stats_seeds01", [0, 1])):
        ts = s[tag] or {}
        full2 = t2stats(u, sorted(k for k in eps[u] if k[1] in sd))
        for k_s, k_r in (("turn_w1", "w1"), ("abs_diff_mean", "abs_err"), ("coverage_mean", "cov"),
                         ("sim_turns_mean", "sim"), ("human_turns_mean", "hum"),
                         ("human_turns_mean_uncapped", "hum_uncapped")):
            ok(ts.get(k_s) is not None and abs(ts[k_s] - full2[k_r]) < 1e-9,
               "%s %s.%s %r != recomputed %r" % (LABEL[u], tag, k_s, ts.get(k_s), full2[k_r]))
        print("%-18s Task 2 %-18s %2d episodes: sim %.2f human %.2f W1 %.3f |diff| %.2f cov %.3f" % (
            LABEL[u], tag, ts.get("n_episodes") or 0, full2["sim"], full2["hum"], full2["w1"], full2["abs_err"], full2["cov"]))
    for k in ("term_f1", "bal_p", "auc", "premature_end_rate", "nll"):
        ok(close(s["task1"].get(k), full1.get(k)), "%s %s %r != recomputed %r" % (LABEL[u], k, s["task1"].get(k), full1.get(k)))
    print("%-18s Task 1 %d convs: term_f1 %.3f bal_p %.3f auc %s nll %s premature %.3f (unclean Task 2 episodes: %d)" % (
        LABEL[u], len(test_all), full1["term_f1"], full1["bal_p"], None if full1["auc"] is None else round(full1["auc"], 3),
        None if full1["nll"] is None else round(full1["nll"], 4), full1["premature"], s["n_unclean_episodes"]))
if fails:
    print("TEST CHECK FAILED (%d)" % len(fails))
    sys.exit(1)
print("TEST CHECK PASSED")

# ---- paired bootstrap
LB = {"w1": True, "abs_err": True, "cov": False, "sim": None, "term_f1": False, "bal_p": False, "auc": False}
pairs = [(0, fu)] + ([("base", 0), ("base", fu)] if with_base else [])
for a, b in pairs:
    rng = random.Random(0)
    both = sorted(k for k in set(eps[a]) & set(eps[b]) if eps[a][k]["clean"] and eps[b][k]["clean"])
    dropped = len(set(eps[a]) | set(eps[b])) - len(both)
    if dropped:
        print("WARNING: %d (conversation, seed) pairs unclean in %s or %s after the re-runs are left out of the paired "
              "Task 2 comparison (the per-policy summaries above include every clean episode)" % (dropped, LABEL[a], LABEL[b]))

    def boot2(keys):
        by_c = {c: [k for k in keys if k[0] == c] for c in test}
        convs = [c for c in test if by_c[c]]
        d = collections.defaultdict(list)
        for _ in range(10000):
            ks = [k for c in (rng.choice(convs) for _ in convs) for k in by_c[c]]
            x, y = t2stats(a, ks), t2stats(b, ks)
            for m in ("w1", "abs_err", "cov", "sim"):
                d[m].append(y[m] - x[m])
        return d, {u: t2stats(u, keys) for u in (a, b)}, len(convs)

    def show(m, d, fa, fb, lower_better):
        d = sorted(d)
        lo, hi = d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]
        extra = "" if lower_better is None else "  P(%s better) %.3f  P(worse) %.3f" % (
            LABEL[b], sum(1 for x in d if (x < 0 if lower_better else x > 0)) / len(d),
            sum(1 for x in d if (x > 0 if lower_better else x < 0)) / len(d))
        print("   %-8s %s %s -> %s %s  diff %s  95%% CI [%.3f, %.3f]%s" % (
            m, LABEL[a], "None" if fa is None else "%.3f" % fa, LABEL[b], "None" if fb is None else "%.3f" % fb,
            "None" if None in (fa, fb) else "%.3f" % (fb - fa), lo, hi, extra))

    for tag, keys in (("all seeds %s" % seeds, both), ("seeds 0/1 subset", [k for k in both if k[1] in (0, 1)])):
        d2, full2, nc = boot2(keys)
        print("%s vs %s, Task 2 %s: %d paired episodes on %d conversations, 10000 resamples of conversations "
              "(few clusters: coarse intervals)" % (LABEL[b], LABEL[a], tag, len(keys), nc))
        for m in ("w1", "abs_err", "cov", "sim"):
            show(m, d2[m], full2[a][m], full2[b][m], LB[m])
    d1 = collections.defaultdict(list)
    for _ in range(10000):
        cs = [rng.choice(test_all) for _ in test_all]
        x, y = t1stats(a, cs), t1stats(b, cs)
        for m in ("term_f1", "bal_p", "auc"):
            if x[m] is not None and y[m] is not None:
                d1[m].append(y[m] - x[m])
    full1 = {u: t1stats(u, test_all) for u in (a, b)}
    print("%s vs %s, Task 1: %d conversations, 10000 resamples" % (LABEL[b], LABEL[a], len(test_all)))
    for m in ("term_f1", "bal_p", "auc"):
        if d1[m]:
            show(m, d1[m], full1[a][m], full1[b][m], LB[m])
