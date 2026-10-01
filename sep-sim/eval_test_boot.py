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




def boot_v18(run, meta, first):
    """SPEC v18 §8: check the v18 test rows (the final policy only, clean_v18, the coverage diagnostic over the non-null
    episodes) and compare with v17's existing test rows -- v17 SFT (u0) and v17 GRPO (u5), read from the file test_meta
    names (its sha must not have changed). Paired bootstrap by conversation (Task 2 paired on (conversation, seed) pairs
    clean in both: v18 clean_v18, v17 its own clean), 10000 resamples, seed 0. Report wording: v17 SFT (u0), v17 GRPO
    (u5), v18 (GRPO+R)."""
    import v18_rules as V18
    fails_ = []

    def ok_(c, m):
        if not c:
            fails_.append(m)
            print("FAIL", m)
        return c
    args_ = first["config"]["args"]
    ok_(TP.sha_file(args_["splits"]) == first["splits_sha256"] == meta["splits_sha256"], "splits sha")
    fold_ = {int(f["fold"]): f for f in json.load(open(args_["splits"], encoding="utf-8"))["folds"]}[int(args_["fold"])]
    test_, test_all_ = sorted(fold_["test"]), sorted(fold_["test_all"])
    ok_(meta["task2_ids"] == test_ and meta["task1_ids"] == test_all_, "meta ids differ from splits test / test_all")
    for k in ("train", "train_all", "validation", "validation_all"):
        ok_(not set(test_all_) & set(fold_.get(k) or []), "test_all intersects %s" % k)
    here = os.path.dirname(os.path.abspath(__file__))
    for f_ in ("eval_test_rl.py", "eval_test_boot.py"):
        ok_((meta.get("eval_code_sha256") or {}).get(f_) == TP.sha_file(os.path.join(here, f_)), "%s changed" % f_)
    fp_ = os.path.join(run, "final.json")
    fin_ = json.load(open(fp_)) if os.path.exists(fp_) else {}
    fu_ = fin_.get("final_update")
    ok_(meta.get("final_json_sha256") == TP.sha_file(fp_), "final.json changed after the test")
    ok_(meta["updates"] == [fu_] and not meta.get("include_base"), "tested updates %r (v18: [final])" % meta["updates"])
    ref = meta.get("v17_reference") or {}
    ok_(os.path.exists(ref.get("test_jsonl", "")) and TP.sha_file(ref["test_jsonl"]) == ref.get("test_jsonl_sha256"),
        "the v17 comparison file changed / missing")
    if fails_:
        print("TEST CHECK FAILED (%d)" % len(fails_))
        sys.exit(1)
    t_max = int(first["config"]["selection_cfg"]["t_max"])
    seeds_ = meta["seeds"]
    rows18 = [r for r in TP.read_jsonl(os.path.join(run, "test.jsonl"))]
    rows17 = [r for r in TP.read_jsonl(ref["test_jsonl"])]
    g17 = int(ref.get("grpo_update", 5))
    grpo_name = "v17 GRPO (u%d)" % g17
    pols = {"v18 (GRPO+R)": (rows18, fu_, "clean_v18"), "v17 SFT (u0)": (rows17, int(ref.get("sft_update", 0)), "clean"),
            grpo_name: (rows17, g17, "clean")}
    import verify_v18 as VV
    real_vllm = args_.get("planner_backend") == "vllm" and not args_.get("dry_run")
    # audit B (SHOULD 2): v17's checks for the v18 rows -- policy sha from ckpt u's state.json, served adapters,
    # the unscored-point rule, sum(n - 1) probe points
    for msg in VV.check_test_rows(run, rows18, fu_, set(test_), set(test_all_), list(seeds_), real_vllm):
        ok_(False, "v18 (GRPO+R): " + msg)
    ok_({r.get("update") for r in rows18 if r.get("kind")} <= {fu_}, "v18 test rows of another update than the final")
    eps_, t1_ = {}, {}
    for name, (rows, u, ck) in pols.items():
        summ = [r for r in rows if r.get("kind") == "summary" and r.get("update") == u]
        ok_(len(summ) >= 1, "%s: no test summary" % name)
        if name.startswith("v18"):
            psha = json.load(open(os.path.join(run, "ckpt", "u%05d" % fu_, "state.json")))["policy_sha"]
        else:
            psha = summ[-1]["policy_sha"] if summ else None
        e, t = {}, {}
        for r in rows:
            if r.get("update") != u or r.get("policy_sha") != psha:
                continue
            if r.get("kind") == "episode":
                e[(r["conversation_id"], r["seed"])] = r["episode"]
            elif r.get("kind") == "task1":
                t[r["conversation_id"]] = r
        ok_(sorted(e) == sorted((c, s_) for c in test_ for s_ in seeds_), "%s: Task 2 episodes incomplete" % name)
        ok_(sorted(t) == test_all_, "%s: Task 1 conversations != test_all" % name)
        eps_[name], t1_[name] = (e, ck), t
        if name.startswith("v18") and summ:
            sm = summ[-1]
            E = [x for x in e.values() if x[ck]]
            ts = TP.turn_stats_v18(E, t_max) if E else {}
            for k in ("turn_w1", "abs_diff_mean", "sim_turns_mean", "coverage_mean", "coverage_missing"):
                ok_(close((sm.get("turn_stats") or {}).get(k), ts.get(k)), "v18 summary %s %r != recomputed %r"
                    % (k, (sm.get("turn_stats") or {}).get(k), ts.get(k)))
            ok_(sm.get("n_unclean_episodes") == len(e) - len(E), "v18 unclean count")
            rs = [t[c] for c in test_all_]
            m = T1.task1_stop_metrics([r["task1"] for r in rs])
            m.update(T1.task1_prob_metrics([p for r in rs for p in r["end_probs"]]))
            for k in ("term_f1", "bal_p", "auc", "nll"):
                ok_(close(sm["task1"].get(k), m.get(k)), "v18 summary %s" % k)
    if fails_:
        print("TEST CHECK FAILED (%d)" % len(fails_))
        sys.exit(1)
    print("TEST CHECK PASSED")

    def t2(name, keys):
        e, ck = eps_[name]
        E = [e[k] for k in keys]
        sim = [x["emitted_user_turns"] for x in E]
        hum = [min(int(x["human_turns"]), t_max) for x in E]
        covs = [(x.get("coverage_diag") if ck == "clean_v18" else x.get("coverage")) for x in E]
        have = [float(c) for c in covs if c is not None]
        return {"w1": TP.turn_w1(sim, hum), "abs_err": sum(abs(a_ - b_) for a_, b_ in zip(sim, hum)) / len(E),
                "cov": (sum(have) / len(have)) if have else None, "sim": sum(sim) / len(sim), "hum": sum(hum) / len(hum)}

    def t1s(name, convs):
        rs = [t1_[name][c] for c in convs]
        m = T1.task1_stop_metrics([r["task1"] for r in rs])
        m.update(T1.task1_prob_metrics([p for r in rs for p in r["end_probs"]]))
        return m
    lb = {"w1": True, "abs_err": True, "cov": False, "sim": None, "term_f1": False, "bal_p": False, "auc": False}
    for name in pols:
        e, ck = eps_[name]
        keys = sorted(k for k, x in e.items() if x[ck])
        x2, x1 = t2(name, keys), t1s(name, test_all_)
        print("%-15s Task 2 %d clean episodes: sim %.2f human %.2f W1 %.3f |diff| %.2f cov(diag) %s | Task 1: term_f1 "
              "%.3f bal_p %.3f auc %s nll %s" % (name, len(keys), x2["sim"], x2["hum"], x2["w1"], x2["abs_err"],
                                                 None if x2["cov"] is None else round(x2["cov"], 3), x1["term_f1"],
                                                 x1["bal_p"], None if x1["auc"] is None else round(x1["auc"], 3),
                                                 None if x1["nll"] is None else round(x1["nll"], 4)))
    print("NOTE: v18 vs v17 is the combined effect of the reward, the advantage and the reranker (SPEC v18 §0, §8); "
          "9 test conversations: directions only, no significance claims.")
    for base in ("v17 SFT (u0)", grpo_name):
        a_n, b_n = base, "v18 (GRPO+R)"
        ea, cka = eps_[a_n]
        eb, ckb = eps_[b_n]
        both = sorted(k for k in set(ea) & set(eb) if ea[k][cka] and eb[k][ckb])
        rng = random.Random(0)
        by_c = {c: [k for k in both if k[0] == c] for c in test_}
        convs = [c for c in test_ if by_c[c]]
        d = collections.defaultdict(list)
        for _ in range(10000):
            ks = [k for c in (rng.choice(convs) for _ in convs) for k in by_c[c]]
            x, y = t2(a_n, ks), t2(b_n, ks)
            for m in ("w1", "abs_err", "cov", "sim"):
                if x[m] is not None and y[m] is not None:
                    d[m].append(y[m] - x[m])
        full = {n: t2(n, both) for n in (a_n, b_n)}
        print("%s vs %s, Task 2: %d paired episodes on %d conversations (clean in both)" % (b_n, a_n, len(both), len(convs)))
        for m in ("w1", "abs_err", "cov", "sim"):
            show_(m, d[m], full[a_n][m], full[b_n][m], lb[m], a_n, b_n)
        d1 = collections.defaultdict(list)
        for _ in range(10000):
            cs = [rng.choice(test_all_) for _ in test_all_]
            x, y = t1s(a_n, cs), t1s(b_n, cs)
            for m in ("term_f1", "bal_p", "auc"):
                if x[m] is not None and y[m] is not None:
                    d1[m].append(y[m] - x[m])
        f1 = {n: t1s(n, test_all_) for n in (a_n, b_n)}
        print("%s vs %s, Task 1: %d conversations" % (b_n, a_n, len(test_all_)))
        for m in ("term_f1", "bal_p", "auc"):
            show_(m, d1[m], f1[a_n][m], f1[b_n][m], lb[m], a_n, b_n)
    sys.exit(0)


def close(a_, b_):
    return a_ == b_ or (a_ is not None and b_ is not None and abs(a_ - b_) < 1e-9)


def show_(m, d, fa, fb, lower_better, la, lb_):
    if not d:
        print("   %-8s no paired value" % m)
        return
    d = sorted(d)
    lo, hi = d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]
    extra = "" if lower_better is None else "  P(%s better) %.3f" % (
        lb_, sum(1 for x in d if (x < 0 if lower_better else x > 0)) / len(d))
    print("   %-8s %s %s -> %s %s  diff %s  95%% CI [%.3f, %.3f]%s" % (
        m, la, "None" if fa is None else "%.3f" % fa, lb_, "None" if fb is None else "%.3f" % fb,
        "None" if None in (fa, fb) else "%.3f" % (fb - fa), lo, hi, extra))


if (first.get("config") or {}).get("spec_version") == "v18":
    boot_v18(run, meta, first)

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
