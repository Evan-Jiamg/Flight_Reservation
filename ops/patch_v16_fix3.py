"""Re-audit N1-N5 (df69282): splits path from the CLI with a sha check, Task 1 record always written, initial aux
weight below the floor refused, decision_valid logged + checked, D2 streak reset on a summary without Task 1."""
import sys

d = sys.argv[1]


def patch(fn, pairs):
    p = d + "/" + fn
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (fn, s.count(old), old[:80])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


patch("train_planner_rl.py", [
    # N2: the Task 1 record is written whenever Task 1 groups are configured, even without a single row
    ('''        t1_hist = None
        if t1_all:
            fin = [x for x in t1_base if x["real_final"]]
            mid = [x for x in t1_base if not x["real_final"]]
            t1_hist = {"n": len(t1_base), "acc": (sum(x["reward"] for x in t1_base) / len(t1_base)) if t1_base else None,
                       "acc_all": sum(x["reward"] for x in t1_all) / len(t1_all), "n_all": len(t1_all),''',
     '''        t1_hist = None
        if t1_all or self.a.task1_convs > 0:
            fin = [x for x in t1_base if x["real_final"]]
            mid = [x for x in t1_base if not x["real_final"]]
            t1_hist = {"n": len(t1_base), "acc": (sum(x["reward"] for x in t1_base) / len(t1_base)) if t1_base else None,
                       "acc_all": (sum(x["reward"] for x in t1_all) / len(t1_all)) if t1_all else None,
                       "n_all": len(t1_all),'''),
    # N4: the decision validity of every end probability point is logged
    ('''            probs.append({"t": x["t"], "real_final": bool(x["real_final"]), "p_end": pe, "valid": bool(pr["valid"]),
                          "greedy_end": bool(pr["greedy_end"])})''',
     '''            probs.append({"t": x["t"], "real_final": bool(x["real_final"]), "p_end": pe, "valid": bool(pr["valid"]),
                          "decision_valid": bool(pr.get("decision_valid", pr["valid"])),
                          "greedy_end": bool(pr["greedy_end"])})'''),
    # N3: an initial weight below the floor would be overridden silently
    ('''    if a.stop_sup_weight == 0 and a.stop_sup_floor != 0:''',
     '''    if 0 < a.stop_sup_weight < a.stop_sup_floor:
        ap.error("--stop-sup-weight %g is below --stop-sup-floor %g: the floor would override it silently"
                 % (a.stop_sup_weight, a.stop_sup_floor))
    if a.stop_sup_weight == 0 and a.stop_sup_floor != 0:'''),
])
patch("verify_pipeline.py", [
    ('''def _jl(path):''', '''def sha256_file(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _jl(path):'''),
    # N1: the split file given to the verifier (its sha must be the run's), the run's own path only as a fallback
    ('''def check_v16(rl_dir, rep):''', '''def check_v16(rl_dir, rep, splits_path=None):'''),
    ('''    train_all = None
    try:
        sp = json.load(open(args["splits"], encoding="utf-8"))
        f = {int(x["fold"]): x for x in sp["folds"]}[int(args["fold"])]
        train_all = set(f.get("train_all", f.get("train", [])))
    except Exception as e:                       # never skipped silently
        rep.ok("rl.task1_refill", False, "splits", "cannot read the run's split file %r: %r" % (args.get("splits"), e))''',
     '''    train_all = None
    sp_path = splits_path or args.get("splits")
    try:
        sp = json.load(open(sp_path, encoding="utf-8"))
        f = {int(x["fold"]): x for x in sp["folds"]}[int(args["fold"])]
        train_all = set(f.get("train_all", f.get("train", [])))
        want = meta[0].get("splits_sha256")
        if want:
            rep.ok("rl.task1_refill", sha256_file(sp_path) == want, "splits",
                   "split file %r is not the one the run used (sha differs)" % sp_path)
    except Exception as e:                       # never skipped silently
        rep.ok("rl.task1_refill", False, "splits", "cannot read the split file %r: %r" % (sp_path, e))'''),
    # N4: an end probability that could not be scored is 0 (invalid plan) or the greedy decision
    ('''        pts = [p_ for r in rows.values() for p_ in (r.get("end_probs") or [])]
        ok_rng = all(0.0 <= float(p_["p_end"]) <= 1.0 for p_ in pts)''',
     '''        pts = [p_ for r in rows.values() for p_ in (r.get("end_probs") or [])]
        for p_ in pts:
            if not p_.get("valid", True) and "decision_valid" in p_:
                want = 1.0 if (p_["decision_valid"] and p_.get("greedy_end")) else 0.0
                rep.ok("rl.task1_prob", float(p_["p_end"]) == want, w,
                       "t%s: unscored point p_end %r, expected %r" % (p_.get("t"), p_["p_end"], want))
        ok_rng = all(0.0 <= float(p_["p_end"]) <= 1.0 for p_ in pts)'''),
    # N5: a validation without Task 1 resets the streak (as the trainer does)
    ('''    for v in vsum:
        if base_v is None or v["update"] <= base_v["update"] or not v.get("task1") or v["task1"].get("bal_p") is None:
            continue
        met = v["task1"]["bal_p"] >= base_v["task1"]["bal_p"] + margin''',
     '''    for v in vsum:
        if base_v is None or v["update"] <= base_v["update"]:
            continue
        if not v.get("task1") or v["task1"].get("bal_p") is None:
            streak = 0                           # the trainer reads a validation without Task 1 as "not met"
            continue
        met = v["task1"]["bal_p"] >= base_v["task1"]["bal_p"] + margin'''),
    ('''def check_rl(rl_dir, rollouts, ckpt_pattern, rep):
    check_intervention(rl_dir, rep)
    check_v16(rl_dir, rep)''', '''def check_rl(rl_dir, rollouts, ckpt_pattern, rep, splits_path=None):
    check_intervention(rl_dir, rep)
    check_v16(rl_dir, rep, splits_path)'''),
    ('''        check_rl(rl_dir, rollouts, ckpt_pattern, rep)''', '''        check_rl(rl_dir, rollouts, ckpt_pattern, rep, splits_path)'''),
])
print("patched")
