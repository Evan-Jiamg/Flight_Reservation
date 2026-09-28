import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:70])
    s = s.replace(old, new)


rep('''    rep.ok("rl.best", os.path.exists(os.path.join(rl_dir, "best.json")), rl_dir, "best.json missing")
''', '''    rep.ok("rl.best", os.path.exists(os.path.join(rl_dir, "best.json")), rl_dir, "best.json missing")
    check_reselect_ids(rl_dir, val, train_all, rep)
''')
rep('''def check_rl_selection(rl_dir, splits, fold, rep):''', '''def check_reselect_ids(rl_dir, val, train_all, rep):
    """Checkpoint re-selection (reselect.jsonl): only validation ids, only checkpoints that were validated in
    training, the seeds of reselect_meta, and reselect_best.json names the best non-withheld summary."""
    rp = os.path.join(rl_dir, "reselect.jsonl")
    if not os.path.exists(rp):
        return
    rows = [json.loads(l) for l in open(rp, encoding="utf-8") if l.strip()]
    vp = os.path.join(rl_dir, "validation.jsonl")
    validated = {json.loads(l)["update"] for l in open(vp, encoding="utf-8")
                 if l.strip() and json.loads(l).get("kind") == "summary"} if os.path.exists(vp) else set()
    mp = os.path.join(rl_dir, "reselect_meta.jsonl")
    metas = [json.loads(l) for l in open(mp, encoding="utf-8") if l.strip()] if os.path.exists(mp) else []
    rep.ok("rl.reselect_meta", bool(metas), mp, "reselect.jsonl without reselect_meta.jsonl")
    seeds = set(metas[-1]["seeds"]) if metas else set()
    for r in rows:
        w = "reselect u%s %s" % (r.get("update"), str(r.get("conversation_id"))[:12])
        rep.ok("rl.reselect_candidate", r.get("update") in validated, w, "re-selected a checkpoint never validated")
        if r.get("kind") in ("episode", "task1"):
            cid = r.get("conversation_id")
            rep.ok("leak.reselect_ids", cid in val and cid not in train_all, w, "re-selection row on a non-validation id")
        if r.get("kind") == "episode":
            rep.ok("rl.reselect_seeds", r.get("seed") in seeds, w, "seed %r not in %r" % (r.get("seed"), sorted(seeds)))
        if r.get("kind") == "summary":
            rep.ok("rl.reselect_seeds", set(r.get("val_seeds") or []) == seeds and r.get("reselect") is True, w,
                   "summary seeds %r / reselect flag %r" % (r.get("val_seeds"), r.get("reselect")))
    summ = {r["update"]: r for r in rows if r.get("kind") == "summary"}
    bp = os.path.join(rl_dir, "reselect_best.json")
    if summ and rep.ok("rl.reselect_best", os.path.exists(bp), bp, "reselect_best.json missing"):
        b = json.load(open(bp, encoding="utf-8"))
        ok = {u: v["selection_score"] for u, v in summ.items() if v.get("selection_score") is not None}
        exp = max(ok, key=lambda u: (ok[u], -u)) if ok else None
        rep.ok("rl.reselect_best", (b.get("best") or {}).get("update") == exp, bp,
               "reselect_best names u%r, best summary is u%r" % ((b.get("best") or {}).get("update"), exp))
        rep.note("rl.reselect_best", "re-selection scores %s -> u%s" % ({u: round(v, 4) for u, v in sorted(ok.items())}, exp))


def check_rl_selection(rl_dir, splits, fold, rep):''')
# structure / truncation / pend / few-shot checks on the re-selection episodes, like validation.jsonl
rep('''            check_selection(vp, rl_dir, rep)
''', '''            check_selection(vp, rl_dir, rep)
        rp = os.path.join(rl_dir, "reselect.jsonl")
        if os.path.exists(rp):
            rrows = load_jsonl(rp, rep)
            if rrows:
                check_structure(rrows, arm, rep)
                check_truncation(rrows, arm, meta, sb, judge_budget, max_new_warn, rep)
                {"a2": check_a2, "pend": check_pend, "a0": check_a0}[check_family(arm)](rrows, rep, arm)
                if arm == "pend":
                    check_fewshot_leak(rrows, splits, fold, rep)
''')
rep('''        vr = [r for r in (load_jsonl(vp, rep) if vp and os.path.exists(vp) else [])]
        check_r0_attribution(all_rows + vr, rep)''', '''        vr = [r for r in (load_jsonl(vp, rep) if vp and os.path.exists(vp) else [])]
        rp = os.path.join(rl_dir, "reselect.jsonl") if rl_dir else None
        rr = [r for r in (load_jsonl(rp, rep) if rp and os.path.exists(rp) else [])]
        check_r0_attribution(all_rows + vr + rr, rep)''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
