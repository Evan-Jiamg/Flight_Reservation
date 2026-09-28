"""v16 checks in verify_pipeline.py (SPEC ops/SPEC_v16_grpo_opt.md)."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''        want = (v["w_sel_cov"] * ts["coverage_mean"] - v["w_sel_w1"] * ts["turn_w1"]
                + v["w_sel_task1"] * (t1["term_f1"] if t1 else 0.0))''',
    '''        m1 = v.get("selection_task1_metric", "term_f1")        # v16 runs: bal_p; older runs: term_f1
        want = (v["w_sel_cov"] * ts["coverage_mean"] - v["w_sel_w1"] * ts["turn_w1"]
                + v["w_sel_task1"] * (t1[m1] if t1 else 0.0))''')
rep('''def check_rl(rl_dir, rollouts, ckpt_pattern, rep):
    check_intervention(rl_dir, rep)
''', '''def _jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


def check_v16(rl_dir, rep):
    """SPEC v16 (user 2026-09-28): Task 1 refill / group size, Dr. GRPO setting, aux floor, w_dist floor, the
    continuous Task 1 metric (recomputed from the logged end probabilities) and the two-point D2 trigger."""
    import task1_stop as T1
    meta = _jl(os.path.join(rl_dir, "run_meta.jsonl"))
    if not meta:
        return
    args = (meta[0].get("config") or {}).get("args") or {}
    if "task1_G" not in args:
        rep.note("rl.v16", "run written before v16 (no task1_G): v16 checks skipped")
        return
    acfg = (meta[0].get("config") or {}).get("algo_cfg") or {}
    rep.ok("rl.adv_norm", acfg.get("grpo_std_norm") is False or bool(args.get("ablation")), "run_meta",
           "algo_cfg.grpo_std_norm %r (spec v16: false)" % acfg.get("grpo_std_norm"))
    splits = None
    try:
        sp = json.load(open(args["splits"], encoding="utf-8"))
        splits = {int(f["fold"]): f for f in sp["folds"]}.get(int(args["fold"]))
    except Exception:
        pass
    train_all = set((splits or {}).get("train_all", (splits or {}).get("train", []))) if splits else None
    # Task 1 groups: size, base / refill conversations
    t1 = _jl(os.path.join(rl_dir, "rollouts_task1.jsonl"))
    by_u = {}
    for r in t1:
        w = "task1 u%s %s t%s" % (r.get("update"), str(r.get("conversation_id"))[:10], r.get("t"))
        rep.ok("rl.task1_G", len(r.get("samples") or []) == int(args["task1_G"]), w,
               "%d samples, task1_G %s" % (len(r.get("samples") or []), args["task1_G"]))
        by_u.setdefault(r["update"], []).append(r)
    upd = {u["update"]: u for u in _jl(os.path.join(rl_dir, "updates.jsonl"))}
    for u, rows in sorted(by_u.items()):
        base = {r["conversation_id"] for r in rows if not r.get("refill")}
        refill = {r["conversation_id"] for r in rows if r.get("refill")}
        w = "task1 u%s" % u
        if train_all is not None:
            rep.ok("rl.task1_refill", (base | refill) <= train_all, w, "Task 1 conversation outside train_all")
            rep.ok("rl.task1_G", len(base) == min(int(args["task1_convs"]), len(train_all)), w,
                   "%d base conversations, task1_convs %s" % (len(base), args["task1_convs"]))
        rep.ok("rl.task1_refill", not (base & refill), w, "a refill conversation is also a base conversation")
        rep.ok("rl.task1_refill", len(refill) <= int(args["task1_convs"]), w, "%d refill conversations > cap" % len(refill))
        n_base_groups = sum(1 for r in rows if not r.get("refill"))
        st = (upd.get(u) or {}).get("learner_stats") or {}
        if st.get("aux_n") is not None:
            rep.ok("rl.task1_refill", st["aux_n"] <= n_base_groups, w,
                   "aux examples %s > base groups %d (aux must come from base groups only)" % (st["aux_n"], n_base_groups))
        th = ((upd.get(u) or {}).get("train_aggregate") or {}).get("task1_train") or {}
        if th:
            rep.ok("rl.task1_refill", th.get("n_base_groups") == n_base_groups and th.get("n_refill_groups") ==
                   len(rows) - n_base_groups, w, "task1_train refill counts %r / %r do not match the logged rows %d / %d"
                   % (th.get("n_base_groups"), th.get("n_refill_groups"), n_base_groups, len(rows) - n_base_groups))
    # aux floor, w_dist floor, D2 trigger
    ctl = ((meta[0].get("config") or {}).get("controller") or {}).get("llm4") or {}
    wlo = ((ctl.get("bounds") or {}).get("w_dist") or [None])[0]
    vsum = sorted((v for v in _jl(os.path.join(rl_dir, "validation.jsonl")) if v.get("kind") == "summary"),
                  key=lambda v: v["update"])
    trig = [v for v in vsum if (v.get("d2") or {}).get("triggered_at") == v["update"]]
    rep.ok("rl.d2_trigger", len(trig) <= 1, "validation", "D2 triggered more than once: %r" % [v["update"] for v in trig])
    for v in trig:
        rep.ok("rl.d2_trigger", v["d2"]["streak"] >= 2 and v["d2"]["met"], "validation u%s" % v["update"],
               "D2 triggered without two consecutive validations over the margin: %r" % v["d2"])
    t_at = trig[0]["update"] if trig else None
    for u, row in sorted(upd.items()):
        ag, cfg = row.get("train_aggregate") or {}, row.get("cfg_used") or {}
        w = "update %s" % u
        if ag.get("aux_floor") is not None and ag.get("aux_weight") is not None:
            rep.ok("rl.aux_floor", ag["aux_weight"] >= ag["aux_floor"] - 1e-12, w,
                   "aux weight %r below the floor %r" % (ag["aux_weight"], ag["aux_floor"]))
            if ag["aux_weight"] < float(cfg.get("w_aux", 0.0)) - 1e-12:
                rep.ok("rl.d2_trigger", t_at is not None and u > t_at, w,
                       "stop supervision annealed (%r < w_aux %r) before any D2 trigger" % (ag["aux_weight"], cfg.get("w_aux")))
        if wlo is not None and "w_dist" in cfg:
            rep.ok("rl.w_dist_floor", float(cfg["w_dist"]) >= float(wlo) - 1e-12, w,
                   "w_dist %r below the controller bound %r" % (cfg["w_dist"], wlo))
    # continuous Task 1 metric recomputed from the logged end probabilities
    vrows = _jl(os.path.join(rl_dir, "validation.jsonl"))
    for v in vsum:
        w = "validation u%s" % v["update"]
        rows = {}
        for r in vrows:
            if r.get("kind") == "task1" and r["update"] == v["update"] and r.get("policy_sha") == v.get("policy_sha"):
                rows[r["conversation_id"]] = r                      # the latest row per conversation
        if not rows or not v.get("task1"):
            continue
        pts = [p_ for r in rows.values() for p_ in (r.get("end_probs") or [])]
        ok_rng = all(0.0 <= float(p_["p_end"]) <= 1.0 for p_ in pts)
        rep.ok("rl.task1_prob", ok_rng, w, "an end probability outside [0, 1]")
        n_exp = sum(len(r["task1"]["turns"]) - 1 for r in rows.values())
        rep.ok("rl.task1_prob", len(pts) == n_exp, w, "%d decision points, expected sum(n - 1) = %d" % (len(pts), n_exp))
        if pts and ok_rng:
            m = T1.task1_prob_metrics(pts)
            rep.ok("rl.task1_prob", abs(m["bal_p"] - v["task1"].get("bal_p", float("nan"))) < 1e-9, w,
                   "summary bal_p %r != recomputed %r" % (v["task1"].get("bal_p"), m["bal_p"]))


def check_rl(rl_dir, rollouts, ckpt_pattern, rep):
    check_intervention(rl_dir, rep)
    check_v16(rl_dir, rep)
''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
