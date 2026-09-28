"""Round-4 audit fixes (2026-09-26), batch 5: verify_pipeline / verify_task1. Run from sep-sim/."""


def patch1(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


patch1("verify_pipeline.py", [
    # (Task 2 #5) a Planner end followed by an all-blank Speaker = 'empty' (blank = END), not a failure
    ('''            if first is not None:
                rep.ok("e16.planner_end", first is tr[-1] and first.get("emitted") and
                       r.get("end_kind") in ("planner_end", "speaker_end"), where,
                       "Planner ended at t%s but episode went on / end_kind %r" % (first.get("t"), r.get("end_kind")))''',
     '''            if first is not None:
                ok_emit = first.get("emitted") and r.get("end_kind") in ("planner_end", "speaker_end")
                ok_blank = (not first.get("emitted")) and r.get("end_kind") == "empty"      # blank close = END
                rep.ok("e16.planner_end", first is tr[-1] and (ok_emit or ok_blank), where,
                       "Planner ended at t%s but episode went on / end_kind %r" % (first.get("t"), r.get("end_kind")))'''),
    # (Task 2 #10) pend: R0 / ledger judge are gpt-oss-120b on the local server, as recorded in the run meta;
    # (Task 2 #13) closes rejected as verbatim reuse are counted; (C26 / GRPO #2 / #10) validation checks
    ('''        vp = os.path.join(rl_dir, "validation.jsonl")
        if os.path.exists(vp):
            # the episodes that drive best.json get the same structure / truncation / pend checks
            vrows = load_jsonl(vp, rep)
            if vrows:
                check_structure(vrows, arm, rep)
                check_truncation(vrows, arm, meta, sb, judge_budget, max_new_warn, rep)
                {"a2": check_a2, "pend": check_pend, "a0": check_a0}[check_family(arm)](vrows, rep, arm)
    return rep''',
     '''        vp = os.path.join(rl_dir, "validation.jsonl")
        if os.path.exists(vp):
            # the episodes that drive best.json get the same structure / truncation / pend / few-shot checks
            vrows = load_jsonl(vp, rep)
            if vrows:
                check_structure(vrows, arm, rep)
                check_truncation(vrows, arm, meta, sb, judge_budget, max_new_warn, rep)
                {"a2": check_a2, "pend": check_pend, "a0": check_a0}[check_family(arm)](vrows, rep, arm)
                if arm == "pend":
                    check_fewshot_leak(vrows, splits, fold, rep)
            check_selection(vp, rl_dir, rep)
    if arm == "pend":
        check_endpoints(meta, rep)
        n_reuse = sum(1 for r in all_rows for s in (r.get("trace") or []) if s.get("ended_planner")
                      and s.get("guard_reasons") and "reuse" in [x for x in s["guard_reasons"] if x])
        name = "pend.close_rejected_as_reuse"
        rep._c(name)
        (rep.warn if n_reuse else rep.note)(name, "closing turns with a candidate rejected as verbatim reuse: %d" % n_reuse)
    return rep


def check_endpoints(meta, rep):
    """pend: R0 and the ledger judge are our gpt-oss-120b (option A); a bypass is recorded, never silent."""
    if not meta.get("arm"):
        return
    for k in ("r0_model", "ledger_judge_model"):
        rep.ok("pend.endpoints", meta.get(k) == "gpt-oss-120b" or meta.get("task1_only"), "meta",
               "%s = %r (spec: gpt-oss-120b)" % (k, meta.get(k)))
    for k in ("r0_base_url", "judge_base_url"):
        u = str(meta.get(k) or "")
        rep.ok("pend.endpoints", ("127.0.0.1:8029" in u or "localhost:8029" in u) or meta.get("task1_only"), "meta",
               "%s = %r (spec: the local gpt-oss server on port 8029)" % (k, u))
    rep.ok("pend.endpoints", not meta.get("endpoint_bypass"), "meta", "PEND_ALLOW_OTHER_ENDPOINTS was set")


def check_selection(vp, rl_dir, rep):
    """validation summaries: D5 settings, the selection score recomputed from its logged parts, best = argmax."""
    summ = [json.loads(l) for l in open(vp, encoding="utf-8") if l.strip() and json.loads(l).get("kind") == "summary"]
    scored = {}
    for v in summ:
        w = "validation u%s" % v.get("update")
        rep.ok("rl.validation_d5", v.get("val_temperature") == 0.7 and sorted(v.get("val_seeds") or []) == [0, 1], w,
               "validation temperature %r / seeds %r (spec D5: 0.7, seeds 0 and 1)" % (v.get("val_temperature"), v.get("val_seeds")))
        if v.get("selection_withheld"):
            rep.warn("rl.validation_withheld", "%s: %s" % (w, v["selection_withheld"]))
            continue
        ts, t1 = v.get("turn_stats") or {}, v.get("task1")
        if v.get("selection_score") is None:
            continue
        want = (v["w_sel_cov"] * ts["coverage_mean"] - v["w_sel_w1"] * ts["turn_w1"]
                + v["w_sel_task1"] * (t1["term_f1"] if t1 else 0.0))
        rep.ok("rl.selection", abs(v["selection_score"] - want) < 1e-9, w,
               "selection_score %r != recomputed %r" % (v["selection_score"], want))
        scored[v["update"]] = v["selection_score"]
    bp = os.path.join(rl_dir, "best.json")
    if scored and os.path.exists(bp):
        best = json.load(open(bp, encoding="utf-8"))
        rep.ok("rl.selection", best.get("selection_score") == max(scored.values()) and best.get("update") in scored,
               bp, "best.json is not the argmax of the validation selection scores")'''),
])

patch1("verify_task1.py", [
    # (Task 1 #9) real rows: Speaker prompt tokens required; a capped plan cannot end
    ('''            rep.ok("trunc.speaker", stoks is None or stoks <= sb, "%s speaker prompt %s > %s" % (w, stoks, sb))''',
     '''            rep.ok("trunc.speaker", stoks is not None and stoks <= sb, "%s speaker prompt %s > %s" % (w, stoks, sb))
            rep.ok("planner_cap", not (r.get("planner_hit_max_new") and r.get("planner_ends_session")),
                   "%s a Planner output cut by the cap ended the session" % w)
            d = r.get("planner_diag") or {}
            n_end_no_complete += bool(d.get("end_without_complete_entry"))
            n_complete_kept += bool(d.get("complete_kept_no_alternative"))'''),
    ('''    n_steps = n_hit = n_unp = n_cand = n_cand_hit = n_first_dup = 0''',
     '''    n_steps = n_hit = n_unp = n_cand = n_cand_hit = n_first_dup = 0
    n_end_no_complete = n_complete_kept = 0'''),
    # (Task 1 #9) K+1 rows: few-shot leakage and selection among the eligible candidates, like the real rows
    ('''        for r in kk:                                   # the K+1 turn is generated like any other: same checks
            w = "%s K+1" % cid[:10]''',
     '''        for r in kk:                                   # the K+1 turn is generated like any other: same checks
            w = "%s K+1" % cid[:10]
            reasons = r.get("guard_reasons") or []
            elig = [i for i, x in enumerate(reasons) if not x]
            if elig and r.get("selected_index") is not None:
                rep.ok("selection", r["selected_index"] in elig, "%s selected %s not in eligible %s" % (w, r["selected_index"], elig))
            for slot in (r.get("fewshot") or []):
                for ex_cid, _ in slot:
                    rep.ok("leakage.fewshot", ex_cid != cid and ex_cid not in forbidden
                           and goal_of.get(ex_cid) is not None and goal_of.get(ex_cid) != goal_of.get(cid)
                           and persona_of.get(ex_cid) is not None and persona_of.get(ex_cid) != persona_of.get(cid),
                           "%s example %s: same conversation/goal/persona, no ids, or validation/test" % (w, ex_cid[:10]))'''),
    # (Task 1 #6) the two recorded fallbacks are counted
    ('''    if n_first_dup:
        rep.warn.append("duplicate first-turn candidates left: %d" % n_first_dup)''',
     '''    if n_first_dup:
        rep.warn.append("duplicate first-turn candidates left: %d" % n_first_dup)
    if n_end_no_complete:
        rep.warn.append("end_session without a Complete entry (drawn act kept): %d" % n_end_no_complete)
    if n_complete_kept:
        rep.warn.append("Complete act kept while not ending (no alternative entry): %d" % n_complete_kept)'''),
])
print("ok")
