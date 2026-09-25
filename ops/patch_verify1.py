p = "verify_pipeline.py"
s = open(p, encoding="utf-8").read()


def rep_(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:80])
    s = s.replace(old, new)


# validation.jsonl also holds Task 1 rows (kind "task1"): not episodes
rep_('''                    if r.get("kind") == "summary":
                        continue''',
'''                    if r.get("kind") in ("summary", "task1"):
                        continue''')

# compaction: for the pend arm any compacted prompt is a FAIL (no history is ever dropped)
rep_('''            rep.ok("trunc.planner", n is not None and b is not None and n <= b, w,
                   "prompt_tokens %s budget %s" % (n, b))
            n_pc += bool(pf.get("compacted"))''',
'''            rep.ok("trunc.planner", n is not None and b is not None and n <= b, w,
                   "prompt_tokens %s budget %s" % (n, b))
            n_pc += bool(pf.get("compacted"))
            if arm == "pend":
                rep.ok("trunc.planner_compacted", not pf.get("compacted"), w, "Planner prompt compacted (history dropped)")''')
rep_('''                    rep.ok("trunc.speaker", tok is not None and tok <= speaker_budget, w,
                           "speaker tokens %s budget %s" % (tok, speaker_budget))
                    n_sc += bool(sf.get("compacted"))''',
'''                    rep.ok("trunc.speaker", tok is not None and tok <= speaker_budget, w,
                           "speaker tokens %s budget %s" % (tok, speaker_budget))
                    n_sc += bool(sf.get("compacted"))
                    if arm == "pend":
                        fits = s.get("speaker_fits")
                        rep.ok("trunc.speaker_fits_recorded", isinstance(fits, list) and len(fits) == len(s.get("candidates") or []),
                               w, "per-candidate speaker_fits missing")
                        for j, f in enumerate(fits or []):
                            f = f or {}
                            tk = f.get("final_tokens") if f.get("compacted") else f.get("original_tokens")
                            rep.ok("trunc.speaker", tk is not None and tk <= speaker_budget, w,
                                   "candidate %d speaker tokens %s budget %s" % (j, tk, speaker_budget))
                            rep.ok("trunc.speaker_compacted", not f.get("compacted"), w,
                                   "candidate %d Speaker prompt compacted (history dropped)" % j)''')

# pend: emitted capped message, zero-turn episodes, clean flag, stop/note mask gating
rep_('''    rep.note("pend.end_session", "Planner ends %d; unparsed plans %d" % (n_end, n_unparsed))
    # generation-stage checks (fields written by Task2Env._pend_generate)''',
'''    rep.note("pend.end_session", "Planner ends %d; unparsed plans %d" % (n_end, n_unparsed))
    for r in rows:
        w = "%s s%s" % (str(r.get("conversation_id"))[:12], r.get("seed"))
        rep.ok("pend.zero_turns", (r.get("emitted_user_turns") or 0) > 0, w, "episode with no emitted message")
        if "clean" in r or "episode_counters" in r:
            c = r.get("episode_counters") or {}
            exp = (c.get("r0_len_truncated", 0) == 0 and c.get("judge_empty", 0) == 0 and c.get("judge_unparseable", 0) == 0
                   and not r.get("emitted_capped_steps") and (r.get("emitted_user_turns") or 0) > 0)
            rep.ok("pend.clean_flag", r.get("clean") is exp, w, "clean %r but counters %r" % (r.get("clean"), c))
        else:
            rep.ok("pend.clean_flag", False, w, "episode has no clean flag / per-episode counters")
        for s in r.get("trace") or []:
            ws = "%s t%s" % (w, s.get("t"))
            rep.ok("trunc.emitted_capped", not s.get("emitted_capped"), ws, "a capped (cut) message was emitted")
            g = s.get("planner_gen")
            if isinstance(g, dict):
                d = s.get("planner_diag") or {}
                decision = (not s.get("planner_unparsed") and not g.get("hit_max_new") and s.get("t", 0) >= 2
                            and d.get("end_session_valid") is True and not d.get("end_session_t1_ignored"))
                if not decision:
                    rep.ok("rl.stop_mask_gated", g.get("stop_mask") is None, ws,
                           "stop credit on a step that is not a decision (unparsed / capped / turn 1 / invalid)")
                for key in ("stop_mask", "note_mask"):
                    m = g.get(key)
                    if m is not None:
                        rep.ok("rl.mask_length", len(m) == len(g.get("gen_ids") or []) and 1 in m, ws,
                               "%s length %d != %d generated tokens (or empty)" % (key, len(m), len(g.get("gen_ids") or [])))
    # generation-stage checks (fields written by Task2Env._pend_generate)''')

# RL dir: best.json, validation ids, Task 1 groups gated
rep_('''        check_rl(rl_dir, rollouts, ckpt_pattern, rep)
    return rep''',
'''        check_rl(rl_dir, rollouts, ckpt_pattern, rep)
        check_rl_selection(rl_dir, splits, fold, rep)
    return rep


def check_rl_selection(rl_dir, splits, fold, rep):
    """validation.jsonl only on validation ids (never train, never test); best.json and manifests present."""
    folds = {int(f["fold"]): f for f in splits["folds"]}
    f = folds.get(fold, {})
    val, train_all = set(f.get("validation", [])), set(f.get("train_all", f.get("train", [])))
    vp = os.path.join(rl_dir, "validation.jsonl")
    if rep.ok("rl.validation_present", os.path.exists(vp), vp, "missing"):
        n = 0
        for line in open(vp, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") in ("episode", "task1"):
                n += 1
                cid = r.get("conversation_id")
                rep.ok("leak.validation_ids", cid in val and cid not in train_all, "validation %s" % str(cid)[:12],
                       "validation row on a non-validation id")
        rep.note("rl.validation_present", "%d validation rows" % n)
    rep.ok("rl.best", os.path.exists(os.path.join(rl_dir, "best.json")), rl_dir, "best.json missing")
    for d in sorted(glob.glob(os.path.join(rl_dir, "ckpt", "u*"))):
        if d.endswith(".tmp"):
            continue
        mp = os.path.join(d, "rl_manifest.json")
        if rep.ok("rl.manifest", os.path.exists(mp), d, "rl_manifest.json missing"):
            m = json.load(open(mp, encoding="utf-8"))
            forb = set(f.get("forbidden_for_training", []))
            rep.ok("leak.manifest", not (set(m.get("train_scenarios", [])) | set(m.get("train_conversations", []))
                                         | set(m.get("fewshot_pool", []))) & forb, d, "manifest lists a forbidden id")''')

# Task 1 groups: t >= 2, invalid samples reward 0 and no stop mask
rep_('''                for x in r.get("samples") or []:
                    g = x.get("planner_gen") or {}
                    rep.ok("rl.task1_groups", bool(g.get("prompt_ids")) and bool(g.get("gen_ids")) and
                           x.get("reward") == float(bool(x.get("ended_planner")) == bool(r.get("real_final"))), w,
                           "sample without generation ids or with a reward that disagrees with the label")''',
'''                rep.ok("rl.task1_groups", (r.get("t") or 0) >= 2, w, "Task 1 stop group at turn 1 (cannot end)")
                for x in r.get("samples") or []:
                    g = x.get("planner_gen") or {}
                    valid = x.get("decision_valid", True)
                    exp = float(valid and bool(x.get("ended_planner")) == bool(r.get("real_final")))
                    rep.ok("rl.task1_groups", bool(g.get("prompt_ids")) and bool(g.get("gen_ids")) and
                           x.get("reward") == exp, w,
                           "sample without generation ids or with a reward that disagrees with the label")
                    if not valid:
                        rep.ok("rl.stop_mask_gated", g.get("stop_mask") is None, w, "stop mask on a non-decision sample")''')
open(p, "w", encoding="utf-8", newline="\n").write(s)

# ---------------------------------------------------------------- verify_task1
p = "verify_task1.py"
s = open(p, encoding="utf-8").read()
rep_('''  selection   the selected candidate failed a guard while another passed''',
'''  selection   the selected candidate failed a guard while another passed
  compaction  any Planner or Speaker (any candidate) prompt compacted -- no history is ever dropped
  M2          greedy_ended != (Speaker blank at t OR Planner end at t-1); K+1 ended != (end at n OR blank)
  capped      a capped (cut) message emitted''')
rep_('''            n_hit += bool(r.get("planner_hit_max_new"))
            n_steps += 1
        for r in rs:''',
'''            n_hit += bool(r.get("planner_hit_max_new"))
            n_steps += 1
            last = max(rs, key=lambda x: x["turn_index"]) if rs else None
            if last is not None:
                exp = bool(last.get("planner_ends_session")) or bool(r.get("ended_speaker"))
                rep.ok("M2", r.get("end_mapping") == "M2" and bool(r.get("ended")) == exp and bool(r.get("greedy_ended")) == exp,
                       "%s K+1 ended %r != end at n %r OR blank %r" % (w, r.get("ended"), last.get("planner_ends_session"),
                                                                       r.get("ended_speaker")))
            check_fits(rep, w, r, pb, sb)
        prev_dec = False
        for r in sorted(rs, key=lambda x: x["turn_index"]):''')
rep_('''            reasons = r.get("guard_reasons") or []
            elig = [i for i, x in enumerate(reasons) if not x]''',
'''            exp = bool(r.get("speaker_ended")) or prev_dec
            rep.ok("M2", r.get("end_mapping") == "M2" and bool(r.get("greedy_ended")) == exp
                   and bool(r.get("ended_by_prev_decision")) == prev_dec, "%s greedy_ended %r != blank %r OR end at t-1 %r" % (
                       w, r.get("greedy_ended"), r.get("speaker_ended"), prev_dec))
            se = r.get("samples_speaker_ended")
            if isinstance(se, list):
                rep.ok("M2", [bool(x) or prev_dec for x in se] == [bool(x) for x in (r.get("samples_ended") or [])], w,
                       "samples_ended does not follow M2")
            prev_dec = bool(r.get("planner_ends_session"))
            rep.ok("planner_diag", isinstance(r.get("planner_diag"), dict), w, "planner_diag missing")
            check_fits(rep, w, r, pb, sb)
            reasons = r.get("guard_reasons") or []
            elig = [i for i, x in enumerate(reasons) if not x]''')
rep_('''def main(argv=None):''',
'''def check_fits(rep, w, r, pb, sb):
    pf = r.get("planner_fit") or {}
    rep.ok("compaction.planner", not pf.get("compacted"), "%s Planner prompt compacted" % w)
    fits = r.get("speaker_fits")
    if rep.ok("trunc.speaker_fits_recorded", isinstance(fits, list) and fits, "%s speaker_fits missing" % w):
        for j, f in enumerate(fits):
            f = f or {}
            tk = f.get("final_tokens") if f.get("compacted") else f.get("original_tokens")
            rep.ok("trunc.speaker", tk is not None and tk <= sb, "%s candidate %d speaker prompt %s > %s" % (w, j, tk, sb))
            rep.ok("compaction.speaker", not f.get("compacted"), "%s candidate %d Speaker prompt compacted" % (w, j))
    rep.ok("capped", not r.get("emitted_capped"), "%s a capped message was emitted" % w)


def main(argv=None):''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
