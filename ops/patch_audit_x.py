"""Fixes from the cross-cutting audit (2026-09-25), open items O1, O3, O5, O7-O11, O13, O14. Run from sep-sim/."""


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


# ---------------------------------------------------------------- task2_env.py (O1, O3, O9)
patch("task2_env.py", [
    ('''def _ep_count(name):
    c = getattr(_EP, "counts", None)
    if c is not None:
        c[name] += 1''',
     '''_ORPHANS = {"n": 0}
_ORPHAN_LOCK = threading.Lock()


def _ep_count(name):
    c = getattr(_EP, "counts", None)
    if c is not None:
        c[name] += 1
    else:
        # an incident on a thread with no active episode: if an episode is running, the one-episode-per-thread
        # assumption is broken (e.g. the Ledger fans judge calls out to a pool) -- run_episode refuses then
        with _ORPHAN_LOCK:
            _ORPHANS["n"] += 1


def orphan_incidents():
    return _ORPHANS["n"]'''),
    ('''        episode_begin()
        try:
            ep = run_episode(T_MAX, None, speak, respond, planner_end=(arm in EMIT_END_ARMS))
        finally:
            counts = episode_end()''',
     '''        orphans0 = orphan_incidents()
        episode_begin()
        try:
            ep = run_episode(T_MAX, None, speak, respond, planner_end=(arm in EMIT_END_ARMS))
        finally:
            counts = episode_end()
        if orphan_incidents() != orphans0:
            raise RuntimeError("an R0/judge incident was counted outside any episode thread while %s ran: per-episode "
                               "attribution is broken, so no episode can be marked clean" % conversation_id)'''),
    ('''                with self._lock:
                    self.n_retries += 1
                _ep_count("judge_retries")
                out = super().chat(system, user, JUDGE_RETRY_TOKENS)''',
     '''                with self._lock:
                    self.n_retries += 1
                _ep_count("judge_retries")
                try:
                    out = super().chat(system, user, JUDGE_RETRY_TOKENS)
                except Exception:                  # e.g. prompt + 8000 over the server context: keep the first answer
                    with self._lock:
                        self.n_retry_failed += 1
                    _ep_count("judge_retry_failed")'''),
    ('''            self.n_empty = self.n_unparseable = self.n_retries = 0''',
     '''            self.n_empty = self.n_unparseable = self.n_retries = self.n_retry_failed = 0'''),
    ('''EPISODE_COUNTERS = ("r0_len_retries", "r0_len_truncated", "r0_ctx_fit", "r0_empty",
                    "judge_empty", "judge_unparseable", "judge_retries")''',
     '''EPISODE_COUNTERS = ("r0_len_retries", "r0_len_truncated", "r0_ctx_fit", "r0_empty",
                    "judge_empty", "judge_unparseable", "judge_retries", "judge_retry_failed")'''),
    ('''        self.r0_effort = os.environ.get("R0_REASONING_EFFORT", "minimal")''',
     '''        if arm == "pend" and os.environ.get("PEND_ALLOW_OTHER_ENDPOINTS") != "1":
            # user decision (option A): R0 and the ledger judge are our own gpt-oss-120b on vLLM
            for k in ("R0_BASE_URL", "JUDGE_BASE_URL"):
                if not os.environ.get(k) or "api.openai.com" in os.environ[k]:
                    raise RuntimeError("%s must point at the local gpt-oss-120b server for the pend arm" % k)
            for k in ("R0_MODEL", "JUDGE_MODEL"):
                if os.environ.get(k) != "gpt-oss-120b":
                    raise RuntimeError("%s must be gpt-oss-120b for the pend arm (got %r)" % (k, os.environ.get(k)))
        self.r0_effort = os.environ.get("R0_REASONING_EFFORT", "minimal")'''),
])

# ---------------------------------------------------------------- implicit_profile.py (O5)
patch("implicit_profile.py", [
    ('''        self.goal_of, self.persona_of = dict(goal_of), dict(persona_of)
        self.allowed = set(allowed)''',
     '''        self.goal_of, self.persona_of = dict(goal_of), dict(persona_of)
        self.allowed = set(allowed)
        missing = sorted(c for c in self.allowed if self.goal_of.get(c) is None or self.persona_of.get(c) is None)
        if missing:
            # without the ids, an example sharing the goal/persona could not be excluded
            raise KeyError("pool conversations without goal/persona ids: %s" % missing[:5])'''),
])

# ---------------------------------------------------------------- verify_task1.py (O5, O11)
patch("verify_task1.py", [
    ('''                    rep.ok("leakage.fewshot", goal_of.get(ex_cid) != goal_of.get(cid), "%s example shares the goal" % w)
                    rep.ok("leakage.fewshot", persona_of.get(ex_cid) != persona_of.get(cid), "%s example shares the persona" % w)''',
     '''                    rep.ok("leakage.fewshot", goal_of.get(ex_cid) is not None and goal_of.get(ex_cid) != goal_of.get(cid),
                           "%s example shares the goal (or has no goal id)" % w)
                    rep.ok("leakage.fewshot", persona_of.get(ex_cid) is not None and persona_of.get(ex_cid) != persona_of.get(cid),
                           "%s example shares the persona (or has no persona id)" % w)'''),
    ('''            pt = r.get("planner_prompt_tokens")
            rep.ok("trunc.planner", pt is None or pt <= pb, "%s prompt %s > %s" % (w, pt, pb))''',
     '''            pt = r.get("planner_prompt_tokens")
            rep.ok("trunc.planner", pt is not None and pt <= pb, "%s prompt %s > %s" % (w, pt, pb))'''),
    ('''    forbidden = set()
    if a.fold >= 0:''',
     '''    forbidden = set()
    if a.fold < 0 and meta.get("fold", -1) >= 0:
        a.fold = int(meta["fold"])                     # a fold run is always checked against its fold
    if meta.get("sessions") in ("fold-validation", "fold-test") and a.fold < 0:
        rep.ok("leakage.fold", False, "fold run without a fold: the forbidden-pool check cannot run")
    if a.fold >= 0 and meta.get("sessions") in ("fold-validation", "fold-test"):
        spx = [x for x in json.load(open(a.splits, encoding="utf-8"))["folds"] if x["fold"] == a.fold][0]
        want = sorted(spx["validation"] if meta["sessions"] == "fold-validation" else spx["test_all"])
        rep.ok("structure", sorted(meta.get("session_ids") or []) == want,
               "scored ids are not splits[%d].%s" % (a.fold, "validation" if meta["sessions"] == "fold-validation" else "test_all"))
    if a.fold >= 0:'''),
])

# ---------------------------------------------------------------- verify_pipeline.py (O5, O7)
patch("verify_pipeline.py", [
    ('''    check_leakage(all_rows, splits, fold, split, training, rep)''',
     '''    check_leakage(all_rows, splits, fold, split, training, rep)
    if arm == "pend":
        check_fewshot_leak(all_rows, splits, fold, rep)
        if not training:
            # evaluation: an unclean episode (cut R0 reply, lost ledger verdict, capped emission, compaction)
            # has a wrong coverage/turn count and cannot be silently averaged in -- rerun it
            for r in rows:
                rep.ok("eval.clean", r.get("clean") is True, "%s s%s" % (str(r.get("conversation_id"))[:12], r.get("seed")),
                       "evaluation episode is not clean: %r" % (r.get("episode_counters"),))'''),
    ('''def check_rl_selection(rl_dir, splits, fold, rep):''',
     '''FOLDS_GP = "/tmp2/hchsu/trec2026-usersim-benchmark/domains/main_dataset_search/folds3_goal_persona_v1.json"


def check_fewshot_leak(rows, splits, fold, rep, folds_gp=FOLDS_GP):
    """Task 2 few-shot examples: never the same conversation, goal or persona; never validation/test."""
    used = [(r, s) for r in rows for s in (r.get("trace") or []) if s.get("fewshot")]
    if not used:
        return
    if not os.path.exists(folds_gp):
        rep.ok("leak.fewshot", False, "fewshot", "goal/persona manifest %s missing: cannot check" % folds_gp)
        return
    G = json.load(open(folds_gp, encoding="utf-8"))
    goal_of, persona_of = G["goal_of"], G["persona_of"]
    f = {int(x["fold"]): x for x in splits["folds"]}.get(fold, {})
    forb = set(f.get("forbidden_for_training", []))
    for r, s in used:
        cid = r.get("conversation_id")
        w = "%s t%s" % (str(cid)[:12], s.get("t"))
        for slot in s["fewshot"]:
            for ex_cid, _ in slot:
                ok = (ex_cid != cid and ex_cid not in forb and goal_of.get(ex_cid) is not None
                      and goal_of.get(ex_cid) != goal_of.get(cid) and persona_of.get(ex_cid) is not None
                      and persona_of.get(ex_cid) != persona_of.get(cid))
                rep.ok("leak.fewshot", ok, w, "example %s: same conversation/goal/persona, no ids, or validation/test" % str(ex_cid)[:12])


def check_rl_selection(rl_dir, splits, fold, rep):'''),
])

# ---------------------------------------------------------------- rl_controllers.py (O8)
patch("rl_controllers.py", [
    ('''            resp = self.transport(req)
            rec["response_sha256"] = sha256(json.dumps(resp, sort_keys=True))
            content = resp["choices"][0]["message"].get("content") or ""
            rec["response_content"] = content[:4000]
            prop = first_json_object(content)
            facs = prop.get("factors") or {}''',
     '''            resp = self.transport(req)
            rec["response_sha256"] = sha256(json.dumps(resp, sort_keys=True))
            rec["finish_reason"] = resp["choices"][0].get("finish_reason")
            content = resp["choices"][0]["message"].get("content") or ""
            rec["response_content"] = content[:4000]
            if rec["finish_reason"] == "length":
                raise ValueError("controller reply cut by max_tokens=%s" % self.opt["max_tokens"])
            prop = first_json_object(content)
            facs = prop.get("factors") or {}'''),
])

# ---------------------------------------------------------------- train_planner_rl.py (O8, O13, O14)
patch("train_planner_rl.py", [
    ('''    assert not set(f.get("train_all", train)) & forbidden, "train_all intersects forbidden_for_training"''',
     '''    assert not set(f.get("train_all", train)) & forbidden, "train_all intersects forbidden_for_training"
    for k, n in (f.get("sizes") or {}).items():          # the declared sizes of the split file
        assert len(f[k]) == n, "splits fold %d: %s has %d ids, declared %d" % (fold, k, len(f[k]), n)'''),
    ('''               "train_aggregate": hist, "learner_stats": stats, "n_samples": len(samples),''',
     '''               "train_aggregate": hist, "learner_stats": stats, "n_samples": len(samples),
               "controller_failures": getattr(self.controller, "n_failures", None),
               "controller_rollbacks": getattr(self.controller, "n_rollbacks", None),'''),
    ('''    if a.controller != "llm":
        off["controller"] = a.controller''',
     '''    if a.controller != "llm":
        off["controller"] = a.controller
    if a.planner_path and "Qwen3-4B-Instruct-2507" not in a.planner_path:
        off["planner_path"] = a.planner_path          # the spec's Planner is Qwen3-4B-Instruct-2507'''),
])

# ---------------------------------------------------------------- rollout_v4.py / task1_v4.py (O10)
patch("rollout_v4.py", [
    ('''    m = json.load(open(path))
    if "fold" in m and int(m["fold"]) != fold:
        raise SystemExit("LEAK GATE: %s trained for fold %s, evaluating fold %d" % (what, m["fold"], fold))''',
     '''    m = json.load(open(path))
    if "fold" in m and int(m["fold"]) != fold:
        raise SystemExit("LEAK GATE: %s trained for fold %s, evaluating fold %d" % (what, m["fold"], fold))
    if m.get("splits_sha256") and SPLITS_SHA.get("sha") and m["splits_sha256"] != SPLITS_SHA["sha"]:
        raise SystemExit("LEAK GATE: %s was trained with another split file (sha differs)" % what)'''),
    ('''def check_manifest(path, fold, split_fold, what):''',
     '''SPLITS_SHA = {"sha": None}


def check_manifest(path, fold, split_fold, what):'''),
    ('''    gate = {"fold": args.fold, "split": args.split, "n_scenarios": len(scen), "splits_sha256": sha_file(args.splits)}''',
     '''    gate = {"fold": args.fold, "split": args.split, "n_scenarios": len(scen), "splits_sha256": sha_file(args.splits)}
    SPLITS_SHA["sha"] = gate["splits_sha256"]'''),
])
patch("task1_v4.py", [
    ('''            used = set(man.get("train_scenarios", [])) | set(man.get("train_conversations", [])) | set(man.get("fewshot_pool", []))''',
     '''            used = set(man.get("train_scenarios", [])) | set(man.get("train_conversations", [])) | set(man.get("fewshot_pool", []))
            if man.get("splits_sha256") and man["splits_sha256"] != sha_file(a.splits):
                raise SystemExit("LEAK GATE: adapter was trained with another split file (sha differs)")
            if used & set(f["forbidden_for_training"]):
                raise SystemExit("LEAK GATE: adapter used validation/test conversations: %s" % sorted(used & set(f["forbidden_for_training"]))[:5])'''),
])
print("ok")
