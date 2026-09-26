"""Round-4 audit fixes (2026-09-26), batch 1: prompt / Implicit Profile / Task2Env. Run from sep-sim/."""


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


# ---------------------------------------------------------------- planner_prompt_v3.py
patch("planner_prompt_v3.py", [
    # Complete redraw: drop malformed entries exactly as E1.6 sample_act does
    ('''            mv, ac = acts.normalise(str(e.get("move", "")), str(e.get("act", "")))
            try:
                p = float(e.get("p", 0) or 0)
            except (TypeError, ValueError):
                p = 0.0
            if mv != "Complete" and p > 0:''',
     '''            mv, ac = acts.normalise(str(e.get("move", "")), str(e.get("act", "")))
            if ac == "other" and str(e.get("act", "")).strip().lower() != "other":
                continue                    # malformed entry: dropped, as E1.6 sample_act drops it
            try:
                p = float(e.get("p", 0) or 0)
            except (TypeError, ValueError):
                p = 0.0
            if mv != "Complete" and p > 0:'''),
])

# ---------------------------------------------------------------- implicit_profile.py
patch("implicit_profile.py", [
    ('''    low = (text or "").lower()
    return any(h.strip().lstrip("- ").rstrip(": ").lower() in low for h in (SPK_NOTES, SPK_EXAMPLES)) \\
        or "profile_note" in low''',
     '''    low = (text or "").lower()
    return any(h.strip().lstrip("- ").rstrip(": ").lower() in low for h in (SPK_NOTES, SPK_EXAMPLES)) \\
        or "profile_note" in low or "this is their last message" in low'''),
    ('''Both tasks run from the first message on. After every message the Planner writes a `profile_note`:
how THIS person's writing differs from the message the simulator last produced for them and what to
do differently.''',
     '''Both tasks run from the first message on. After every message the Planner writes a `profile_note`:
how THIS person's writing differs from the message the simulator last produced for them (a description
of their writing, not advice).'''),
])

# ---------------------------------------------------------------- task2_env.py
patch("task2_env.py", [
    # note mask also for a capped generation (a closed profile_note string still gets no advantage)
    ('''    nm = None if g.get("hit_max_new") else planner.field_mask(g["gen_ids"], "profile_note")
    return sm, nm''',
     '''    nm = planner.field_mask(g["gen_ids"], "profile_note")      # None unless a closed string value exists
    return sm, nm'''),
    # judge errors are counted per episode (and make it unclean), then re-raised
    ('''EPISODE_COUNTERS = ("r0_len_retries", "r0_len_truncated", "r0_ctx_fit", "r0_empty",
                    "judge_empty", "judge_unparseable", "judge_retries", "judge_retry_failed")''',
     '''EPISODE_COUNTERS = ("r0_len_retries", "r0_len_truncated", "r0_ctx_fit", "r0_empty",
                    "judge_empty", "judge_unparseable", "judge_retries", "judge_retry_failed", "judge_error")'''),
    ('''    return counts.get("r0_len_truncated", 0) == 0 and counts.get("r0_empty", 0) == 0 \\
        and counts.get("judge_empty", 0) == 0 and counts.get("judge_unparseable", 0) == 0''',
     '''    return counts.get("r0_len_truncated", 0) == 0 and counts.get("r0_empty", 0) == 0 \\
        and counts.get("judge_empty", 0) == 0 and counts.get("judge_unparseable", 0) == 0 \\
        and counts.get("judge_error", 0) == 0'''),
    ('''        def chat(self, system, user, max_tokens=400):
            budget = max(max_tokens, self.floor) if self.floor else max_tokens
            out = super().chat(system, user, budget)
            if not _parses_as_object(out) and not self.gpt5 and JUDGE_RETRY_TOKENS > budget:
                # one re-request at a larger budget (an answer cut by the budget is the usual cause)
                with self._lock:
                    self.n_retries += 1
                _ep_count("judge_retries")
                try:
                    out = super().chat(system, user, JUDGE_RETRY_TOKENS)
                except Exception:                  # e.g. prompt + 8000 over the server context: keep the first answer
                    with self._lock:
                        self.n_retry_failed += 1
                    _ep_count("judge_retry_failed")''',
     '''        def chat(self, system, user, max_tokens=400):
            budget = max(max_tokens, self.floor) if self.floor else max_tokens
            try:
                out = super().chat(system, user, budget)
            except Exception:
                # a verdict lost to an exception would otherwise vanish if the caller swallows it
                with self._lock:
                    self.n_errors += 1
                _ep_count("judge_error")
                raise
            if not _parses_as_object(out) and not self.gpt5 and JUDGE_RETRY_TOKENS > budget:
                # one re-request at a larger budget (an answer cut by the budget is the usual cause), through a
                # judge with its OWN cache directory, so a cached bad answer is never replayed
                with self._lock:
                    self.n_retries += 1
                _ep_count("judge_retries")
                try:
                    out = self._retry_judge().chat_raw(system, user, JUDGE_RETRY_TOKENS)
                except Exception:                  # e.g. prompt + 8000 over the server context: keep the first answer
                    with self._lock:
                        self.n_retry_failed += 1
                    _ep_count("judge_retry_failed")'''),
    ('''            self.n_empty = self.n_unparseable = self.n_retries = self.n_retry_failed = 0''',
     '''            self.n_empty = self.n_unparseable = self.n_retries = self.n_retry_failed = self.n_errors = 0
            self._init_args, self._init_kw = a, dict(kw)
            self._retry = None

        def chat_raw(self, system, user, max_tokens):
            return super().chat(system, user, max_tokens)

        def _retry_judge(self):
            with self._lock:
                if self._retry is None:
                    kw = dict(self._init_kw)
                    if kw.get("cache_dir"):
                        kw["cache_dir"] = os.path.join(kw["cache_dir"], "retry_%d" % JUDGE_RETRY_TOKENS)
                    self._retry = FloorJudge.__bases__[0](*self._init_args, **kw)
                    self._retry.chat_raw = self._retry.chat
            return self._retry'''),
    # judge cache keyed by model and endpoint: never replays verdicts of another judge
    ('''        self.ledger_judge = make_floor_judge(Judge)(reasoning_effort=self.judge_effort, verbose=False,
                                                    cache_dir=os.path.join(WORK, "judge_cache"))''',
     '''        jkey = hashlib.sha256(("%s|%s" % (os.environ.get("JUDGE_MODEL", "default"),
                                          os.environ.get("JUDGE_BASE_URL", "default"))).encode()).hexdigest()[:12]
        self.judge_cache_dir = os.path.join(WORK, "judge_cache_%s" % jkey)      # one cache per judge model+endpoint
        self.ledger_judge = make_floor_judge(Judge)(reasoning_effort=self.judge_effort, verbose=False,
                                                    cache_dir=self.judge_cache_dir)'''),
    # endpoint: the local 8029 server; the bypass is recorded
    ('''            for k in ("R0_BASE_URL", "JUDGE_BASE_URL"):
                if not os.environ.get(k) or "api.openai.com" in os.environ[k]:
                    raise RuntimeError("%s must point at the local gpt-oss-120b server for the pend arm" % k)''',
     '''            for k in ("R0_BASE_URL", "JUDGE_BASE_URL"):
                u = os.environ.get(k) or ""
                if not any(h in u for h in ("127.0.0.1:%s" % PEND_PORT, "localhost:%s" % PEND_PORT)):
                    raise RuntimeError("%s must point at the local gpt-oss-120b server (port %s) for the pend arm, got %r"
                                       % (k, PEND_PORT, u))'''),
    ('''JUDGE_RETRY_TOKENS = int(os.environ.get("JUDGE_RETRY_TOKENS", "8000"))   # one re-request of an unparseable verdict''',
     '''JUDGE_RETRY_TOKENS = int(os.environ.get("JUDGE_RETRY_TOKENS", "8000"))   # one re-request of an unparseable verdict
PEND_PORT = os.environ.get("PEND_R0_PORT", "8029")        # our gpt-oss-120b vLLM server (user decision, option A)'''),
    ('''                "selector": self.selector, "speaker_max_new": self.speaker.max_new, "planner_max_new": self.planner.max_new,''',
     '''                "selector": self.selector, "speaker_max_new": self.speaker.max_new, "planner_max_new": self.planner.max_new,
                "judge_cache_dir": getattr(self, "judge_cache_dir", None), "task1_only": self.task1_only,
                "endpoint_bypass": os.environ.get("PEND_ALLOW_OTHER_ENDPOINTS") == "1",'''),
    # Task 1 only: no R0 / ledger judge needed (and no endpoint requirement)
    ('''    def __init__(self, arm, gpu, planner, judge=None, ditto_path=DITTO, corpus="/home/mzjiang/v5-latency/data.jsonl",
                 batch=False, max_batch=8, implicit_profile=False, fewshot_pool=None, selector="length"):''',
     '''    def __init__(self, arm, gpu, planner, judge=None, ditto_path=DITTO, corpus="/home/mzjiang/v5-latency/data.jsonl",
                 batch=False, max_batch=8, implicit_profile=False, fewshot_pool=None, selector="length", task1_only=False):
        self.task1_only = bool(task1_only)     # Task 1 generation only: no R0 agent, no ledger judge'''),
    ('''        if arm == "pend" and os.environ.get("PEND_ALLOW_OTHER_ENDPOINTS") != "1":''',
     '''        if arm == "pend" and not self.task1_only and os.environ.get("PEND_ALLOW_OTHER_ENDPOINTS") != "1":'''),
    # ledger read-outs inside the episode scope (any lazy judge call is attributed to this episode)
    ('''        orphans0 = orphan_incidents()
        episode_begin()
        try:
            ep = run_episode(T_MAX, None, speak, respond, planner_end=(arm in EMIT_END_ARMS))
        finally:
            counts = episode_end()''',
     '''        if self.task1_only:
            raise RuntimeError("this Task2Env was built for Task 1 only (no R0 / ledger judge)")
        orphans0 = orphan_incidents()
        episode_begin()
        try:
            ep = run_episode(T_MAX, None, speak, respond, planner_end=(arm in EMIT_END_ARMS))
            cov_final, complete_final, ledger_dict = round(ledger.coverage(), 4), ledger.complete(), ledger.as_dict()
        finally:
            counts = episode_end()'''),
    ('''                "coverage": round(ledger.coverage(), 4), "complete": ledger.complete(),
                "n_req": len(self.reqs[conversation_id]["req"]), "ledger": ledger.as_dict(), "trace": ep["trace"],''',
     '''                "coverage": cov_final, "complete": complete_final,
                "n_req": len(self.reqs[conversation_id]["req"]), "ledger": ledger_dict, "trace": ep["trace"],'''),
    # Task 1 samples_ended: the Speaker flags, as E1.6 writes them (M2 is defined for greedy_ended only)
    ('''                         "samples": samples, "samples_ended": [b or prev_decision for b in s_blank],
                         "samples_speaker_ended": s_blank, "profile": st.get("block"),''',
     '''                         "samples": samples, "samples_ended": s_blank,
                         "samples_speaker_ended": s_blank, "profile": st.get("block"),'''),
    # Task 1 prompts carry the cap evidence of every turn (groups after a capped emission are skipped)
    ('''        return [{"t": r["turn_index"], "n_real": n, "real_final": r["turn_index"] == n, "user_prompt": r["planner_prompt"],
                 "planner_fit": r.get("planner_fit")} for r in rows]''',
     '''        return [{"t": r["turn_index"], "n_real": n, "real_final": r["turn_index"] == n, "user_prompt": r["planner_prompt"],
                 "planner_fit": r.get("planner_fit"), "emitted_capped": bool(r.get("emitted_capped")),
                 "planner_hit_max_new": bool(r.get("planner_hit_max_new"))} for r in rows]'''),
    # a real decision is scored whether or not its stop mask could be located
    ('''            unparsed = fields is None
            sm, nm = rl_masks(self.planner, gen, unparsed, diag, t)
            valid = sm is not None''',
     '''            unparsed = fields is None
            sm, nm = rl_masks(self.planner, gen, unparsed, diag, t)
            # a decision = parsed, not capped, a valid end_session value, t >= 2 (the reward does not depend on
            # whether the stop mask could be located; without a mask the sample just gets no stop credit)
            valid = (not unparsed and not gen["hit_max_new"] and (diag or {}).get("end_session_valid") is True
                     and not (diag or {}).get("end_session_t1_ignored"))'''),
    ('''        # one stop-supervision example per position: the first VALID sample
        for x in out:
            if not x["decision_valid"]:
                continue''',
     '''        # one stop-supervision example per position: the first VALID sample with a located stop mask
        for x in out:
            if not x["decision_valid"] or x["planner_gen"]["stop_mask"] is None:
                continue'''),
    # docstrings
    ('''        """Task 1 (teacher-forced) generations for one REAL conversation, in the benchmark's generations
        schema (tools/score_method.py): one row per real user turn with greedy (the selected candidate),
        samples (the other candidates), greedy_ended (Speaker end OR the Planner's end, as E1.6's
        PLANNER_END records it).''',
     '''        """Task 1 (teacher-forced) generations for one REAL conversation, in the benchmark's generations
        schema (tools/score_method.py): one row per real user turn with greedy (the selected candidate),
        samples (the other candidates), greedy_ended under M2 (Speaker blank at t OR the Planner's end
        decided at t-1), samples_ended = the Speaker flags (E1.6 convention).'''),
    ('''  * PlannerLM   any causal LM (32B NF4, or a smaller 7-9B/20B model), optional LoRA adapter''',
     '''  * PlannerLM   any causal LM (pend: Qwen3-4B-Instruct-2507), optional LoRA adapter'''),
])
print("ok")
