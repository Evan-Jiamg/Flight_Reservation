"""Round-4 audit fixes (2026-09-26), batch 3: trainer, controller bounds, Task 1 metrics. Run from sep-sim/."""


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


patch("train_planner_rl.py", [
    # (D7) per-step timing keys are wall-clock noise too
    ('''TIME_KEYS = ("time", "wall_s", "rollout_s", "update_s", "timing", "validation_s")''',
     '''TIME_KEYS = ("time", "wall_s", "rollout_s", "update_s", "timing", "validation_s",
             "planner_s", "speaker_s", "r0_s", "ledger_s")
VAL_RETRIES = 2          # an unclean validation episode (infrastructure incident) is re-run up to this many times'''),
    # (C21b) a torn last line is cut before the next append (otherwise the next row merges into it)
    ('''def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:''',
     '''def append_jsonl(path, row):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\\n":                       # torn last line from a crash: drop it
                f.seek(0)
                data = f.read()
                f.seek(0)
                f.truncate(data.rfind(b"\\n") + 1)
    with open(path, "a", encoding="utf-8") as f:'''),
    # (D5) FakeEnv: turn 1 cannot end (as the spec); fake greedy Task 1 so D2 / task1_base / selection run
    ('''            stop = (rng.random() < p) if planner_temperature > 0 else p > 0.5''',
     '''            stop = ((rng.random() < p) if planner_temperature > 0 else p > 0.5) and t >= 2   # turn 1 cannot end'''),
    ('''    def human_turns(self, conversation_id):
        return 2 + seed_of("human", conversation_id) % 6''',
     '''    def human_turns(self, conversation_id):
        return 2 + seed_of("human", conversation_id) % 6

    def run_task1(self, conversation_id, seed=0, keep_prompts=False):
        """Greedy teacher-forced stop decisions of the fake policy (theta[3] at the real last message,
        theta[1] before it); turn 1 cannot end."""
        n = self.human_turns(conversation_id)
        turns = [{"t": t, "n_real": n, "real_final": t == n, "speaker_blank": False, "planner_unparsed": False,
                  "ended_planner": t >= 2 and _sig(self.learner.theta[3 if t == n else 1]) > 0.5} for t in range(1, n + 1)]
        return {"conversation_id": conversation_id, "n_real": n, "end_mapping": "M2", "k1_speaker_blank": False,
                "turns": turns}'''),
    # (D13) q from ALL clean episodes of the update (singleton groups included); (docs) CRN
    ('''        groups = [[row for row in grp if row["episode"]["clean"]] for grp in groups_all]
        n_unclean = sum(len(g0) - len(g1) for g0, g1 in zip(groups_all, groups))
        n_singletons = sum(1 for g in groups if len(g) == 1)       # a lone clean episode has no baseline
        groups = [g for g in groups if len(g) >= 2]
        clean_eps = [row["episode"] for grp in groups for row in grp]''',
     '''        groups = [[row for row in grp if row["episode"]["clean"]] for grp in groups_all]
        n_unclean = sum(len(g0) - len(g1) for g0, g1 in zip(groups_all, groups))
        n_singletons = sum(1 for g in groups if len(g) == 1)       # a lone clean episode has no baseline
        all_clean = [row["episode"] for grp in groups for row in grp]   # q: every clean rollout of this update
        groups = [g for g in groups if len(g) >= 2]
        clean_eps = [row["episode"] for grp in groups for row in grp]
        # note: the G replicates of a group share the scenario seed (Speaker, act-RNG, few-shot draws), so
        # only the sampled Planner (and R0) differ within a group: common random numbers, by design'''),
    ('''        ctx = self.reward_ctx(clean_eps, cfg)
        shadow_ctx = self.reward_ctx(clean_eps, self.selection_cfg)''',
     '''        ctx = self.reward_ctx(all_clean, cfg)
        shadow_ctx = self.reward_ctx(all_clean, self.selection_cfg)'''),
    ('''        turn_hist = [0] * (tm + 1)
        for e in clean_eps:''',
     '''        turn_hist = [0] * (tm + 1)
        for e in all_clean:'''),
    # (A3) no Task 1 group after a capped emission earlier in the greedy pass (its cut text is a prediction)
    ('''            prompts = self.env.task1_prompts(cid)
            assert len(prompts) == n
            for t in missing:''',
     '''            prompts = self.env.task1_prompts(cid)
            assert len(prompts) == n
            capped_before = [t for t in missing if any(p.get("emitted_capped") for p in prompts[: t - 1])]
            if capped_before:
                with self.io_lock:
                    self.t1_skipped_capped += len(capped_before)
            for t in [t for t in missing if t not in capped_before]:'''),
    ('''        a, pv, psha = self.a, u - 1, self.learner.policy_sha()
        if a.task1_convs <= 0:
            return []''',
     '''        a, pv, psha = self.a, u - 1, self.learner.policy_sha()
        self.t1_skipped_capped = 0
        if a.task1_convs <= 0:
            return []'''),
    ('''                       "n_not_decisions": sum(1 for x in t1_all if not x.get("decision_valid", True)),''',
     '''                       "n_not_decisions": sum(1 for x in t1_all if not x.get("decision_valid", True)),
                       "n_skipped_capped_history": self.t1_skipped_capped,'''),
    # (D2) unclean validation episodes are re-run; if still unclean, the checkpoint gets no selection score
    ('''        def run_val(job):
            cid, s = job
            r = done.get((cid, s))
            if r is None or r["policy_sha"] != psha:
                ep = self.env.run_episode(cid, seed=s, replicate=0, planner_temperature=a.val_temperature,
                                          planner_top_p=a.val_top_p, record_generation=False)
                r = {"kind": "episode", "update": u, "policy_sha": psha, "conversation_id": cid, "seed": s,
                     "split": "validation", "episode": ep, "time": time.time()}
                with self.io_lock:
                    append_jsonl(self.p_val, r)
            return r''',
     '''        def run_val(job):
            cid, s = job
            r = done.get((cid, s))              # the latest attempt (later rows overwrite earlier ones)
            if r is not None and r["policy_sha"] == psha and (r["episode"]["clean"] or r.get("attempt", 0) >= VAL_RETRIES):
                return r
            attempt = r.get("attempt", 0) + 1 if (r is not None and r["policy_sha"] == psha) else 0
            while True:
                ep = self.env.run_episode(cid, seed=s, replicate=0, planner_temperature=a.val_temperature,
                                          planner_top_p=a.val_top_p, record_generation=False)
                r = {"kind": "episode", "update": u, "policy_sha": psha, "conversation_id": cid, "seed": s,
                     "attempt": attempt, "split": "validation", "episode": ep, "time": time.time()}
                with self.io_lock:
                    append_jsonl(self.p_val, r)
                if ep["clean"] or attempt >= VAL_RETRIES:
                    return r
                attempt += 1                    # an infrastructure incident, not the policy: run it again'''),
    ('''        workers = max(1, a.rollout_workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            vrows = list(ex.map(run_val, jobs))
            t1rows = list(ex.map(run_t1, sorted(self.split["validation"]))) if hasattr(self.env, "run_task1") else []
        eps = [r["episode"] for r in vrows if r["episode"]["clean"]]
        n_unclean = len(vrows) - len(eps)''',
     '''        if not hasattr(self.env, "run_task1"):
            raise RuntimeError("validation needs Task 1 (run_task1): D2, task1_base and the selection score use it")
        workers = max(1, a.rollout_workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            vrows = list(ex.map(run_val, jobs))
            t1rows = list(ex.map(run_t1, sorted(self.split["validation"])))
        eps = [r["episode"] for r in vrows if r["episode"]["clean"]]
        n_unclean = len(vrows) - len(eps)
        withheld = n_unclean > 0                # every checkpoint is scored on the SAME validation set, or not at all'''),
    # (D9) the people's turn counts capped at t_max like the simulated ones
    ('''                          "turn_w1": turn_w1([e["emitted_user_turns"] for e in eps], [e["human_turns"] for e in eps]),''',
     '''                          "turn_w1": turn_w1([e["emitted_user_turns"] for e in eps],
                                             [min(int(e["human_turns"]), int(self.selection_cfg["t_max"])) for e in eps]),'''),
    ('''        sel = None
        if turn_stats is not None and turn_stats.get("turn_w1") is not None:''',
     '''        sel = None
        if not withheld and turn_stats is not None and turn_stats.get("turn_w1") is not None:'''),
    # (D3) best.json and the checkpoint's best first, the summary row LAST (a crash in between re-runs the summary)
    ('''        append_jsonl(self.p_val, {"kind": "summary", "update": u, "policy_sha": psha, "split": "validation",
                                  "validation_s": round(time.time() - _tv, 1),''',
     '''        summary = {"kind": "summary", "update": u, "policy_sha": psha, "split": "validation",
                                  "selection_withheld": "unclean validation episode(s) after %d re-runs" % VAL_RETRIES
                                  if withheld else None,
                                  "validation_s": round(time.time() - _tv, 1),'''),
    ('''                                  "selection_cfg_sha256": RR.cfg_sha(self.selection_cfg), "time": time.time()})
        if sel is not None and (self.best is None or sel > self.best["selection_score"]):''',
     '''                                  "selection_cfg_sha256": RR.cfg_sha(self.selection_cfg), "time": time.time()}
        if sel is not None and (self.best is None or sel > self.best["selection_score"]):'''),
    ('''            st = json.load(open(sp))
            st["best"] = self.best
            write_json_atomic(sp, st)

    # -------------------------------------------------------- main loop''',
     '''            st = json.load(open(sp))
            st["best"] = self.best
            write_json_atomic(sp, st)
        append_jsonl(self.p_val, summary)

    # -------------------------------------------------------- main loop'''),
    # (D8) docstring
    ('''     -> validation.jsonl only; best.json = best checkpoint by validation reward under the FIXED initial cfg
     (selection_cfg) + w * validation Task 1 term_f1 (M2).''',
     '''     -> validation.jsonl only; best.json = best checkpoint by w_sel_cov*coverage - w_sel_w1*W1(turn counts)
     + w_sel_task1*Task 1 term_f1 (M2); unclean validation episodes are re-run, else the score is withheld.'''),
    # (C21a) the config file content is part of provenance
    ('''        for k in ("code_sha256", "splits_sha256", "judge_adapter_sha256", "init_adapter_sha256"):''',
     '''        for k in ("code_sha256", "splits_sha256", "judge_adapter_sha256", "init_adapter_sha256", "config_file_sha256"):'''),
    # (C11) every spec setting is gated
    ('''    if a.config:
        ver = (json.load(open(a.config, encoding="utf-8")).get("reward") or {}).get("version", "v4")
        if ver != "v4":
            off["reward.version"] = ver''',
     '''    if a.config:
        cj = json.load(open(a.config, encoding="utf-8"))
        for sect in ("reward", "algo", "llm", "dual"):
            if cj.get(sect):
                off["config." + sect] = cj[sect]            # any override of the declared defaults
    for k, want in (("stop_credit", 1), ("kl", RC.TRAIN_DEFAULTS["kl_coef"]), ("val_temperature", 0.7),
                    ("stop_sup_anneal", 10), ("algo", "grpo")):
        if getattr(a, k) != want:
            off[k] = getattr(a, k)
    if sorted(a.val_seeds) != [0, 1]:
        off["val_seeds"] = a.val_seeds
    if a.task1_convs <= 0:
        off["task1_convs"] = a.task1_convs
    if a.stop_sup_weight == 0:
        off["stop_sup_weight"] = 0.0
    elif not (0.01 <= a.stop_sup_weight <= 5.0):
        ap.error("--stop-sup-weight must be 0 (off) or within the controller bounds [0.01, 5]")'''),
])

# (D6) cfg0 values must lie inside the controller's bounds (no silent clamp at the first decision)
patch("rl_controllers.py", [
    ('''        for k in self.opt["keys"]:
            lo, hi = self.opt["bounds"][k]
            if not (CFG_BOUNDS[k][0] <= lo <= hi <= CFG_BOUNDS[k][1]):
                raise ValueError("controller bounds for %s outside the reward bounds" % k)''',
     '''        for k in self.opt["keys"]:
            lo, hi = self.opt["bounds"][k]
            if not (CFG_BOUNDS[k][0] <= lo <= hi <= CFG_BOUNDS[k][1]):
                raise ValueError("controller bounds for %s outside the reward bounds" % k)
            v = float(self.cfg[k])
            if not (lo <= v <= hi) and not (k == "w_aux" and v == 0.0):
                raise ValueError("initial %s=%g outside the controller bounds [%g, %g]: it would be clamped silently "
                                 "at the first decision" % (k, v, lo, hi))'''),
])

# (A3) Task 1 metrics report capped emissions in the validation conversations
patch("task1_stop.py", [
    ('''    tp = fp = fn = rows_n = unparsed = premature = 0''',
     '''    tp = fp = fn = rows_n = unparsed = premature = capped = 0'''),
    ('''        for r in rows:
            unparsed += bool(r.get("planner_unparsed"))''',
     '''        for r in rows:
            unparsed += bool(r.get("planner_unparsed"))
            capped += bool(r.get("emitted_capped"))'''),
    ('''            "unparsed_rate": unparsed / rows_n}''',
     '''            "unparsed_rate": unparsed / rows_n, "n_emitted_capped_turns": capped}'''),
])
print("ok")
