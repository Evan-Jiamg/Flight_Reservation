"""Fixes from the GRPO + Task 2 audit (2026-09-25). Run from sep-sim/."""


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


# ---------------------------------------------------------------- task2_env.py
patch("task2_env.py", [
    # (10) a capped Planner output is never read: no ledger gain, no act-RNG draw
    ('''            elif arm == "pend":
                fields, diag, end_session = V3.read_plan_pend(raw, t, scenario, rng, led)''',
     '''            elif arm == "pend" and g["hit_max_new"]:
                # a cut output is not a plan: it is not read at all (no stopping-ledger gain, no act-RNG draw)
                fields, diag, end_session = None, {"cut_by_max_new": True}, False
            elif arm == "pend":
                fields, diag, end_session = V3.read_plan_pend(raw, t, scenario, rng, led)'''),
    # (12) a final empty R0 reply is an incident of this episode
    ('''EPISODE_COUNTERS = ("r0_len_retries", "r0_len_truncated", "r0_ctx_fit",
                    "judge_empty", "judge_unparseable", "judge_retries")''',
     '''EPISODE_COUNTERS = ("r0_len_retries", "r0_len_truncated", "r0_ctx_fit", "r0_empty",
                    "judge_empty", "judge_unparseable", "judge_retries")'''),
    ('''    return counts.get("r0_len_truncated", 0) == 0 and counts.get("judge_empty", 0) == 0 \\
        and counts.get("judge_unparseable", 0) == 0''',
     '''    return counts.get("r0_len_truncated", 0) == 0 and counts.get("r0_empty", 0) == 0 \\
        and counts.get("judge_empty", 0) == 0 and counts.get("judge_unparseable", 0) == 0'''),
    ('''            self.n_len_retries = self.n_len_truncated = self.n_ctx_fit = 0
            self._tlock = threading.Lock()
''',
     '''            self.n_len_retries = self.n_len_truncated = self.n_ctx_fit = self.n_empty_final = 0
            self._tlock = threading.Lock()

        def reply(self, *a, **kw):
            out = super().reply(*a, **kw)
            if not (out or "").strip():            # still empty after the client's own budget ladder
                with self._tlock:
                    self.n_empty_final += 1
                _ep_count("r0_empty")
            return out
'''),
    # (13) the stop mask must cover the value that was actually parsed
    ('''    sm = planner.stop_mask(g["gen_ids"]) if ok else None
    nm = None if g.get("hit_max_new") else planner.field_mask(g["gen_ids"], "profile_note")
    return sm, nm''',
     '''    sm = planner.stop_mask(g["gen_ids"]) if ok else None
    if sm is not None:
        raw = diag.get("end_session_raw")
        want = "true" if (raw is True or str(raw).strip().lower() == "true") else "false"
        txt = planner.tok.decode([x for x, m in zip(g["gen_ids"], sm) if m], skip_special_tokens=False)
        if want not in txt.lower():
            diag["stop_mask_mismatch"] = True      # the first match is not the parsed value: no stop credit
            sm = None
    nm = None if g.get("hit_max_new") else planner.field_mask(g["gen_ids"], "profile_note")
    return sm, nm'''),
    # (7) Task 1 records keep the fit / cap evidence
    ('''        return [{"t": r["turn_index"], "n_real": n, "real_final": r["turn_index"] == n, "user_prompt": r["planner_prompt"]}
                for r in rows]''',
     '''        return [{"t": r["turn_index"], "n_real": n, "real_final": r["turn_index"] == n, "user_prompt": r["planner_prompt"],
                 "planner_fit": r.get("planner_fit")} for r in rows]'''),
    ('''        for g in range(G):
            pseed, gen = seeds[g], gens[g]
            fields, diag, end = V3.read_plan_pend(gen["raw"], t, scenario, random.Random(pseed),
                                                  stopping.StoppingLedger(scenario))
            if gen["hit_max_new"] and fields is not None:
                diag = dict(diag or {}, cut_by_max_new=True)
                fields, end = None, False''',
     '''        for g in range(G):
            pseed, gen = seeds[g], gens[g]
            if (gen.get("fit") or {}).get("compacted"):
                raise RuntimeError("Task 1 group %s t%d: Planner prompt compacted (history dropped)" % (conversation_id, t))
            if gen["hit_max_new"]:
                fields, diag, end = None, {"cut_by_max_new": True}, False
            else:
                fields, diag, end = V3.read_plan_pend(gen["raw"], t, scenario, random.Random(pseed),
                                                      stopping.StoppingLedger(scenario))'''),
    ('''                        "decision_valid": valid, "planner_diag": diag,''',
     '''                        "decision_valid": valid, "planner_diag": diag, "planner_fit": gen.get("fit"),'''),
    ('''                 "goal_met": r.get("goal_met"), "planner_diag": r.get("planner_diag"),
                 "planner_fit": r.get("planner_fit")}''',
     '''                 "goal_met": r.get("goal_met"), "planner_diag": r.get("planner_diag"),
                 "planner_fit": r.get("planner_fit"), "speaker_fits": r.get("speaker_fits"),
                 "speaker_hit_max_new": r.get("speaker_hit_max_new"), "emitted_capped": r.get("emitted_capped")}'''),
    # (19) header
    ('''  * arms        a0 = original Planner prompt/read_plan (override on); Planner end logged only.
                a2 = v3 system/user prompt, read_plan_v3 (stop = end_session, no length clamp),
                     judge GOAL STATUS, silent Planner exit.''',
     '''  * arms        pend (the method, user 2026-09-25) = E1.6 tree, no annotations, no goal judge; the Planner
                     judges the goal and its end_session makes the planned message the last one (emitted
                     close); Implicit Profile, per-slot few-shot, Borda selector (see ops/AUDIT_SPEC_pend_grpo.md).
                e16 = the E1.6 baseline as generated; a0 / a2 / final = older arms kept for comparison only.'''),
])

# ---------------------------------------------------------------- planner_prompt_v3.py (11)
patch("planner_prompt_v3.py", [
    ('''            if mv not in ("Complete", "Other") and p > 0:
                rest.append((p, mv, ac, e.get("length_words")))''',
     '''            if mv != "Complete" and p > 0:
                rest.append((p, mv, ac, e.get("length_words")))'''),
    ('''            diag["complete_redrawn_to"] = [mv, ac]''',
     '''            diag["complete_redrawn_to"] = [mv, ac]
        else:
            diag["complete_kept_no_alternative"] = True   # no non-Complete entry with p > 0: counted by verify'''),
])

# ---------------------------------------------------------------- rl_reward.py (15)
patch("rl_reward.py", [
    ('''    turns = int(episode["emitted_user_turns"])
    T = min(max(turns, 0), t_max)
    dist = math.log(p_h[T]) - math.log(q[T])
    cov = float(episode["coverage"])
    if not (0.0 <= cov <= 1.0):
        raise ValueError("coverage %r outside [0, 1]" % cov)
    counts = {
        "unparsed": sum(bool(s.get("planner_unparsed")) for s in trace),''',
     '''    turns = int(episode["emitted_user_turns"])
    T = min(max(turns, 0), t_max)
    dist = math.log(p_h[T]) - math.log(q[T])
    cov = float(episode["coverage"])
    if not (0.0 <= cov <= 1.0):
        raise ValueError("coverage %r outside [0, 1]" % cov)
    counts = {
        # a capped plan is also unparsed: it pays lambda_hit_max_new only, not both
        "unparsed": sum(bool(s.get("planner_unparsed")) and not bool(s.get("planner_hit_max_new")) for s in trace),'''),
])

# ---------------------------------------------------------------- rl_controllers.py (3)
patch("rl_controllers.py", [
    ('''            changed = any(abs(cfg[k] - self.cfg[k]) > 1e-12 for k in self.opt["keys"])
            if changed:
                self.pending = {"prev_cfg": copy.deepcopy(self.cfg), "baseline": shadow, "bad": 0}''',
     '''            changed = any(abs(cfg[k] - self.cfg[k]) > 1e-12 for k in self.opt["keys"])
            if changed:
                if self.pending is not None and self.pending["bad"] > 0:
                    # an earlier change is still under suspicion: keep ITS last-good cfg, baseline and count, so
                    # a second worse point rolls back to before it even though the LLM changed something between
                    pass
                else:
                    self.pending = {"prev_cfg": copy.deepcopy(self.cfg), "baseline": shadow, "bad": 0}'''),
])

# ---------------------------------------------------------------- train_planner_rl.py (17, 18a, 19)
patch("train_planner_rl.py", [
    ('''                ep = self.env.run_episode(cid, seed=s, replicate=s, planner_temperature=a.val_temperature,''',
     '''                ep = self.env.run_episode(cid, seed=s, replicate=0, planner_temperature=a.val_temperature,'''),
    ('''        groups = [[row for row in grp if row["episode"]["clean"]] for grp in groups_all]
        n_unclean = sum(len(g0) - len(g1) for g0, g1 in zip(groups_all, groups))
        groups = [g for g in groups if len(g) >= 2]''',
     '''        groups = [[row for row in grp if row["episode"]["clean"]] for grp in groups_all]
        n_unclean = sum(len(g0) - len(g1) for g0, g1 in zip(groups_all, groups))
        n_singletons = sum(1 for g in groups if len(g) == 1)       # a lone clean episode has no baseline
        groups = [g for g in groups if len(g) >= 2]'''),
    ('''                "n_groups": len(groups), "n_groups_skipped_zero_std": skipped, "n_unclean_episodes": n_unclean,''',
     '''                "n_groups": len(groups), "n_groups_skipped_zero_std": skipped, "n_unclean_episodes": n_unclean,
                "n_dropped_singleton_episodes": n_singletons,'''),
    ('''"""Planner RL on Task 2 (GRPO / RLOO / PPO; fixed / dual-ascent / LLM controller).

Loop (update u = 1, 2, ...; the policy that generates update u's rollouts has policy_version u-1):
  1. sample --scenarios-per-update scenarios (seeded by (seed, u)) from splits[fold]["train"] ONLY;
     every id is asserted to be in train and not in forbidden_for_training;
  2. G rollouts each: Task2Env(arm="a2").run_episode(..., planner_temperature>0, record_generation=True);
     each row is appended to rollouts.jsonl with update, policy_version and policy_sha;
  3. rewards = rl_reward.reward_v2(episode, cfg) with the controller's current cfg;
  4. advantages (rl_algos, pure python) -> samples (ids exactly as recorded) -> learner.update;
     assert every sample was produced by the current policy_version (on-policy);
  5. checkpoint EVERY update (adapter, optimizer, value head, controller state, RNG states, update index,
     policy_version, history) atomically; then updates.jsonl;
  6. controller.propose(train aggregates only) -> cfg for the next update;
  7. every --val-every updates: greedy episodes on splits[fold]["validation"] -> validation.jsonl only;
     best.json = best checkpoint by mean validation reward under the FIXED initial cfg (selection_cfg).
Validation never enters history, reward statistics or the controller; the test ids are never read.''',
     '''"""Planner RL for the pend arm (GRPO; v4 LLM factor controller). Design: ops/AUDIT_SPEC_pend_grpo.md.

Loop (update u = 1, 2, ...; the policy that generates update u's rollouts has policy_version u-1):
  1. sample --scenarios-per-update scenarios (seeded by (seed, u)) from splits[fold]["train"] ONLY;
     every id is asserted to be in train and not in forbidden_for_training;
  2. G rollouts each: Task2Env(arm="pend").run_episode(..., planner_temperature>0, record_generation=True);
     each row is appended to rollouts.jsonl with update, policy_version and policy_sha; episodes that are
     not clean (cut R0 reply, lost ledger verdict, emitted capped message, compacted prompt) are dropped;
  3. rewards = rl_reward.reward(episode, cfg, ctx) -- v4: coverage + log p_h(T) - log q(T) - penalties, p_h
     from splits[fold]["train_all"], q from this update's clean rollouts;
  4. advantages normalised once per group, split into the stop part (end_session tokens) and the rest;
     Task 1 stop groups on train_all conversations + the annealed stop supervision (D2);
     assert every sample was produced by the current policy_version (on-policy);
  5. checkpoint EVERY update (adapter, optimizer, controller state, RNG states, update index, policy_version,
     history, rl_manifest.json) atomically; then updates.jsonl;
  6. controller.propose(train aggregates only) -> cfg for the next update;
  7. every --val-every updates: SAMPLED-Planner episodes (D5) and greedy Task 1 on splits[fold]["validation"]
     -> validation.jsonl only; best.json = best checkpoint by validation reward under the FIXED initial cfg
     (selection_cfg) + w * validation Task 1 term_f1 (M2).
Validation never enters history, reward statistics or the controller; the test ids are never read.'''),
])

# ---------------------------------------------------------------- verify_pipeline.py (8, 11, 12)
patch("verify_pipeline.py", [
    ('''            exp = (c.get("r0_len_truncated", 0) == 0 and c.get("judge_empty", 0) == 0 and c.get("judge_unparseable", 0) == 0
                   and not r.get("emitted_capped_steps") and (r.get("emitted_user_turns") or 0) > 0)''',
     '''            exp = (c.get("r0_len_truncated", 0) == 0 and c.get("r0_empty", 0) == 0 and c.get("judge_empty", 0) == 0
                   and c.get("judge_unparseable", 0) == 0 and not r.get("emitted_capped_steps")
                   and not r.get("compacted_steps") and (r.get("emitted_user_turns") or 0) > 0)'''),
    ('''    name = "pend.duplicates_left"
    rep._c(name)''',
     '''    n_kept = sum(1 for r in rows for s in (r.get("trace") or []) if (s.get("planner_diag") or {}).get("complete_kept_no_alternative"))
    name = "pend.complete_kept"
    rep._c(name)
    (rep.warn if n_kept else rep.note)(name, "Complete act kept while not ending (no alternative entry): %d" % n_kept)
    name = "pend.duplicates_left"
    rep._c(name)'''),
    ('''        check_rl(rl_dir, rollouts, ckpt_pattern, rep)
        check_rl_selection(rl_dir, splits, fold, rep)''',
     '''        check_rl(rl_dir, rollouts, ckpt_pattern, rep)
        check_rl_selection(rl_dir, splits, fold, rep)
        vp = os.path.join(rl_dir, "validation.jsonl")
        if os.path.exists(vp):
            # the episodes that drive best.json get the same structure / truncation / pend checks
            vrows = load_jsonl(vp, rep)
            if vrows:
                check_structure(vrows, arm, rep)
                check_truncation(vrows, arm, meta, sb, judge_budget, max_new_warn, rep)
                {"a2": check_a2, "pend": check_pend, "a0": check_a0}[check_family(arm)](vrows, rep, arm)'''),
])

# ---------------------------------------------------------------- eval CLIs (14)
CHECK = '''
def check_rl_settings(adapter, settings):
    """An RL adapter is evaluated only under the settings it was trained with, and only when no init adapter
    was merged into its base (this CLI loads the LoRA on the plain base)."""
    for d in (adapter, os.path.dirname(os.path.abspath(adapter.rstrip("/\\\\")))):
        p = os.path.join(d, "rl_manifest.json")
        if os.path.exists(p):
            m = json.load(open(p, encoding="utf-8"))
            if m.get("init_adapter"):
                raise SystemExit("adapter was trained on top of init adapter %s: evaluating it on the plain base is wrong" % m["init_adapter"])
            diff = {k: (m.get(k), v) for k, v in settings.items() if k in m and m.get(k) != v}
            if diff:
                raise SystemExit("adapter trained with other settings than this evaluation: %r" % diff)
            return m
    return None

'''
s = open("rollout_v4.py", encoding="utf-8").read()
s = s.replace("\n\ndef main():", "\n" + CHECK + "\ndef main():", 1)
open("rollout_v4.py", "w", encoding="utf-8", newline="\n").write(s)
patch("rollout_v4.py", [
    ('''        gate["planner_adapter"] = found''',
     '''        gate["planner_adapter"] = found
        check_rl_settings(args.planner_adapter, {"arm": args.arm, "implicit_profile": args.implicit_profile,
                                                 "fewshot": args.fewshot, "selector": args.selector})'''),
])
s = open("task1_v4.py", encoding="utf-8").read()
s = s.replace("\n\ndef main(argv=None):", "\n" + CHECK + "\ndef main(argv=None):", 1)
open("task1_v4.py", "w", encoding="utf-8", newline="\n").write(s)
patch("task1_v4.py", [
    ('''            if man is None:
                raise SystemExit("LEAK GATE: adapter has no manifest")''',
     '''            if man is None:
                raise SystemExit("LEAK GATE: adapter has no manifest")
            check_rl_settings(a.planner_adapter, {"arm": "pend", "implicit_profile": a.implicit_profile,
                                                  "selector": a.selector})'''),
])
print("ok")
