"""v16 in train_planner_rl.py: items 1 (Task 1 refill), 2 (task1 G / convs), 3 (std_norm at the stop-credit call),
4 (aux floor), 7 (continuous Task 1 metric: probes, selection, D2 trigger). See ops/SPEC_v16_grpo_opt.md."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---------------------------------------------------------------- fakes (dry run)
rep('''    def load_policy(self, d):
        self.theta = json.load(open(os.path.join(d, "fake_learner.json")))["theta"]
''', '''    def load_policy(self, d):
        self.theta = json.load(open(os.path.join(d, "fake_learner.json")))["theta"]

    def end_prob(self, x):
        return _sig(self.theta[x["prompt_ids"][0]])
''')
rep('''FakeEnv.task1_prompts = _fake_task1_prompts
FakeEnv.task1_sample = _fake_task1_sample
''', '''def _fake_task1_end_probe(self, conversation_id, t, user_prompt, real_final):
    k = 3 if real_final else 1
    return {"valid": True, "decision_valid": True, "greedy_end": _sig(self.learner.theta[k]) > 0.5,
            "prompt_ids": [k, t], "prefix_ids": [], "target_true": [1], "target_false": [0]}


def _fake_run_task1_prompts(self, conversation_id, seed=0, keep_prompts=False):
    out = FakeEnv._run_task1_plain(self, conversation_id, seed=seed)
    if keep_prompts:
        for x in out["turns"]:
            x["user_prompt"] = "fake"
    return out


FakeEnv.task1_prompts = _fake_task1_prompts
FakeEnv.task1_sample = _fake_task1_sample
FakeEnv.task1_end_probe = _fake_task1_end_probe
FakeEnv._run_task1_plain = FakeEnv.run_task1
FakeEnv.run_task1 = _fake_run_task1_prompts
''')

# ---------------------------------------------------------------- controller gets the aux floor (prompt text)
rep('''        self.controller = RC.make_controller(
            a.controller, cfg0, log_path=os.path.join(a.out, "llm_controller.jsonl"),
            transport=stub_llm_transport if (a.dry_run and a.controller == "llm") else None,
            **user_cfg.get(a.controller, {}))''', '''        ctl_opts = dict(user_cfg.get(a.controller, {}))
        if a.controller == "llm" and cfg0.get("version") == "v4":
            ctl_opts["aux_floor"] = float(a.stop_sup_floor)       # v16: the controller prompt names the aux floor
        self.controller = RC.make_controller(
            a.controller, cfg0, log_path=os.path.join(a.out, "llm_controller.jsonl"),
            transport=stub_llm_transport if (a.dry_run and a.controller == "llm") else None, **ctl_opts)''')

# ---------------------------------------------------------------- item 4: aux floor
rep('''    def aux_weight(self, u, cfg):
        """Effective stop-supervision weight = cfg["w_aux"] (initial --stop-sup-weight, then tuned by the LLM
        controller from TRAIN statistics) x the D2 anneal: 1 until validation Task 1 term_f1 first beats the
        untrained policy's, then linearly to 0 over --stop-sup-anneal updates."""
        w = float(cfg["w_aux"])
        if self.aux_anneal_start is None or self.a.stop_sup_anneal <= 0:
            return w
        return w * max(0.0, 1.0 - (u - self.aux_anneal_start) / float(self.a.stop_sup_anneal))''',
    '''    def aux_weight(self, u, cfg):
        """Effective stop-supervision weight = max(--stop-sup-floor, cfg["w_aux"] x the D2 anneal); cfg["w_aux"] is the
        initial --stop-sup-weight, then tuned by the LLM controller from TRAIN statistics; the anneal is 1 until the
        D2 trigger (validation Task 1 bal_p above the untrained policy's by --t1-trigger-margin at two consecutive
        validations, v16), then goes linearly to 0 over --stop-sup-anneal updates. v16 (user 2026-09-28): the
        supervision never goes below the floor."""
        w = float(cfg["w_aux"])
        if self.aux_anneal_start is not None and self.a.stop_sup_anneal > 0:
            w = w * max(0.0, 1.0 - (u - self.aux_anneal_start) / float(self.a.stop_sup_anneal))
        return max(float(self.a.stop_sup_floor), w)''')

# ---------------------------------------------------------------- items 1 + 2: Task 1 groups with refill
rep('''    def task1_rollouts(self, u):
        """Task 1 stop groups on REAL train conversations: for --task1-convs conversations (seeded by u,
        train split only), at the person's last message and at one earlier message (t >= 2), G sampled
        Planner decisions from the state the current policy reaches greedily; reward 1 when
        end_session agrees with the real person (the real last message is the only positive)."""''',
    '''    def task1_rollouts(self, u):
        """Task 1 stop groups on REAL train conversations: for --task1-convs conversations (seeded by u,
        train split only), at the person's last message and at one earlier message (t >= 2), --task1-G sampled
        Planner decisions from the state the current policy reaches greedily; reward 1 when
        end_session agrees with the real person (the real last message is the only positive).
        v16 item 1 (dynamic sampling): a group whose rewards all agree has no gradient; while the informative groups
        are fewer than the base groups, further unused train_all conversations are drawn from the same rng (rounds of
        ceil(deficit / 2) conversations, at most --task1-convs extra conversations in total), rows marked refill."""''')
rep('''        rng = random.Random(seed_of(a.seed, "task1", u))
        pool = sorted(self.split["train_all"])     # Task 1 needs no requirement shards: every train session
        cids = rng.sample(pool, min(a.task1_convs, len(pool)))
        jobs = []
        for cid in cids:
            assert cid in self.split["train_all"] and cid not in self.split["forbidden"], "non-train conversation %r in a Task 1 group" % cid
            jobs.append((cid, rng.random()))

        def run_conv(job):
            cid, x = job''', '''        rng = random.Random(seed_of(a.seed, "task1", u))
        pool = sorted(self.split["train_all"])     # Task 1 needs no requirement shards: every train session
        cids = rng.sample(pool, min(a.task1_convs, len(pool)))
        jobs = []
        for cid in cids:
            assert cid in self.split["train_all"] and cid not in self.split["forbidden"], "non-train conversation %r in a Task 1 group" % cid
            jobs.append((cid, rng.random()))

        def run_conv(job, refill=False):
            cid, x = job''')
rep('''            rows = [reuse[(cid, t)] for t in pos if (cid, t) in reuse]
            missing = [t for t in pos if (cid, t) not in reuse]''', '''            rows = [reuse[(cid, t)] for t in pos if (cid, t) in reuse]
            for r_ in rows:
                assert bool(r_.get("refill", False)) == refill, "reused Task 1 row %s t%d changed its refill role" % (cid, r_["t"])
            missing = [t for t in pos if (cid, t) not in reuse]''')
rep('''                smp = self.env.task1_sample(cid, t, pr["user_prompt"], pr["real_final"], a.G,
                                            a.temperature, a.top_p, seed_of(a.seed, "t1s", u))
                row = {"update": u, "conversation_id": cid, "t": t, "n_real": n, "real_final": t == n,
                       "split": "train", "policy_version": pv, "policy_sha": psha, "samples": smp, "time": time.time()}''',
    '''                smp = self.env.task1_sample(cid, t, pr["user_prompt"], pr["real_final"], a.task1_G,
                                            a.temperature, a.top_p, seed_of(a.seed, "t1s", u))
                row = {"update": u, "conversation_id": cid, "t": t, "n_real": n, "real_final": t == n,
                       "split": "train", "policy_version": pv, "policy_sha": psha, "samples": smp, "refill": refill,
                       "time": time.time()}''')
rep('''        out = []
        if a.rollout_workers <= 1:
            for job in jobs:
                out += run_conv(job)
        else:
            with ThreadPoolExecutor(max_workers=a.rollout_workers) as ex:
                for rows in ex.map(run_conv, jobs):
                    out += rows
        return sorted(out, key=lambda r: (r["conversation_id"], r["t"]))''',
    '''        def run_many(js, refill):
            res = []
            if a.rollout_workers <= 1:
                for job in js:
                    res += run_conv(job, refill)
            else:
                with ThreadPoolExecutor(max_workers=a.rollout_workers) as ex:
                    for rows in ex.map(lambda j: run_conv(j, refill), js):
                        res += rows
            return res

        def informative(r):
            rs = [x["reward"] for x in r["samples"]]
            return len(rs) >= 2 and RA.pstd(rs) > self.acfg["min_group_std"]

        out = run_many(jobs, False)
        n_base = len(out)
        rest = [c for c in pool if c not in set(cids)]
        rng.shuffle(rest)                             # the refill order: after the base draws, same rng
        n_refill_convs, stop = 0, "none_needed"
        while True:
            deficit = n_base - sum(1 for r in out if informative(r))
            if deficit <= 0:
                stop = "filled" if n_refill_convs else "none_needed"
                break
            if not rest:
                stop = "pool_empty"
                break
            if n_refill_convs >= a.task1_convs:
                stop = "cap"
                break
            k = max(1, min(len(rest), a.task1_convs - n_refill_convs, -(-deficit // 2)))
            take, rest = rest[:k], rest[k:]
            js = []
            for cid in take:
                assert cid in self.split["train_all"] and cid not in self.split["forbidden"], "non-train conversation %r in a Task 1 refill" % cid
                js.append((cid, rng.random()))
            out += run_many(js, True)
            n_refill_convs += k
        self.t1_refill = {"n_base_groups": n_base, "n_refill_groups": len(out) - n_base, "n_refill_convs": n_refill_convs,
                          "n_informative_groups": sum(1 for r in out if informative(r)), "refill_stop": stop}
        return sorted(out, key=lambda r: (bool(r.get("refill", False)), r["conversation_id"], r["t"]))''')

# ---------------------------------------------------------------- item 3: std_norm at the stop-credit call
rep('''                a1, a2 = RA.split_group_advantages(rs, sp, self.acfg["adv_eps"], self.acfg["min_group_std"])''',
    '''                a1, a2 = RA.split_group_advantages(rs, sp, self.acfg["adv_eps"], self.acfg["min_group_std"],
                                                   self.acfg["grpo_std_norm"])''')

# ---------------------------------------------------------------- Task 1 train statistics (base groups) + refill record
rep('''        t1_all = [x for r in t1rows for x in r["samples"]]
        t1_hist = None
        if t1_all:
            fin = [x for x in t1_all if x["real_final"]]
            mid = [x for x in t1_all if not x["real_final"]]
            t1_hist = {"n": len(t1_all), "acc": sum(x["reward"] for x in t1_all) / len(t1_all),
                       "end_at_final": (sum(x["ended_planner"] for x in fin) / len(fin)) if fin else None,
                       "end_at_nonfinal": (sum(x["ended_planner"] for x in mid) / len(mid)) if mid else None,
                       "n_not_decisions": sum(1 for x in t1_all if not x.get("decision_valid", True)),
                       "n_skipped_capped_history": self.t1_skipped_capped,
                       "groups_skipped_zero_std": t1_skipped}''',
    '''        t1_all = [x for r in t1rows for x in r["samples"]]
        t1_base = [x for r in t1rows if not r.get("refill") for x in r["samples"]]     # comparable across updates
        t1_hist = None
        if t1_all:
            fin = [x for x in t1_base if x["real_final"]]
            mid = [x for x in t1_base if not x["real_final"]]
            t1_hist = {"n": len(t1_base), "acc": (sum(x["reward"] for x in t1_base) / len(t1_base)) if t1_base else None,
                       "acc_all": sum(x["reward"] for x in t1_all) / len(t1_all), "n_all": len(t1_all),
                       "end_at_final": (sum(x["ended_planner"] for x in fin) / len(fin)) if fin else None,
                       "end_at_nonfinal": (sum(x["ended_planner"] for x in mid) / len(mid)) if mid else None,
                       "n_not_decisions": sum(1 for x in t1_all if not x.get("decision_valid", True)),
                       "n_skipped_capped_history": self.t1_skipped_capped,
                       "groups_skipped_zero_std": t1_skipped, **getattr(self, "t1_refill", {})}''')
# aux only from the base groups
rep('''        if w_aux > 0:
            for r in t1rows:
                x = (r["samples"] or [{}])[0].get("aux")''', '''        if w_aux > 0:
            for r in t1rows:
                if r.get("refill"):
                    continue                 # v16: the supervision amount stays fixed (base groups only)
                x = (r["samples"] or [{}])[0].get("aux")''')
# advantage magnitude (item 3 monitoring) into the learner stats
rep('''        timing["learner_s"] = round(time.time() - _t, 1)''', '''        timing["learner_s"] = round(time.time() - _t, 1)
        stats["adv_abs_mean"] = (sum(abs(float(s_.get("adv") or 0.0)) + abs(float(s_.get("adv_stop") or 0.0)) for s_ in samples)
                                 / len(samples)) if samples else None''')
rep('''                "aux_weight": w_aux, "lr": cfg["lr"], "kl_coef": cfg["kl_coef"],''',
    '''                "aux_weight": w_aux, "aux_floor": float(self.a.stop_sup_floor),
                "aux_floor_active": bool(w_aux > 0 and abs(w_aux - float(self.a.stop_sup_floor)) < 1e-12
                                         and float(cfg["w_aux"]) != w_aux),
                "lr": cfg["lr"], "kl_coef": cfg["kl_coef"],''')

# ---------------------------------------------------------------- item 7: validation Task 1 end probabilities
rep('''        def run_t1(cid):
            r = done_t1.get(cid)
            if r is None or r["policy_sha"] != psha:
                r = {"kind": "task1", "update": u, "policy_sha": psha, "conversation_id": cid,
                     "split": "validation", "task1": self.env.run_task1(cid), "time": time.time()}
                with self.io_lock:
                    append_jsonl(p_val, r)
            return r''', '''        def run_t1(cid):
            r = done_t1.get(cid)
            if r is None or r["policy_sha"] != psha or "end_probs" not in r:
                t1r = self.env.run_task1(cid, keep_prompts=True)
                probs = []
                for x in t1r["turns"]:
                    if x["t"] < 2:
                        continue               # turn 1 cannot end: not a decision point
                    pr = self.env.task1_end_probe(cid, x["t"], x["user_prompt"], x["real_final"])
                    pe = float(self.learner.end_prob(pr)) if pr["valid"] else 0.0
                    probs.append({"t": x["t"], "real_final": bool(x["real_final"]), "p_end": pe, "valid": bool(pr["valid"]),
                                  "greedy_end": bool(pr["greedy_end"])})
                for x in t1r["turns"]:
                    x.pop("user_prompt", None)
                r = {"kind": "task1", "update": u, "policy_sha": psha, "conversation_id": cid,
                     "split": "validation", "task1": t1r, "end_probs": probs, "time": time.time()}
                with self.io_lock:
                    append_jsonl(p_val, r)
            return r''')
rep('''        t1 = T1.task1_stop_metrics([r["task1"] for r in t1rows]) if t1rows else None
        if not reselect and t1 is not None and self.task1_base is None:''',
    '''        t1 = T1.task1_stop_metrics([r["task1"] for r in t1rows]) if t1rows else None
        if t1 is not None:
            t1.update(T1.task1_prob_metrics([p_ for r in t1rows for p_ in r["end_probs"]]))
        if not reselect and t1 is not None and self.task1_base is None:''')
rep('''        if not reselect and t1 is not None and self.aux_anneal_start is None and u > self.task1_base["update"] \\
                and t1["term_f1"] > self.task1_base["term_f1"]:
            self.aux_anneal_start = u            # D2: from here the stop supervision goes linearly to 0''',
    '''        # D2 (v16): the continuous Task 1 metric bal_p at least --t1-trigger-margin above the untrained policy's at
        # TWO consecutive validations (after the base); one noisy validation (4 conversations) cannot trigger it
        d2 = None
        if t1 is not None and self.task1_base is not None and self.task1_base.get("bal_p") is not None \\
                and u > self.task1_base["update"]:
            met = t1["bal_p"] >= self.task1_base["bal_p"] + a.t1_trigger_margin
            streak = 0
            for v_ in sorted((r for r in prev if r.get("kind") == "summary" and r["update"] < u
                              and r["update"] > self.task1_base["update"]), key=lambda r: r["update"]):
                streak = streak + 1 if (v_.get("d2") or {}).get("met") else 0
            streak = streak + 1 if met else 0
            d2 = {"met": met, "streak": streak, "margin": a.t1_trigger_margin, "base_bal_p": self.task1_base["bal_p"],
                  "bal_p": t1["bal_p"], "triggered_at": self.aux_anneal_start}
        if not reselect and d2 is not None and self.aux_anneal_start is None and d2["streak"] >= 2:
            d2["triggered_at"] = u
            self.aux_anneal_start = u            # D2: from here the stop supervision goes linearly to its floor''')
rep('''        sel = None
        if not withheld and turn_stats is not None and turn_stats.get("turn_w1") is not None:
            sel = (a.w_sel_cov * turn_stats["coverage_mean"] - a.w_sel_w1 * turn_stats["turn_w1"]
                   + (a.w_sel_task1 * t1["term_f1"] if t1 is not None else 0.0))''',
    '''        # v16 (user 2026-09-28): Task 1 enters the selection through the continuous bal_p (20 decision points rather
        # than 4 end events); term_f1 stays the reported metric
        sel = None
        if not withheld and turn_stats is not None and turn_stats.get("turn_w1") is not None:
            sel = (a.w_sel_cov * turn_stats["coverage_mean"] - a.w_sel_w1 * turn_stats["turn_w1"]
                   + (a.w_sel_task1 * t1["bal_p"] if t1 is not None else 0.0))''')
rep('''                                  "selection_score": sel, "w_sel_task1": a.w_sel_task1,
                                  "w_sel_w1": a.w_sel_w1, "w_sel_cov": a.w_sel_cov,
                                  "selection_formula": "w_sel_cov*coverage_mean - w_sel_w1*turn_w1 + w_sel_task1*task1.term_f1",''',
    '''                                  "selection_score": sel, "w_sel_task1": a.w_sel_task1,
                                  "w_sel_w1": a.w_sel_w1, "w_sel_cov": a.w_sel_cov, "selection_task1_metric": "bal_p",
                                  "selection_formula": "w_sel_cov*coverage_mean - w_sel_w1*turn_w1 + w_sel_task1*task1.bal_p",
                                  "d2": d2, "aux_floor": float(a.stop_sup_floor),''')

# ---------------------------------------------------------------- arguments / spec gate
rep('''    ap.add_argument("--task1-convs", type=int, default=4,
                    help="Task 1 stop groups per update: real TRAIN conversations (last + one earlier message, G samples each); 0 = off")''',
    '''    ap.add_argument("--task1-convs", type=int, default=8,
                    help="Task 1 stop groups per update: real TRAIN conversations (last + one earlier message, --task1-G "
                         "samples each); 0 = off. v16 spec: 8")
    ap.add_argument("--task1-G", type=int, default=8, help="samples per Task 1 stop group (v16 spec: 8; Task 2 keeps --G)")
    ap.add_argument("--t1-trigger-margin", type=float, default=0.10,
                    help="D2 (v16): validation Task 1 bal_p must exceed the untrained policy's by this at two consecutive "
                         "validations before the stop supervision anneals (spec 0.10)")
    ap.add_argument("--stop-sup-floor", type=float, default=None,
                    help="v16: the stop supervision never goes below this weight (value pending the user's decision; "
                         "required except with --dry-run, where it defaults to 0)")''')
rep('''    ap.add_argument("--w-sel-task1", type=float, default=1.0,
                    help="checkpoint selection: weight of validation Task 1 term_f1 (M2)")''',
    '''    ap.add_argument("--w-sel-task1", type=float, default=1.0,
                    help="checkpoint selection: weight of the validation Task 1 bal_p (v16; term_f1 is reported)")''')
rep('''    if a.task1_convs <= 0:
        off["task1_convs"] = a.task1_convs''', '''    if a.task1_convs != 8:
        off["task1_convs"] = a.task1_convs
    if a.task1_G != 8:
        off["task1_G"] = a.task1_G
    if a.task1_G < 2:
        ap.error("--task1-G must be >= 2 (group baselines)")
    if a.t1_trigger_margin != 0.10:
        off["t1_trigger_margin"] = a.t1_trigger_margin
    if a.stop_sup_floor is None:
        if not a.dry_run:
            ap.error("--stop-sup-floor is required: its value is pending the user's decision (SPEC v16 item 4)")
        a.stop_sup_floor = 0.0
    if not (0.0 <= a.stop_sup_floor <= 5.0):
        ap.error("--stop-sup-floor must lie in [0, 5]")
    if a.stop_sup_weight == 0 and a.stop_sup_floor != 0:
        ap.error("--stop-sup-weight 0 (no stop supervision) needs --stop-sup-floor 0")''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
