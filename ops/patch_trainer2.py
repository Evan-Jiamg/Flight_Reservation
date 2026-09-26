p = "train_planner_rl.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:80])
    s = s.replace(old, new)


a = s.index("    def one_update(self, u):")
b = s.index("    # -------------------------------------------------------- validation")
ONE_UPDATE = '''    def one_update(self, u):
        t0 = time.time()
        cfg = copy.deepcopy(self.cfg)
        groups_all = self.rollouts(u)
        pv, psha = u - 1, self.learner.policy_sha()
        for grp in groups_all:
            for row in grp:
                assert row["policy_version"] == pv and row["policy_sha"] == psha, "off-policy rollout"
        # an episode with a cut R0 reply, a lost ledger verdict or an emitted capped message never enters a
        # reward group (its reward would be wrong); a group left with < 2 episodes has no baseline
        groups = [[row for row in grp if row["episode"]["clean"]] for grp in groups_all]
        n_unclean = sum(len(g0) - len(g1) for g0, g1 in zip(groups_all, groups))
        groups = [g for g in groups if len(g) >= 2]
        clean_eps = [row["episode"] for grp in groups for row in grp]
        if not clean_eps:
            raise SystemExit("update %d: no clean episode (R0 / ledger judge failing?) -- stopping, not training on it" % u)
        ctx = self.reward_ctx(clean_eps, cfg)
        shadow_ctx = self.reward_ctx(clean_eps, self.selection_cfg)
        samples, rewarded, rew_groups, stop_groups, shadow = [], [], [], [], []
        for grp in groups:
            rs, sp = [], []
            for row in grp:
                rw = RR.reward(row["episode"], cfg, ctx)
                rs.append(rw["total"])
                sp.append(float(rw["components"].get("stop_part", 0.0)))
                rewarded.append((row["episode"], rw))
                # fixed-weight shadow reward (selection_cfg): comparable across controller changes
                shadow.append(RR.reward(row["episode"], self.selection_cfg, shadow_ctx)["total"])
            rew_groups.append(rs)
            stop_groups.append(sp)
        stop_credit = self.a.stop_credit and cfg["version"] in ("v3", "v4")
        if stop_credit:
            # stop credit assignment: the length term (v3 |T - target|, v4 log p_h(T) - log q(T)) is caused
            # only by the end_session decisions, so its group advantage goes to the end_session value tokens
            # of every decision step; coverage and the format constraints keep the sequence-level advantage
            seq_groups = [[R - S for R, S in zip(rs, sp)] for rs, sp in zip(rew_groups, stop_groups)]
            advs, _ = RA.advantages_for_groups(seq_groups, self.a.algo, self.acfg)
            advs_stop, _ = RA.advantages_for_groups(stop_groups, self.a.algo, self.acfg)
            skipped = sum(1 for a1, a2 in zip(advs, advs_stop) if a1 is None and a2 is None)
        else:
            advs, skipped = RA.advantages_for_groups(rew_groups, self.a.algo, self.acfg)
            advs_stop = [None] * len(groups)
        for grp, rs, ad, ads in zip(groups, rew_groups, advs, advs_stop):
            if ad is None and ads is None:
                continue
            for j, (row, R) in enumerate(zip(grp, rs)):
                for s_ in RA.episode_samples(row["episode"], policy_version=row["policy_version"]):
                    s_["ret"], s_["adv"] = R, (ad[j] if ad is not None else 0.0)
                    s_["adv_stop"] = ads[j] if ads is not None else 0.0
                    samples.append(s_)
        # Task 1 stop groups (same policy version; one Planner step per sample). A sample that is not a
        # decision (unparsed / capped / no valid end_session) has reward 0 and no stop mask.
        t1rows = self.task1_rollouts(u)
        t1_rewards = [[x["reward"] for x in r["samples"]] for r in t1rows]
        t1_advs, t1_skipped = RA.advantages_for_groups(t1_rewards, self.a.algo, self.acfg) if t1rows else ([], 0)
        for r, rs, ad in zip(t1rows, t1_rewards, t1_advs):
            assert r["policy_version"] == pv and r["policy_sha"] == psha, "off-policy task1 rollout"
            assert r["t"] >= 2, "Task 1 stop group at turn 1"
            if ad is None:
                continue
            for x, R, A in zip(r["samples"], rs, ad):
                pseudo = {"conversation_id": r["conversation_id"], "replicate": x["replicate"],
                          "trace": [{"t": x["t"], "planner_gen": x["planner_gen"]}]}
                for s_ in RA.episode_samples(pseudo, policy_version=pv):
                    if self.a.stop_credit:
                        s_["ret"], s_["adv"], s_["adv_stop"], s_["source"] = R, 0.0, A, "task1"
                    else:
                        s_["ret"], s_["adv"], s_["source"] = R, A, "task1"
                    samples.append(s_)
        t1_all = [x for r in t1rows for x in r["samples"]]
        t1_hist = None
        if t1_all:
            fin = [x for x in t1_all if x["real_final"]]
            mid = [x for x in t1_all if not x["real_final"]]
            t1_hist = {"n": len(t1_all), "acc": sum(x["reward"] for x in t1_all) / len(t1_all),
                       "end_at_final": (sum(x["ended_planner"] for x in fin) / len(fin)) if fin else None,
                       "end_at_nonfinal": (sum(x["ended_planner"] for x in mid) / len(mid)) if mid else None,
                       "n_not_decisions": sum(1 for x in t1_all if not x.get("decision_valid", True)),
                       "groups_skipped_zero_std": t1_skipped}
        assert all(s_["policy_version"] == pv for s_ in samples), "sample from another policy version"
        aux, w_aux = [], self.aux_weight(u)
        if w_aux > 0:
            for r in t1rows:
                x = (r["samples"] or [{}])[0].get("aux")
                if x is not None:
                    assert x["want_end"] == r["real_final"], "stop-supervision label disagrees with the human"
                    aux.append(dict(x, weight=w_aux))
        stats = self.learner.update(samples, cfg, seed=seed_of(self.a.seed, "update", u), aux=aux or None)
        agg = RR.aggregate(rewarded)
        tm = int(cfg["t_max"])
        turn_hist = [0] * (tm + 1)
        for e in clean_eps:
            turn_hist[min(max(int(e["emitted_user_turns"]), 0), tm)] += 1
        hist = {"update": u, "split": "train", "reward_version": cfg["version"], "task1_train": t1_hist, **agg,
                "n_groups": len(groups), "n_groups_skipped_zero_std": skipped, "n_unclean_episodes": n_unclean,
                "shadow_reward_mean": sum(shadow) / len(shadow), "turn_hist": turn_hist, "p_h": self.p_h,
                "aux_weight": w_aux, "lr": cfg["lr"], "kl_coef": cfg["kl_coef"],
                **{k: stats.get(k) for k in ("loss", "kl", "ratio_mean", "clip_frac", "grad_norm", "n_tokens",
                                              "value_mse", "ratio_init_maxdev")}}
        self.history.append(hist)
        self.cfg = self.controller.propose(self.history)
        self.update_done = u
        row = {"update": u, "policy_version_rollouts": pv, "policy_version_after": u, "algo": self.a.algo,
               "controller": self.a.controller, "cfg_used": cfg, "cfg_used_sha256": RR.cfg_sha(cfg),
               "reward_ctx": ctx, "next_cfg": self.cfg, "scenarios": [g[0]["conversation_id"] for g in groups_all],
               "train_aggregate": hist, "learner_stats": stats, "n_samples": len(samples),
               "policy_sha_after": self.learner.policy_sha(), "time": time.time(), "update_s": time.time() - t0}
        self.save_checkpoint(u, row)
        append_jsonl(self.p_upd, row)
        return row

'''
s = s[:a] + ONE_UPDATE + s[b:]

# ---- validation: sampled Planner (D5), clean episodes only, v4 ctx, W1, D2 anneal trigger
rep('''        """Greedy episodes on the validation ids; logged only to validation.jsonl; drives best.json."""''',
'''        """Task 2 episodes on the validation ids with the SAMPLED Planner (D5: --val-temperature, one
        replicate per --val-seeds entry) + Task 1 (greedy, as the benchmark); logged only to validation.jsonl;
        drives best.json and the D2 annealing trigger. Never enters history or the controller."""''')
rep('''        for cid in sorted(self.split["validation"]):
            assert cid in self.split["validation"] and cid not in self.split["train"]
            for s in a.val_seeds:
                jobs.append((cid, s))''',
'''        for cid in sorted(self.split["validation"]):
            assert cid in self.split["validation"] and cid not in self.split["train"] \\
                and cid not in self.split["train_all"], "validation id %r is also a training id" % cid
            for s in a.val_seeds:
                jobs.append((cid, s))''')
rep('''                ep = self.env.run_episode(cid, seed=s, replicate=0, planner_temperature=0.0, planner_top_p=1.0,
                                          record_generation=False)
                rw = RR.reward(ep, self.selection_cfg)
                r = {"kind": "episode", "update": u, "policy_sha": psha, "conversation_id": cid, "seed": s,
                     "split": "validation", "reward_selection": rw, "episode": ep, "time": time.time()}''',
'''                ep = self.env.run_episode(cid, seed=s, replicate=s, planner_temperature=a.val_temperature,
                                          planner_top_p=a.val_top_p, record_generation=False)
                r = {"kind": "episode", "update": u, "policy_sha": psha, "conversation_id": cid, "seed": s,
                     "split": "validation", "episode": ep, "time": time.time()}''')
rep('''        totals = [r["reward_selection"]["total"] for r in vrows]
        score = sum(totals) / len(totals) if totals else float("nan")
        eps = [r["episode"] for r in vrows]''',
'''        eps = [r["episode"] for r in vrows if r["episode"]["clean"]]
        n_unclean = len(vrows) - len(eps)
        vctx = self.reward_ctx(eps, self.selection_cfg) if eps else None
        totals = [RR.reward(e, self.selection_cfg, vctx)["total"] for e in eps]
        score = sum(totals) / len(totals) if totals else float("nan")''')
rep('''                          "coverage_mean": sum(float(e["coverage"]) for e in eps) / len(eps),''',
'''                          "coverage_mean": sum(float(e["coverage"]) for e in eps) / len(eps),
                          "turn_w1": turn_w1([e["emitted_user_turns"] for e in eps], [e["human_turns"] for e in eps]),''')
rep('''        if t1 is not None and self.task1_base is None:
            self.task1_base = {"update": u, **t1}''',
'''        if t1 is not None and self.task1_base is None:
            self.task1_base = {"update": u, **t1}
        if t1 is not None and self.aux_anneal_start is None and u > self.task1_base["update"] \\
                and t1["term_f1"] > self.task1_base["term_f1"]:
            self.aux_anneal_start = u            # D2: from here the stop supervision goes linearly to 0
            sp = os.path.join(self.ckpt_dir(self.update_done), "state.json")
            st = json.load(open(sp))
            st["aux_anneal_start"] = u
            write_json_atomic(sp, st)''')
rep('''                                  "n_episodes": len(totals), "mean_reward_selection": score,''',
'''                                  "n_episodes": len(totals), "n_unclean_episodes": n_unclean,
                                  "val_temperature": a.val_temperature, "val_seeds": a.val_seeds,
                                  "mean_reward_selection": score, "aux_anneal_start": self.aux_anneal_start,''')
rep('''def parse_args(argv=None):''',
'''def turn_w1(sim, human):
    """Wasserstein-1 between two samples of integer conversation lengths (sum of |CDF difference|)."""
    if not sim or not human:
        return None
    hi = max(max(sim), max(human))
    w, cs, ch = 0.0, 0.0, 0.0
    for k in range(0, hi + 1):
        cs += sum(1 for x in sim if x == k) / len(sim)
        ch += sum(1 for x in human if x == k) / len(human)
        w += abs(cs - ch)
    return w


SPEC = {"implicit_profile": 1, "fewshot": "fold", "selector": "borda"}      # the pend design (user, 2026-09-25)


def parse_args(argv=None):''')

# ---- arguments
rep('''    ap.add_argument("--arm", choices=("pend", "final", "a2"), default="pend",
                    help="policy env: pend = E1.6 base + fixes, Planner judges the goal and ends (no goal judge)")''',
'''    ap.add_argument("--arm", choices=("pend",), default="pend",
                    help="policy env: pend = E1.6 base + fixes, Planner judges the goal and ends (no goal judge)")
    ap.add_argument("--ablation", default=None,
                    help="name of a declared ablation; required to run with any setting that differs from the pend spec")''')
rep('''    ap.add_argument("--implicit-profile", type=int, choices=(0, 1), default=0)
    ap.add_argument("--fewshot", choices=("off", "fold"), default="off", help="fold: examples from splits[fold].train_all only")
    ap.add_argument("--selector", choices=("length", "borda"), default="length")''',
'''    ap.add_argument("--implicit-profile", type=int, choices=(0, 1), default=SPEC["implicit_profile"])
    ap.add_argument("--fewshot", choices=("off", "fold"), default=SPEC["fewshot"], help="fold: examples from splits[fold].train_all only")
    ap.add_argument("--selector", choices=("length", "borda"), default=SPEC["selector"])''')
rep('''    ap.add_argument("--stop-credit", type=int, choices=(0, 1), default=1,''',
'''    ap.add_argument("--stop-sup-anneal", type=int, default=10,
                    help="D2: updates over which the stop supervision goes linearly to 0 once validation Task 1 "
                         "term_f1 beats the untrained policy's; 0 = never anneal")
    ap.add_argument("--stop-credit", type=int, choices=(0, 1), default=1,''')
rep('''    ap.add_argument("--val-seeds", type=int, nargs="+", default=[0])''',
'''    ap.add_argument("--val-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--val-temperature", type=float, default=0.7, help="D5: validation Task 2 uses the sampled Planner")
    ap.add_argument("--val-top-p", type=float, default=1.0)''')
rep('''    if a.stop_credit and a.algo != "grpo":
        ap.error("--stop-credit 1 is implemented for grpo")''',
'''    if a.stop_credit and a.algo != "grpo":
        ap.error("--stop-credit 1 is implemented for grpo")
    off = {k: getattr(a, k) for k in SPEC if getattr(a, k) != SPEC[k]}
    if off and not a.ablation:
        ap.error("settings %r differ from the pend spec %r; name the ablation with --ablation" % (off, SPEC))
    if not (a.val_temperature > 0):
        ap.error("--val-temperature must be > 0 (D5: sampled Planner at validation)")''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
