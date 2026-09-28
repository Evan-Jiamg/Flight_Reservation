p = "train_planner_rl.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:80])
    s = s.replace(old, new)


# ---- FakeEnv: clean flag + human_turns method
rep('''                "end_kind": end_kind, "coverage": level / 2.0, "complete": level == 2, "trace": trace,
                "human_turns": 2 + seed_of("human", conversation_id) % 6}''',
'''                "end_kind": end_kind, "coverage": level / 2.0, "complete": level == 2, "trace": trace,
                "clean": True, "episode_counters": {}, "human_turns": self.human_turns(conversation_id)}

    def human_turns(self, conversation_id):
        return 2 + seed_of("human", conversation_id) % 6''')

# ---- stub transport understands the v4 controller request
rep('''    cur = json.loads(request["messages"][1]["content"])["current"]
    prop = {k: v * 1.5 if v else 0.01 for k, v in cur.items()}
    prop["rationale"] = "stub"''',
'''    cur = json.loads(request["messages"][1]["content"])["current"]
    if "factors" in request["messages"][0]["content"]:            # v4 factor controller
        prop = {"factors": {k: 1.25 for k in cur}, "rationale": "stub"}
    else:
        prop = {k: v * 1.5 if v else 0.01 for k, v in cur.items()}
        prop["rationale"] = "stub"''')

# ---- reward defaults for pend: v4 (D1(b))
rep('''        # pend: reward v3 with format constraints on (declared defaults; a --config file may override)
        base_reward = {"version": "v3", "lambda_unparsed": 1.0, "lambda_hit_max_new": 1.0} if a.arm == "pend" else {}''',
'''        # pend: reward v4 (D1(b): human length DISTRIBUTION matching) with format constraints on
        # (declared defaults; a --config file may override)
        base_reward = {"version": "v4", "lambda_unparsed": 1.0, "lambda_hit_max_new": 1.0} if a.arm == "pend" else {}''')
rep('''        self.task1_base = None       # Task 1 stop metrics of the starting policy (update 0 validation)''',
'''        self.task1_base = None       # Task 1 stop metrics of the starting policy (update 0 validation)
        self.aux_anneal_start = None # D2: first update at which validation Task 1 term_f1 beat task1_base
        self.p_h = None              # v4: smoothed human length distribution of the TRAIN conversations''')

# ---- build: t_max check, p_h
rep('''        if a.dry_run:
            self.learner = FakeLearner(a.algo, self.acfg, self.cfg["lr"], seed=a.seed)
            self.env = FakeEnv(self.learner)
            return''',
'''        if a.dry_run:
            self.learner = FakeLearner(a.algo, self.acfg, self.cfg["lr"], seed=a.seed)
            self.env = FakeEnv(self.learner)
            self.set_p_h()
            return''')
rep('''        self.learner = RA.TorchLearner(model, a.algo, self.acfg, lr=self.cfg["lr"], seed=a.seed)
        self.config_record["env"] = self.env.describe()''',
'''        import task2_env as TE
        assert int(self.cfg["t_max"]) == TE.T_MAX, "reward t_max %r != environment T_MAX %d" % (self.cfg["t_max"], TE.T_MAX)
        self.set_p_h()
        self.learner = RA.TorchLearner(model, a.algo, self.acfg, lr=self.cfg["lr"], seed=a.seed)
        self.config_record["env"] = self.env.describe()''')
rep('''    def meta(self, kind):''',
'''    def set_p_h(self):
        """v4: p_h from the real people's number of messages in splits[fold].train_all ONLY (Task 2 length
        target; train_all has no requirement shards but its lengths are training data like any other)."""
        ids = sorted(self.split["train_all"])
        for cid in ids:
            assert cid not in self.split["forbidden"], "p_h would read a validation/test conversation"
        turns = [self.env.human_turns(cid) for cid in ids]
        self.p_h = RR.turn_distribution(turns, self.cfg["t_max"], self.selection_cfg["alpha_smooth"])
        self.config_record["p_h"] = {"source": "train_all", "n_conversations": len(ids), "dist": self.p_h,
                                     "alpha_smooth": self.selection_cfg["alpha_smooth"]}

    def reward_ctx(self, episodes, cfg):
        """v4 context: p_h (train, fixed) and q = smoothed length distribution of THESE episodes."""
        if cfg["version"] != "v4":
            return None
        q = RR.turn_distribution([e["emitted_user_turns"] for e in episodes], cfg["t_max"], cfg["alpha_smooth"])
        return {"p_h": self.p_h, "q": q}

    def aux_weight(self, u):
        """D2: full weight until validation Task 1 term_f1 first beats the untrained policy's, then linearly
        to 0 over --stop-sup-anneal updates."""
        w = self.a.stop_sup_weight
        if self.aux_anneal_start is None or self.a.stop_sup_anneal <= 0:
            return w
        return w * max(0.0, 1.0 - (u - self.aux_anneal_start) / float(self.a.stop_sup_anneal))

    def meta(self, kind):''')

# ---- provenance: every argument except the ones that may legitimately change on resume
rep('''        for k in ("algo", "controller", "fold", "G", "scenarios_per_update", "seed", "temperature", "top_p"):
            if first["config"]["args"][k] != row["config"]["args"][k]:''',
'''        for k in sorted(set(first["config"]["args"]) | set(row["config"]["args"])):
            if k in RESUME_MAY_CHANGE:
                continue
            if first["config"]["args"].get(k) != row["config"]["args"].get(k):''')
rep('''                raise SystemExit("argument %s changed on resume (%r -> %r)" % (k, first["config"]["args"][k],
                                                                             row["config"]["args"][k]))''',
'''                raise SystemExit("argument %s changed on resume (%r -> %r)" % (k, first["config"]["args"].get(k),
                                                                             row["config"]["args"].get(k)))''')
rep('''TIME_KEYS = ("time", "wall_s", "rollout_s", "update_s")''',
'''TIME_KEYS = ("time", "wall_s", "rollout_s", "update_s")
RESUME_MAY_CHANGE = ("resume", "allow_code_change", "updates", "rollout_workers", "gpu", "max_batch",
                     "keep_optimizer_last", "dry_run_crash_after_episodes")''')

# ---- checkpoint: manifest + anneal state
rep('''                 "best": self.best, "update_row": update_row, "task1_base": self.task1_base,''',
'''                 "best": self.best, "update_row": update_row, "task1_base": self.task1_base,
                 "aux_anneal_start": self.aux_anneal_start,''')
rep('''        write_json_atomic(os.path.join(tmp, "state.json"), state)
        if os.path.exists(d):''',
'''        write_json_atomic(os.path.join(tmp, "state.json"), state)
        write_json_atomic(os.path.join(tmp, "rl_manifest.json"), self.manifest())
        if os.path.exists(d):''')
rep('''    def load_checkpoint(self):''',
'''    def manifest(self):
        """What this policy was trained on (checked by the evaluation CLIs before any validation/test run)."""
        a = self.a
        return {"kind": "planner_rl", "fold": self.split["fold"], "splits_sha256": self.split["sha256"],
                "train_scenarios": sorted(self.split["train"]), "train_conversations": sorted(self.split["train_all"]),
                "fewshot_pool": sorted(self.split["train_all"]) if a.fewshot == "fold" else [],
                "p_h_source": "train_all", "validation_used_for": "checkpoint selection only",
                "planner_path": a.planner_path, "init_adapter": a.init_adapter,
                "init_adapter_sha256": sha_path(a.init_adapter) if a.init_adapter else None,
                "arm": a.arm, "implicit_profile": a.implicit_profile, "fewshot": a.fewshot, "selector": a.selector}

    def load_checkpoint(self):''')
rep('''        self.task1_base = st.get("task1_base")
        py = st["python_rng"]''',
'''        self.task1_base = st.get("task1_base")
        self.aux_anneal_start = st.get("aux_anneal_start")
        py = st["python_rng"]''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
