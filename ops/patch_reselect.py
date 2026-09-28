import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:70])
    s = s.replace(old, new)


rep('''    def validate(self, u):
        """Task 2 episodes on the validation ids with the SAMPLED Planner (D5: --val-temperature, one
        replicate per --val-seeds entry) + Task 1 (greedy, as the benchmark); logged only to validation.jsonl;
        drives best.json and the D2 annealing trigger. Never enters history or the controller."""
        _tv = time.time()
        self.sync_generation_policy(u)          # validation generates with the policy after update u
        a = self.a
        prev = read_jsonl(self.p_val)
        if any(r.get("kind") == "summary" and r["update"] == u for r in prev):
            return
''', '''    def validate(self, u, seeds=None, p_val=None, reselect=False):
        """Task 2 episodes on the validation ids with the SAMPLED Planner (D5: --val-temperature, one
        replicate per --val-seeds entry) + Task 1 (greedy, as the benchmark); logged only to validation.jsonl;
        drives best.json and the D2 annealing trigger. Never enters history or the controller.
        reselect=True (checkpoint re-selection, user 2026-09-27): the same procedure with other seeds into another
        file (p_val), without any side effect (best.json, task1_base, D2 trigger and the checkpoints untouched)."""
        _tv = time.time()
        self.sync_generation_policy(u)          # validation generates with the policy after update u
        a = self.a
        seeds = list(a.val_seeds if seeds is None else seeds)
        p_val = p_val or self.p_val
        prev = read_jsonl(p_val)
        for r in prev:
            if r.get("kind") == "summary" and r["update"] == u:
                return r
''')
rep('''            for s in a.val_seeds:
                jobs.append((cid, s))''', '''            for s in seeds:
                jobs.append((cid, s))''')
rep('''                with self.io_lock:
                    append_jsonl(self.p_val, r)
                if ep["clean"]''', '''                with self.io_lock:
                    append_jsonl(p_val, r)
                if ep["clean"]''')
rep('''                with self.io_lock:
                    append_jsonl(self.p_val, r)
            return r''', '''                with self.io_lock:
                    append_jsonl(p_val, r)
            return r''')
rep('''        if t1 is not None and self.task1_base is None:''', '''        if not reselect and t1 is not None and self.task1_base is None:''')
rep('''        if t1 is not None and self.aux_anneal_start is None and u > self.task1_base["update"]''',
    '''        if not reselect and t1 is not None and self.aux_anneal_start is None and u > self.task1_base["update"]''')
rep('''                                  "val_temperature": a.val_temperature, "val_seeds": a.val_seeds,''',
    '''                                  "val_temperature": a.val_temperature, "val_seeds": seeds, "reselect": reselect,''')
rep('''        if sel is not None and (self.best is None or sel > self.best["selection_score"]):''',
    '''        if not reselect and sel is not None and (self.best is None or sel > self.best["selection_score"]):''')
rep('''            write_json_atomic(sp, st)
        append_jsonl(self.p_val, summary)
''', '''            write_json_atomic(sp, st)
        append_jsonl(p_val, summary)
        return summary
''')
# ------------------------------------------------------------------ re-selection driver
rep('''    # -------------------------------------------------------- main loop
    def run(self):''', '''    # -------------------------------------------------------- checkpoint re-selection
    def reselect(self):
        """Checkpoint re-selection (user 2026-09-27): every checkpoint that was validated during training is
        validated again with --reselect-seeds (same validation ids, temperature, env, clean re-run rule and
        selection formula as validate()); results go to reselect.jsonl, the choice to reselect_best.json.
        Training files (validation.jsonl, best.json, checkpoints) are never written."""
        a = self.a
        self.build()
        last = json.load(open(os.path.join(self.ckpt_root, "LATEST.json")))["update"]
        st = json.load(open(os.path.join(self.ckpt_dir(last), "state.json")))
        self.task1_base, self.update_done = st.get("task1_base"), last
        cands = sorted({r["update"] for r in read_jsonl(self.p_val) if r.get("kind") == "summary"})
        if not cands:
            raise SystemExit("no validated checkpoint in %s" % self.p_val)
        row = self.meta("reselect")
        self.check_provenance(row)
        out = os.path.join(a.out, "reselect.jsonl")
        append_jsonl(os.path.join(a.out, "reselect_meta.jsonl"),
                     {**row, "candidates": cands, "seeds": list(a.reselect_seeds)})
        summaries = []
        for u in cands:
            d = self.ckpt_dir(u)
            cst = json.load(open(os.path.join(d, "state.json")))
            self.learner.load_policy(d)
            if self.learner.policy_sha() != cst["policy_sha"]:
                raise AssertionError("checkpoint u%d: loaded policy sha differs from its record" % u)
            s = self.validate(u, seeds=a.reselect_seeds, p_val=out, reselect=True)
            summaries.append(s)
            print(json.dumps({"reselect_update": u, "selection_score": s["selection_score"],
                              "withheld": s["selection_withheld"], "n_episodes": s["n_episodes"]}), flush=True)
        ok = [s for s in summaries if s["selection_score"] is not None]
        best = max(ok, key=lambda s: (s["selection_score"], -s["update"])) if ok else None
        rec = {"seeds": list(a.reselect_seeds), "candidates": {str(s["update"]): s["selection_score"] for s in summaries},
               "withheld": [s["update"] for s in summaries if s["selection_score"] is None],
               "selection_formula": summaries[0]["selection_formula"],
               "selection_cfg_sha256": RR.cfg_sha(self.selection_cfg),
               "best": None if best is None else {
                   "update": best["update"], "selection_score": best["selection_score"], "policy_sha": best["policy_sha"],
                   "checkpoint": os.path.relpath(self.ckpt_dir(best["update"]), a.out).replace("\\\\", "/")},
               "time": time.time()}
        write_json_atomic(os.path.join(a.out, "reselect_best.json"), rec)
        return rec

    # -------------------------------------------------------- main loop
    def run(self):''')
rep('''"keep_optimizer_last", "dry_run_crash_after_episodes", "vllm_url", "intervention")''',
    '''"keep_optimizer_last", "dry_run_crash_after_episodes", "vllm_url", "intervention",
                     "reselect_seeds")''')
rep('''    ap.add_argument("--intervention", default=None,''', '''    ap.add_argument("--reselect-seeds", type=int, nargs="+", default=None,
                    help="checkpoint re-selection: re-validate every validated checkpoint with these seeds "
                         "(writes reselect.jsonl / reselect_best.json only; no training)")
    ap.add_argument("--intervention", default=None,''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
