"""Behaviour-preserving refactor so the GPU smoke test exercises the REAL code: Task 1 samples, aux examples and the
validation Task 1 row become Trainer methods (one_update / validate call them)."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''        _t = time.time()
        t1rows = self.task1_rollouts(u)
        timing["task1_groups_s"] = round(time.time() - _t, 1)
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
        t1_all = [x for r in t1rows for x in r["samples"]]''', '''        _t = time.time()
        t1rows = self.task1_rollouts(u)
        timing["task1_groups_s"] = round(time.time() - _t, 1)
        t1_samples, t1_skipped = self.task1_samples(t1rows, pv, psha)
        samples += t1_samples
        t1_all = [x for r in t1rows for x in r["samples"]]''')
rep('''        aux, w_aux = [], self.aux_weight(u, cfg)
        if w_aux > 0:
            for r in t1rows:
                if r.get("refill"):
                    continue                 # v16: the supervision amount stays fixed (base groups only)
                x = (r["samples"] or [{}])[0].get("aux")
                if x is not None:
                    assert x["want_end"] == r["real_final"], "stop-supervision label disagrees with the human"
                    aux.append(dict(x, weight=w_aux))''', '''        w_aux = self.aux_weight(u, cfg)
        aux = self.aux_examples(t1rows, w_aux)''')
rep('''    # -------------------------------------------------------- one update
    def one_update(self, u):''', '''    def task1_samples(self, t1rows, pv, psha):
        """GRPO samples of the Task 1 stop groups (one Planner step per sample; a group without spread is skipped).
        With stop credit the group advantage acts only on the end_session value tokens. -> (samples, n_skipped)"""
        samples = []
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
        return samples, t1_skipped

    def aux_examples(self, t1rows, w_aux):
        """Stop-supervision examples: one per BASE Task 1 group that carries one (v16: never from refill groups, so the
        supervision amount stays fixed), weighted by the effective aux weight."""
        aux = []
        if w_aux > 0:
            for r in t1rows:
                if r.get("refill"):
                    continue
                x = (r["samples"] or [{}])[0].get("aux")
                if x is not None:
                    assert x["want_end"] == r["real_final"], "stop-supervision label disagrees with the human"
                    aux.append(dict(x, weight=w_aux))
        return aux

    def task1_eval_row(self, u, psha, cid):
        """Validation Task 1 of one conversation: the greedy Task 1 run (term_f1 etc. via task1_stop) and, at every
        decision point t >= 2, the teacher-forced end probability (v16 item 7). -> the validation.jsonl row."""
        t1r = self.env.run_task1(cid, keep_prompts=True)
        probs = []
        for x in t1r["turns"]:
            if x["t"] < 2:
                continue               # turn 1 cannot end: not a decision point
            pr = self.env.task1_end_probe(cid, x["t"], x["user_prompt"], x["real_final"])
            if pr["valid"]:
                # the learner's forward shares the GPU with the Speaker / Planner calls of Task2Env
                with (getattr(self.env, "gpu_lock", None) or contextlib.nullcontext()):
                    pe = float(self.learner.end_prob(pr))
            else:
                # no value to score: an invalid plan reads as "not ending" (the benchmark's reading); a valid
                # decision whose value tokens could not be located counts as its greedy decision
                pe = 1.0 if (pr.get("decision_valid") and pr["greedy_end"]) else 0.0
            probs.append({"t": x["t"], "real_final": bool(x["real_final"]), "p_end": pe, "valid": bool(pr["valid"]),
                          "greedy_end": bool(pr["greedy_end"])})
        for x in t1r["turns"]:
            x.pop("user_prompt", None)
        return {"kind": "task1", "update": u, "policy_sha": psha, "conversation_id": cid,
                "split": "validation", "task1": t1r, "end_probs": probs, "time": time.time()}

    # -------------------------------------------------------- one update
    def one_update(self, u):''')
rep('''            if r is None or r["policy_sha"] != psha or "end_probs" not in r:
                t1r = self.env.run_task1(cid, keep_prompts=True)
                probs = []
                for x in t1r["turns"]:
                    if x["t"] < 2:
                        continue               # turn 1 cannot end: not a decision point
                    pr = self.env.task1_end_probe(cid, x["t"], x["user_prompt"], x["real_final"])
                    if pr["valid"]:
                        # the learner's forward shares the GPU with the Speaker / Planner calls of Task2Env
                        with (getattr(self.env, "gpu_lock", None) or contextlib.nullcontext()):
                            pe = float(self.learner.end_prob(pr))
                    else:
                        # no value to score: an invalid plan reads as "not ending" (the benchmark's reading); a valid
                        # decision whose value tokens could not be located counts as its greedy decision
                        pe = 1.0 if (pr.get("decision_valid") and pr["greedy_end"]) else 0.0
                    probs.append({"t": x["t"], "real_final": bool(x["real_final"]), "p_end": pe, "valid": bool(pr["valid"]),
                                  "greedy_end": bool(pr["greedy_end"])})
                for x in t1r["turns"]:
                    x.pop("user_prompt", None)
                r = {"kind": "task1", "update": u, "policy_sha": psha, "conversation_id": cid,
                     "split": "validation", "task1": t1r, "end_probs": probs, "time": time.time()}
                with self.io_lock:''', '''            if r is None or r["policy_sha"] != psha or "end_probs" not in r:
                r = self.task1_eval_row(u, psha, cid)
                with self.io_lock:''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
