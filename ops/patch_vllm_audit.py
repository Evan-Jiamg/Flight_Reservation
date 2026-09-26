"""Fixes from the audit of the vLLM backend + TIS change (2026-09-26). Run from sep-sim/."""


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


patch("vllm_planner.py", [
    # (1) an adapter already on the server (same name = same content sha) is adopted, never re-posted
    ('''        sha = sha_dir(path)
        name = "%s-%s" % (tag, sha[:12])
        with self.lock:
            if name == self.current:
                return name
            _post(self.url.rsplit("/v1", 1)[0] + "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path})
            served = [m.get("id") for m in _get(self.url + "/models").get("data", [])]
            if name not in served:
                raise RuntimeError("adapter %s not listed by vLLM after loading (serves %r)" % (name, served))''',
     '''        sha = sha_dir(path)
        name = "%s-%s" % (tag, sha[:12])
        with self.lock:
            if name == self.current:
                return name
            served = [m.get("id") for m in _get(self.url + "/models").get("data", [])]
            if name not in served:
                # (a server left up by a crashed run may still hold it: the name carries the content sha, so a listed
                # name is the same adapter and is adopted instead of re-posted, which vLLM would refuse)
                try:
                    _post(self.url + "/load_lora_adapter", {"lora_name": name, "lora_path": path})
                except RuntimeError as e:
                    if "already" not in str(e).lower():
                        raise
                served = [m.get("id") for m in _get(self.url + "/models").get("data", [])]
                if name not in served:
                    raise RuntimeError("adapter %s not listed by vLLM after loading (serves %r)" % (name, served))
            old = [n for n in served if n not in (name, self.base_name)]'''),
    ('''            old = [n for n in self.loaded if n != name]
            self.loaded = [name]
            self.current = name
        for n in old:
            try:
                _post(self.url.rsplit("/v1", 1)[0] + "/v1/unload_lora_adapter", {"lora_name": n})''',
     '''            self.loaded = [name]
            self.current = name
        for n in old:
            try:
                _post(self.url + "/unload_lora_adapter", {"lora_name": n})'''),
    # (7) cached eos ids, prompt ids echoed back must match, a length stop must be exactly max_new, locked counter
    ('''        eos = planner.eos_ids()
        if fin == "stop" and (not gen or gen[-1] not in eos):
            raise RuntimeError("vLLM stopped without an end token at the end of token_ids (stop_reason %r)"
                               % ch.get("stop_reason"))
        with self.lock:
            self.n_requests += 1
        planner.n_calls += 1''',
     '''        eos = self._eos(planner)
        if fin == "stop" and (not gen or gen[-1] not in eos):
            raise RuntimeError("vLLM stopped without an end token at the end of token_ids (stop_reason %r)"
                               % ch.get("stop_reason"))
        if fin == "length" and len(gen) != planner.max_new:
            raise RuntimeError("finish_reason 'length' with %d tokens, max_new is %d" % (len(gen), planner.max_new))
        echoed = ch.get("prompt_token_ids") or r.get("prompt_token_ids")
        if echoed is not None and list(echoed) != list(ids):
            raise RuntimeError("the server's prompt token ids differ from the ids sent")
        with self.lock:
            self.n_requests += 1
            planner.n_calls += 1'''),
    ('''                "fit": {**fit, "prompt_tokens": len(ids), "budget": planner.budget, "backend": "vllm",''',
     '''                "fit": {**fit, "prompt_tokens": len(ids), "budget": planner.budget, "backend": "vllm", "gen_adapter": name,'''),
    ('''    def generate_batch(self, planner, items):''',
     '''    def _eos(self, planner):
        e = getattr(self, "_eos_cache", None)
        if e is None:
            e = self._eos_cache = set(planner.eos_ids())
        return e

    def attach(self, planner):
        """planner.remote = self, and the fitter's budget is capped so a fitted prompt + max_new always fits the
        server context (a longer prompt is compacted -- recorded, unclean -- instead of raising mid-run)."""
        import fit_prompts as F
        planner.remote = self
        cap = self.max_model_len - planner.max_new - F.MARGIN
        if cap < planner.budget:
            planner.budget_hf = planner.budget
            planner.budget = cap
        return planner

    def generate_batch(self, planner, items):'''),
])

patch("task2_env.py", [
    ('''                "planner_backend": "vllm" if getattr(self.planner, "remote", None) is not None else "hf",''',
     '''                "planner_backend": "vllm" if getattr(self.planner, "remote", None) is not None else "hf",
                "planner_vllm_max_model_len": getattr(getattr(self.planner, "remote", None), "max_model_len", None),'''),
])

for p in ("task1_v4.py", "rollout_v4.py"):
    patch(p, [
        ('''    planner = PlannerLM(path, gpu, load_model=False)
    planner.remote = vllm_planner.VLLMPlanner(url)''',
         '''    planner = PlannerLM(path, gpu, load_model=False)
    vllm_planner.VLLMPlanner(url).attach(planner)'''),
    ])
# (8) an HF evaluation of a vLLM-trained adapter is possible as a named ablation
patch("task1_v4.py", [
    ('''                                                  "planner_path": a.planner_path, "planner_backend": a.planner_backend})''',
     '''                                                  "planner_path": a.planner_path,
                                                  **({} if a.ablation else {"planner_backend": a.planner_backend})})'''),
])
patch("rollout_v4.py", [
    ('''                                                     "planner_path": args.planner_path,
                                                     "planner_backend": args.planner_backend})''',
     '''                                                     "planner_path": args.planner_path,
                                                     **({} if args.ablation else {"planner_backend": args.planner_backend})})'''),
])

patch("rl_algos.py", [
    # (4) the learner/vLLM mismatch is checked right after prepare(), before any optimizer step
    ('''    def update(self, samples, cfg, seed, aux=None):
        """cfg: controller cfg (lr, kl_coef). aux: optional stop-token supervision examples
        {prompt_ids, prefix_ids, target_ids, want_end, weight}. -> stats dict."""''',
     '''    def update(self, samples, cfg, seed, aux=None, mismatch_abort=None):
        """cfg: controller cfg (lr, kl_coef). aux: optional stop-token supervision examples
        {prompt_ids, prefix_ids, target_ids, want_end, weight}. mismatch_abort: raise MismatchAbort before any
        optimizer step when the mean |log pi_old - log pi_behaviour| of vLLM-sampled tokens exceeds it. -> stats."""'''),
    ('''        self.prepare(samples)
        _set_mode(self.model, self.acfg["forward_mode"])''',
     '''        self.prepare(samples)
        if mismatch_abort is not None:
            tot, n = 0.0, 0
            for s in samples:
                if s.get("behav_logp") is not None:
                    bl = torch.tensor(s["behav_logp"], dtype=s["old_logp"].dtype)
                    tot += float((s["old_logp"].cpu() - bl).abs().sum())
                    n += int(bl.numel())
            if n and tot / n > mismatch_abort:
                raise MismatchAbort(tot / n)
        _set_mode(self.model, self.acfg["forward_mode"])'''),
    ('''def tis_weights(old_logp, behav_logp, cap):''',
     '''class MismatchAbort(RuntimeError):
    def __init__(self, value):
        super().__init__("mean |log pi_learner - log pi_vllm| = %.4f" % value)
        self.value = value


def tis_weights(old_logp, behav_logp, cap):'''),
])

patch("train_planner_rl.py", [
    # (1) attach through the client (budget capped to the server context)
    ('''            planner.remote = vllm_planner.VLLMPlanner(a.vllm_url)''',
     '''            vllm_planner.VLLMPlanner(a.vllm_url).attach(planner)'''),
    # (4) abort before any optimizer step; the aborted update's rollouts are never reused
    ('''        stats = self.learner.update(samples, cfg, seed=seed_of(self.a.seed, "update", u), aux=aux or None)
        timing["learner_s"] = round(time.time() - _t, 1)
        mm = stats.get("behav_mismatch_mean")
        if mm is not None and mm > self.a.behav_mismatch_abort:
            # before the checkpoint is written: a resume redoes this update
            raise SystemExit("update %d: mean |log pi_learner - log pi_vllm| = %.4f > %.4f (phase-0 measured 0.017): "
                             "the served adapter does not match the learner?" % (u, mm, self.a.behav_mismatch_abort))''',
     '''        try:
            stats = self.learner.update(samples, cfg, seed=seed_of(self.a.seed, "update", u), aux=aux or None,
                                        **({"mismatch_abort": self.a.behav_mismatch_abort} if not self.a.dry_run else {}))
        except RA.MismatchAbort as e:
            # before any optimizer step and before the checkpoint; the marker makes a resume regenerate update u's
            # rollouts instead of reusing the same mismatched ones
            write_json_atomic(os.path.join(self.a.out, "ABORTED_u%05d.json" % u),
                              {"update": u, "mismatch": e.value, "threshold": self.a.behav_mismatch_abort,
                               "adapter": getattr(self, "gen_name", None), "time": time.time()})
            raise SystemExit("update %d: %s > %.4f (phase-0 measured 0.017): the served adapter does not match the "
                             "learner?" % (u, e, self.a.behav_mismatch_abort))
        timing["learner_s"] = round(time.time() - _t, 1)'''),
    ('''        reuse = {}
        for r in read_jsonl(self.p_roll):
            if r["update"] == u and r["policy_version"] == pv and r["policy_sha"] == psha:
                reuse[(r["slot"], r["replicate"])] = r''',
     '''        reuse = {}
        aborted = os.path.exists(os.path.join(a.out, "ABORTED_u%05d.json" % u))
        for r in read_jsonl(self.p_roll):
            if not aborted and r["update"] == u and r["policy_version"] == pv and r["policy_sha"] == psha:
                reuse[(r["slot"], r["replicate"])] = r'''),
    ('''        for r in read_jsonl(self.p_roll_t1):
            if r["update"] == u and r["policy_version"] == pv and r["policy_sha"] == psha:''',
     '''        aborted = os.path.exists(os.path.join(a.out, "ABORTED_u%05d.json" % u))
        for r in read_jsonl(self.p_roll_t1):
            if not aborted and r["update"] == u and r["policy_version"] == pv and r["policy_sha"] == psha:'''),
    # (2) vLLM log-probs are the raw (T=1, no top_p) distribution: training sampling must be T=1, top_p=1
    ('''    if a.planner_backend != "vllm" and not a.dry_run:
        off["planner_backend"] = a.planner_backend''',
     '''    if a.planner_backend != "vllm" and not a.dry_run:
        off["planner_backend"] = a.planner_backend
    if a.planner_backend == "vllm" and (a.temperature != 1.0 or a.top_p != 1.0):
        ap.error("with the vLLM backend the training rollouts must use --temperature 1 --top-p 1: the server's "
                 "behaviour log-probs are the raw distribution, so any other value would bias the TIS weights")'''),
])

patch("verify_pipeline.py", [
    # (5) no RL tokens in an update -> no TIS statistics expected; (6) adapter sha and Task 1 rows
    ('''            st = u.get("learner_stats") or {}
            w = "update %s" % u.get("update")
            mm = st.get("behav_mismatch_mean")''',
     '''            st = u.get("learner_stats") or {}
            w = "update %s" % u.get("update")
            if st.get("skipped_update") or not st.get("n_tokens"):
                continue                     # no RL token in this update: nothing to weight
            mm = st.get("behav_mismatch_mean")'''),
    ('''            if isinstance(u, int):
                rep.ok("rl.vllm_adapter", str(g.get("gen_adapter") or "").startswith("p%d-" % (u - 1)), w,
                       "generated by %r, the policy of update %d is p%d" % (g.get("gen_adapter"), u, u - 1))''',
     '''            if isinstance(u, int):
                rep.ok("rl.vllm_adapter", g.get("gen_adapter") == adapter_name(rl_dir, u - 1), w,
                       "generated by %r, the policy of update %d is %r" % (g.get("gen_adapter"), u, adapter_name(rl_dir, u - 1)))
    t1p = os.path.join(rl_dir, "rollouts_task1.jsonl")
    if os.path.exists(t1p):
        for line in open(t1p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            u = r.get("update")
            for x in r.get("samples") or []:
                g = x.get("planner_gen") or {}
                w = "task1 %s t%s u%s" % (str(r.get("conversation_id"))[:12], r.get("t"), u)
                rep.ok("rl.vllm_logprobs", isinstance(g.get("gen_logprobs"), list)
                       and len(g["gen_logprobs"]) == len(g.get("gen_ids") or []), w, "behaviour log-probs missing")
                rep.ok("rl.vllm_adapter", g.get("gen_adapter") == adapter_name(rl_dir, u - 1), w,
                       "generated by %r, expected %r" % (g.get("gen_adapter"), adapter_name(rl_dir, u - 1)))'''),
    ('''def check_vllm_generation(rl_dir, rollouts, meta, rep, abort=0.1):''',
     '''_ADAPTER_NAMES = {}


def adapter_name(rl_dir, pv):
    """The name the trainer serves checkpoint pv's adapter under: p<pv>-<sha12 of the adapter directory>."""
    key = (rl_dir, pv)
    if key not in _ADAPTER_NAMES:
        import vllm_planner
        d = os.path.join(rl_dir, "ckpt", "u%05d" % pv, "adapter")
        _ADAPTER_NAMES[key] = ("p%d-%s" % (pv, vllm_planner.sha_dir(d)[:12])) if os.path.isdir(d) else None
    return _ADAPTER_NAMES[key]


def check_vllm_generation(rl_dir, rollouts, meta, rep, abort=0.1):'''),
])
print("ok")
