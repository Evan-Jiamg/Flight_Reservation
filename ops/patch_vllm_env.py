"""vLLM Planner backend wiring in task2_env.py (user decision 2026-09-26). Run from sep-sim/."""


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


patch("task2_env.py", [
    ('''    def __init__(self, path, gpu, nf4=False, dtype="bfloat16", adapter=None, trainable=False,
                 max_new=PLANNER_MAX_NEW):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import fit_prompts as F
        self.path, self.gpu, self.max_new = path, gpu, max_new
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token''',
     '''    def __init__(self, path, gpu, nf4=False, dtype="bfloat16", adapter=None, trainable=False,
                 max_new=PLANNER_MAX_NEW, load_model=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import fit_prompts as F
        self.path, self.gpu, self.max_new = path, gpu, max_new
        self.remote = None       # a VLLMPlanner: generation goes to the vLLM server, prompts built here, ids exact
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if not load_model:
            # generation-only use through vLLM (evaluation): tokenizer and config, no weights on the GPU
            from transformers import AutoConfig
            if adapter:
                raise ValueError("load_model=False: the adapter is served by vLLM (VLLMPlanner.use_adapter)")
            self.model = None
            self.context = int(getattr(AutoConfig.from_pretrained(path), "max_position_embeddings", None) or 32768)
            self.budget = min(F.PLANNER_BUDGET, self.context - max_new - F.MARGIN)
            self.adapter = None
            self.n_calls = 0
            return'''),
    ('''    def eos_ids(self):
        gc = getattr(self.model, "generation_config", None)''',
     '''    def eos_ids(self):
        if self.model is None:
            from transformers import GenerationConfig
            e = GenerationConfig.from_pretrained(self.path).eos_token_id
            return set(e if isinstance(e, (list, tuple)) else [e])
        gc = getattr(self.model, "generation_config", None)'''),
    ('''    def generate_batch(self, items):
        """Batched generate: items share temperature/top_p (the Batcher groups them). Per item the result
        has the same fields as generate(); prompt_ids are the item's own unpadded ids, gen_ids run up to
        and including the first end token (as a single generate stops there)."""
        import torch
        import fit_prompts as F
        if not items:
            return []
        t0, p0 = items[0]["temperature"], items[0]["top_p"]
        assert all(it["temperature"] == t0 and it["top_p"] == p0 for it in items), "mixed sampling settings in a batch"
        prompts = []
        for it in items:
            user_fit, fit = F.fit_planner_user(self.tok, it["system"], it["user"], budget=self.budget)
            text = self.tok.apply_chat_template([{"role": "system", "content": it["system"]},
                                                 {"role": "user", "content": user_fit}],
                                                tokenize=False, add_generation_prompt=True)
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) > self.budget:
                raise AssertionError("Planner prompt %d > budget %d after fitting" % (len(ids), self.budget))
            prompts.append((user_fit, fit, ids))''',
     '''    def build_prompt(self, it):
        """(user_fit, fit, ids) for one item: fitted (never truncated) user prompt, chat template, token ids,
        asserted within budget. The ONE prompt builder of both generation backends (HF and vLLM)."""
        import fit_prompts as F
        user_fit, fit = F.fit_planner_user(self.tok, it["system"], it["user"], budget=self.budget)
        text = self.tok.apply_chat_template([{"role": "system", "content": it["system"]},
                                             {"role": "user", "content": user_fit}],
                                            tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.budget:
            raise AssertionError("Planner prompt %d > budget %d after fitting" % (len(ids), self.budget))
        return user_fit, fit, ids

    def generate_batch(self, items):
        """Batched generate: items share temperature/top_p (the Batcher groups them). Per item the result
        has the same fields as generate(); prompt_ids are the item's own unpadded ids, gen_ids run up to
        and including the first end token (as a single generate stops there). With a vLLM backend attached
        (self.remote) the same prompts are generated by the server (plus behaviour log-probs)."""
        import torch
        if not items:
            return []
        if self.remote is not None:
            return self.remote.generate_batch(self, items)
        t0, p0 = items[0]["temperature"], items[0]["top_p"]
        assert all(it["temperature"] == t0 and it["top_p"] == p0 for it in items), "mixed sampling settings in a batch"
        prompts = [self.build_prompt(it) for it in items]'''),
    # Task2Env: a remote Planner takes no GPU lock and needs no batcher (the server batches continuously)
    ('''        self.planner_batcher = self.speaker_batcher = None
        if batch:
            import batching
            if not hasattr(self.speaker, "say_batch"):
                raise RuntimeError("batching needs the E1.6 Ditto speaker (say_batch)")
            self.planner_batcher = batching.Batcher(planner.generate_batch, self.gpu_lock,
                                                    key=lambda it: (it["temperature"], it["top_p"]),
                                                    max_batch=max_batch, name="planner-batcher")''',
     '''        self.planner_batcher = self.speaker_batcher = None
        if batch:
            import batching
            if not hasattr(self.speaker, "say_batch"):
                raise RuntimeError("batching needs the E1.6 Ditto speaker (say_batch)")
            if getattr(planner, "remote", None) is None:
                self.planner_batcher = batching.Batcher(planner.generate_batch, self.gpu_lock,
                                                        key=lambda it: (it["temperature"], it["top_p"]),
                                                        max_batch=max_batch, name="planner-batcher")'''),
    ('''        if self.planner_batcher is not None:
            return self.planner_batcher(item)
        # unbatched: the SAME function as the batched path, one item at a time (identical code, no padding)
        with self.gpu_lock:
            return self.planner.generate_batch([item])[0]''',
     '''        if self.planner_batcher is not None:
            return self.planner_batcher(item)
        if getattr(self.planner, "remote", None) is not None:
            return self.planner.generate_batch([item])[0]     # vLLM server: no local GPU, no lock
        # unbatched: the SAME function as the batched path, one item at a time (identical code, no padding)
        with self.gpu_lock:
            return self.planner.generate_batch([item])[0]'''),
    ('''    def _plan_many(self, items):
        if self.planner_batcher is not None:
            return self.planner_batcher.map(items)''',
     '''    def _plan_many(self, items):
        if self.planner_batcher is not None:
            return self.planner_batcher.map(items)
        if getattr(self.planner, "remote", None) is not None:
            return self.planner.generate_batch(items)         # concurrent requests to the vLLM server'''),
    # the generation records carry the behaviour log-probs and the adapter that produced them
    ('''                base["planner_gen"] = {"prompt_ids": g["prompt_ids"], "gen_ids": g["gen_ids"], "stop_mask": sm,
                                       "note_mask": nm, "hit_max_new": g["hit_max_new"],''',
     '''                base["planner_gen"] = {"prompt_ids": g["prompt_ids"], "gen_ids": g["gen_ids"], "stop_mask": sm,
                                       "note_mask": nm, "hit_max_new": g["hit_max_new"],
                                       "gen_logprobs": g.get("gen_logprobs"), "gen_adapter": g.get("gen_adapter"),'''),
    ('''                        "planner_gen": {"prompt_ids": gen["prompt_ids"], "gen_ids": gen["gen_ids"], "stop_mask": sm,
                                        "note_mask": nm, "hit_max_new": gen["hit_max_new"],''',
     '''                        "planner_gen": {"prompt_ids": gen["prompt_ids"], "gen_ids": gen["gen_ids"], "stop_mask": sm,
                                        "note_mask": nm, "hit_max_new": gen["hit_max_new"],
                                        "gen_logprobs": gen.get("gen_logprobs"), "gen_adapter": gen.get("gen_adapter"),'''),
    ('''        return {"arm": self.arm, "planner": self.planner.path, "planner_adapter": self.planner.adapter,''',
     '''        return {"arm": self.arm, "planner": self.planner.path, "planner_adapter": self.planner.adapter,
                "planner_backend": "vllm" if getattr(self.planner, "remote", None) is not None else "hf",
                "planner_vllm_url": getattr(getattr(self.planner, "remote", None), "url", None),'''),
])
print("ok")
