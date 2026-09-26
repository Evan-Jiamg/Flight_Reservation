# -*- coding: utf-8 -*-
"""Planner generation through a vLLM server (user decision 2026-09-26: Planner only; Ditto stays on HF).

The prompt is built exactly as PlannerLM builds it (fit_prompts, chat template, token ids, budget assert) and sent
as TOKEN IDS, so the learner scores exactly the ids that were generated. The server returns the generated token ids
(must include the end token when the output stopped, as the HF path does), per-token log-probs of the SAMPLED
tokens (behaviour log-probs for truncated importance sampling) and the finish reason ("length" = cut by max_new).
Nothing is repaired silently: a missing field, a length mismatch, a stop without an end token or a prompt that does
not fit the server's context raises.

The policy served is a LoRA adapter loaded at run time (/v1/load_lora_adapter) under a name carrying the policy
version and the adapter's sha256, recorded on every generation (`gen_adapter`).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

VLLM_URL = os.environ.get("PLANNER_VLLM_URL", "http://127.0.0.1:8031/v1")
BASE_NAME = os.environ.get("PLANNER_VLLM_BASE", "planner-base")        # --served-model-name of the server


def sha_dir(p):
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(p)):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            h.update(os.path.relpath(fp, p).replace("\\", "/").encode())
            with open(fp, "rb") as f:
                h.update(hashlib.sha256(f.read()).hexdigest().encode())
    return h.hexdigest()


def _post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError("vLLM %s -> HTTP %s: %s" % (url, e.code, e.read().decode("utf-8", "replace")[:400]))
    try:
        return json.loads(raw)
    except ValueError:
        return {"text": raw}


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class VLLMPlanner:
    """Generation backend attached to a PlannerLM (planner.remote = VLLMPlanner(...))."""

    def __init__(self, url=VLLM_URL, base_name=BASE_NAME, max_model_len=None, max_workers=32):
        self.url = url.rstrip("/")
        self.base_name = base_name
        models = _get(self.url + "/models")
        entry = next((m for m in models.get("data", []) if m.get("id") == base_name), None)
        if entry is None:
            raise RuntimeError("vLLM at %s does not serve %r (serves %r)" % (url, base_name,
                               [m.get("id") for m in models.get("data", [])]))
        self.max_model_len = int(max_model_len or entry.get("max_model_len") or 0)
        if not self.max_model_len:
            raise RuntimeError("vLLM did not report max_model_len; pass it explicitly")
        self.current = base_name           # the model name requests go to (base = the untrained Planner, no LoRA)
        self.loaded = []
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.n_requests = 0

    # ------------------------------------------------------------ policy (adapter) management
    def use_adapter(self, path, tag):
        """Serve the adapter at `path` under a name carrying `tag` (e.g. the policy version) and its sha; the
        previous adapter is unloaded. -> the name."""
        sha = sha_dir(path)
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
            old = [n for n in served if n not in (name, self.base_name)]
            self.loaded = [name]
            self.current = name
        for n in old:
            try:
                _post(self.url + "/unload_lora_adapter", {"lora_name": n})
            except RuntimeError:
                pass                       # an adapter that cannot be unloaded only costs memory; never used again
        return name

    def use_base(self):
        with self.lock:
            self.current = self.base_name
        return self.base_name

    # ------------------------------------------------------------ generation
    def _one(self, planner, it, name):
        user_fit, fit, ids = planner.build_prompt(it)
        if len(ids) + planner.max_new > self.max_model_len:
            raise AssertionError("Planner prompt %d + max_new %d > vLLM max_model_len %d (would be cut)"
                                 % (len(ids), planner.max_new, self.max_model_len))
        t = float(it["temperature"] or 0.0)
        body = {"model": name, "prompt": ids, "max_tokens": planner.max_new, "temperature": t,
                "top_p": float(it["top_p"]) if t > 0 else 1.0, "logprobs": 1, "return_token_ids": True,
                "skip_special_tokens": False}
        if t > 0:
            body["seed"] = int(it["seed"]) % (2 ** 31)
        r = _post(self.url + "/completions", body)
        ch = (r.get("choices") or [None])[0]
        if not ch:
            raise RuntimeError("vLLM returned no choice: %r" % (str(r)[:300],))
        gen = ch.get("token_ids")
        if gen is None:
            raise RuntimeError("vLLM did not return token_ids (return_token_ids unsupported?): refusing to guess ids")
        gen = list(gen)
        lps = ((ch.get("logprobs") or {}).get("token_logprobs"))
        if lps is None or len(lps) != len(gen):
            raise RuntimeError("vLLM log-probs (%s) do not match the %d generated tokens"
                               % (None if lps is None else len(lps), len(gen)))
        fin = ch.get("finish_reason")
        if fin not in ("stop", "length"):
            raise RuntimeError("unexpected finish_reason %r" % fin)
        eos = self._eos(planner)
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
            planner.n_calls += 1
        return {"raw": planner.tok.decode(gen, skip_special_tokens=True), "prompt_text": user_fit,
                "prompt_ids": list(ids), "gen_ids": gen, "gen_logprobs": [float(x) for x in lps],
                "gen_adapter": name,
                "fit": {**fit, "prompt_tokens": len(ids), "budget": planner.budget, "backend": "vllm", "gen_adapter": name,
                        "batched": 1, "batch_seed": int(it["seed"]) if t > 0 else None},
                "hit_max_new": fin == "length"}

    def _eos(self, planner):
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

    def generate_batch(self, planner, items):
        name = self.current
        futs = [self.pool.submit(self._one, planner, it, name) for it in items]
        return [f.result() for f in futs]
