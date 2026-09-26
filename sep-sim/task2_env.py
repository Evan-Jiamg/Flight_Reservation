# -*- coding: utf-8 -*-
"""Task2Env: one reusable Task 2 environment for evaluation rollouts AND RL (Planner is the policy).

Everything that decides behaviour lives here, once, so evaluation and RL cannot drift apart:
  * PlannerLM   any causal LM (pend: Qwen3-4B-Instruct-2507), optional LoRA adapter
                (trainable for RL). Prompts are fitted (fit_prompts, never truncated) to
                min(fit_prompts.PLANNER_BUDGET, model context - max_new - margin). Tokenized with
                add_special_tokens=False (the chat template carries any BOS).
                generate(system, user, temperature, top_p, seed) -> dict(raw, prompt_ids, gen_ids, fit)
  * Speaker     FitDittoSpeaker (D5 fix), frozen, v2fix guards and selector line for line.
  * Judge       goal_judge.GoalJudge (a2 only), frozen.
  * arms        pend (the method, user 2026-09-25) = E1.6 tree, no annotations, no goal judge; the Planner
                     judges the goal and its end_session makes the planned message the last one (emitted
                     close); Implicit Profile, per-slot few-shot, Borda selector (see ops/AUDIT_SPEC_pend_grpo.md).
                e16 = the E1.6 baseline as generated; a0 / a2 / final = older arms kept for comparison only.
The env never reads outcome data or dataset-wide statistics; leak gates live in the callers.

Episode rows have the schema written by rollout_ditto_v3.py (so analyzers, derivations and the
pipeline verifier apply unchanged) plus, when record_generation=True, per step
"planner_gen": {"prompt_ids", "gen_ids", "temperature", "top_p", "seed"} for RL.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import threading
import time

V2FIX = {
    "SEPSIM_ANTILEAK": "1", "SEPSIM_NEARCOPY": "1", "SEPSIM_COPY_SCOPE": "both",
    "SEPSIM_NSAMP": "3", "SEPSIM_SELECTOR": "length", "SEPSIM_LENGTH_SELECT": "1",
    "SEPSIM_POSITION": "system", "SEPSIM_END_PROBE": "0", "SEPSIM_GUARDRAILS": "0",
}
# E1.6 (Sep-1st-Simulator-e1r @ cf19400), copied read-only to /tmp2 by ops/r_e1rtree.sh; the tree
# E1.6 was generated from (the config in endfix/stage_e16.sh), with no uncommitted changes.
E16_COMMON = {
    "SEPSIM_NO_ANN": "1", "SEPSIM_ROLESTOP": "1", "SEPSIM_ACT_FULL": "1", "SEPSIM_T1_SAMPLE": "1",
    "SEPSIM_PLANNER_END": "1",
    # UserLM-only (no Ditto counterpart): off, and DittoSpeaker.load refuses them if set
    "SEPSIM_INTENT_PROSE": "0", "SEPSIM_ENDGATE": "0", "SEPSIM_ENDMASK_RETRY": "0",
    "SEPSIM_NEXTSTEP_INLINE": "0", "SEPSIM_T1_INTENT_ONLY": "0", "SEPSIM_PROSE_PERSONA": "0",
    "SEPSIM_KEEPEND": "0", "SEPSIM_ENDSCORE": "0", "SEPSIM_NOSURV": "0", "SEPSIM_AGENDA_CORROB": "0",
}
ARM_ENV = {
    "a0": {"SEPSIM_ACT_PRIOR": "off"},
    "a2": {"SEPSIM_ACT_PRIOR": "nostopclobber"},
    # Baseline: E1.6's Planner side as generated (override on, gated by the self-judged ledger)
    "e16": dict(E16_COMMON, SEPSIM_SELF_JUDGE="1", SEPSIM_STOP_LEDGER="1", SEPSIM_ACT_PRIOR="off"),
    # Final: E1.6 + v3 fixes; the goal judge replaces SELF_JUDGE, no override, no band
    "final": dict(E16_COMMON, SEPSIM_SELF_JUDGE="0", SEPSIM_STOP_LEDGER="0", SEPSIM_ACT_PRIOR="nostopclobber"),
    # pend (user, 2026-09-25): Planner + Ditto + Selector, no goal judge; the Planner judges the goal
    # itself and its end_session makes the planned message the last one (emitted close, then end)
    "pend": dict(E16_COMMON, SEPSIM_SELF_JUDGE="0", SEPSIM_STOP_LEDGER="0", SEPSIM_ACT_PRIOR="nostopclobber"),
}
V3_ARMS = ("a2", "final")            # goal judge + silent Planner exit
E16_ARMS = ("e16", "final", "pend")  # built on the E1.6 tree
EMIT_END_ARMS = ("e16", "pend")      # Planner end = the planned message is the last one
BENCH = "/tmp2/hchsu/trec2026-usersim-benchmark"
TREE_V2FIX = "/home/mzjiang/Sep-Simulator"
TREE_E16 = "/tmp2/mzjiang_usersim/grpo_planner/trees/e1r_cf19400"
TREE = TREE_V2FIX


FOLDS_GP = os.path.join("/tmp2/hchsu/trec2026-usersim-benchmark", "domains/main_dataset_search/folds3_goal_persona_v1.json")


def make_fewshot_pool(recs_by_cid, allowed, folds_path=FOLDS_GP):
    """Few-shot pool over the allowed conversations; goal/persona ids from the benchmark's fold manifest
    (every selection excludes the current conversation and those sharing its goal or persona)."""
    import implicit_profile as IP
    from sepsim import pipeline
    F = json.load(open(folds_path, encoding="utf-8"))
    return IP.FewShotPool(recs_by_cid, allowed, F["goal_of"], F["persona_of"], pipeline.split_messages)


def tree_of(arm):
    return TREE_E16 if arm in E16_ARMS else TREE_V2FIX
WORK = "/tmp2/mzjiang_usersim/task2"
DITTO = "/tmp2/mzjiang_usersim/models/Ditto-8B"
T_MAX = 10
PLANNER_MAX_NEW = 1536   # Qwen3-4B writes the ACT_FULL JSON in ~700-900 tokens; 600 cut it (smoke 2026-09-25)

HERE = os.path.dirname(os.path.abspath(__file__))
# The benchmark Ledger asks its judge with max_tokens=200. That is enough for gpt-5-mini at minimal
# effort, but a model that reasons before answering (gpt-oss-120b on vLLM) spends all 200 tokens on
# reasoning and returns an EMPTY answer, which the Ledger reads as "nothing revealed" -> coverage 0,
# silently (measured 2026-09-25: 200 -> finish=length, content None; 4000 -> 284 tokens, correct verdict).
# Outside the gpt-5 dialect the request budget is raised to this floor -- the same rule the benchmark
# applies to gpt-5 with MIN_COMPLETION_TOKENS -- and every empty answer is counted, never hidden.
JUDGE_MIN_TOKENS = int(os.environ.get("JUDGE_MIN_TOKENS", "4000"))


R0_CONTEXT = int(os.environ.get("R0_CONTEXT", "12288"))     # gpt-oss-120b on the vLLM server: max_model_len
JUDGE_RETRY_TOKENS = int(os.environ.get("JUDGE_RETRY_TOKENS", "8000"))   # one re-request of an unparseable verdict
PEND_PORT = os.environ.get("PEND_R0_PORT", "8029")        # our gpt-oss-120b vLLM server (user decision, option A)

# Per-episode counters of R0 / ledger-judge incidents. Episodes run one per thread and every R0 and
# ledger call of an episode is made on that episode's thread, so a thread-local dict attributes each
# incident to exactly one episode (the *_total counters on the clients stay process-wide).
_EP = threading.local()
EPISODE_COUNTERS = ("r0_len_retries", "r0_len_truncated", "r0_ctx_fit", "r0_empty",
                    "judge_empty", "judge_unparseable", "judge_retries", "judge_retry_failed", "judge_error")


def episode_begin():
    _EP.counts = {k: 0 for k in EPISODE_COUNTERS}


def episode_end():
    c = getattr(_EP, "counts", None)
    _EP.counts = None
    return dict(c) if c else {k: 0 for k in EPISODE_COUNTERS}


_ORPHANS = {"n": 0}
_ORPHAN_LOCK = threading.Lock()


def _ep_count(name):
    c = getattr(_EP, "counts", None)
    if c is not None:
        c[name] += 1
    else:
        # an incident on a thread with no active episode: if an episode is running, the one-episode-per-thread
        # assumption is broken (e.g. the Ledger fans judge calls out to a pool) -- run_episode refuses then
        with _ORPHAN_LOCK:
            _ORPHANS["n"] += 1


def orphan_incidents():
    return _ORPHANS["n"]


def rl_masks(planner, g, unparsed, diag, t):
    """(stop_mask, note_mask) for one Planner generation.
    stop_mask gets stop credit only for a REAL decision: None when the output was not parsed, was cut by
    the token cap, has no valid end_session value, is turn 1 (cannot end; the answer is ignored), or its
    turn-1 end was ignored. note_mask marks the profile_note value (kept out of the sequence advantage,
    D4); None when absent."""
    diag = diag or {}
    ok = not unparsed and not g.get("hit_max_new") and t >= 2 and diag.get("end_session_valid") is True \
        and not diag.get("end_session_t1_ignored") and not diag.get("cut_by_max_new")
    sm = planner.stop_mask(g["gen_ids"]) if ok else None
    if sm is not None:
        raw = diag.get("end_session_raw")
        want = "true" if (raw is True or str(raw).strip().lower() == "true") else "false"
        txt = planner.tok.decode([x for x, m in zip(g["gen_ids"], sm) if m], skip_special_tokens=False)
        if want not in txt.lower():
            diag["stop_mask_mismatch"] = True      # the first match is not the parsed value: no stop credit
            sm = None
    nm = planner.field_mask(g["gen_ids"], "profile_note")      # None unless a closed string value exists
    return sm, nm


def step_compacted(step):
    """True when the Planner prompt or any Speaker candidate prompt of this step was compacted (history
    dropped to fit): never allowed for pend (verify FAILs; the episode leaves the reward groups)."""
    if (step.get("planner_fit") or {}).get("compacted"):
        return True
    return any((f or {}).get("compacted") for f in (step.get("speaker_fits") or []))


def episode_clean(counts):
    """An episode may enter a reward group only if no R0 reply was cut and no ledger verdict was lost."""
    return counts.get("r0_len_truncated", 0) == 0 and counts.get("r0_empty", 0) == 0 \
        and counts.get("judge_empty", 0) == 0 and counts.get("judge_unparseable", 0) == 0 \
        and counts.get("judge_error", 0) == 0


def prompt_tokens_from_error(msg):
    """Prompt token count from a vLLM / OpenAI 'maximum context length' refusal, in any of the known
    wordings: '(10500 in the messages, 3000 in the completion)', 'your request has 10500 input tokens',
    'you requested 13500 tokens ... 10500 of input'. -> [n] or []."""
    import re
    for pat in (r"(\d+)\s+in the messages", r"has\s+(\d+)\s+input tokens", r"(\d+)\s+input tokens",
                r"(\d+)\s+tokens?\s+(?:of|from the)\s+input", r"prompt\s+(?:has|of|contains)\s+(\d+)\s+tokens"):
        m = re.search(pat, msg, re.I)
        if m:
            return [int(m.group(1))]
    return []


def make_tracking_r0(R0Client):
    """The benchmark R0 client raises its budget only when a reply comes back EMPTY; a reply cut by the
    token budget (finish_reason 'length', non-empty) was accepted silently. Here such a reply is
    re-requested with a larger budget (bounded by the server's context minus the prompt); a reply that
    still cannot finish is counted in n_len_truncated (verify fails on it), never hidden."""
    class TrackingR0(R0Client):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.gpt5 = bool(getattr(self, "gpt5_dialect", False))
            self.ctx = None if self.gpt5 else R0_CONTEXT
            self.n_len_retries = self.n_len_truncated = self.n_ctx_fit = self.n_empty_final = 0
            self._tlock = threading.Lock()

        def reply(self, *a, **kw):
            out = super().reply(*a, **kw)
            if not (out or "").strip():            # still empty after the client's own budget ladder
                with self._tlock:
                    self.n_empty_final += 1
                _ep_count("r0_empty")
            return out


        def _post_fit(self, body):
            """super()._post, but when the server refuses the request for exceeding its context (prompt +
            completion budget), lower the completion budget to what the context leaves. A budget below 512
            tokens would risk a cut reply: that raises instead of degrading silently."""
            import re
            try:
                return super()._post(body), body
            except RuntimeError as e:
                msg = str(e)
                if "context length" not in msg and "maximum context" not in msg:
                    raise
                nums = prompt_tokens_from_error(msg)
                if not nums or not self.ctx:
                    raise
                room = self.ctx - nums[0] - 16
                if room < 512:
                    raise RuntimeError("R0 prompt of %d tokens leaves %d for the reply (< 512): %s" % (nums[0], room, msg[:200]))
                key = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
                body = dict(body, **{key: room})
                with self._tlock:
                    self.n_ctx_fit += 1
                _ep_count("r0_ctx_fit")
                return super()._post(body), body

        def _post(self, body):
            data, body = self._post_fit(body)
            tries = 0
            while True:
                ch = data["choices"][0]
                content = ((ch.get("message") or {}).get("content") or "").strip()
                if ch.get("finish_reason") != "length" or not content:
                    return data          # finished, or empty (the client's own budget ladder handles it)
                key = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
                cur = int(body.get(key) or 0)
                pt = (data.get("usage") or {}).get("prompt_tokens")
                cap = (self.ctx - int(pt) - 16) if (self.ctx and pt) else max(2 * cur, 8000)
                new = min(2 * cur, cap) if cur else cap
                if tries >= 3 or new <= cur:
                    with self._tlock:
                        self.n_len_truncated += 1
                    _ep_count("r0_len_truncated")
                    return data
                body = dict(body, **{key: new})
                tries += 1
                with self._tlock:
                    self.n_len_retries += 1
                _ep_count("r0_len_retries")
                data, body = self._post_fit(body)
    return TrackingR0


def _parses_as_object(text):
    import re
    t = (text or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return False
    try:
        return isinstance(json.loads(m.group(0)), dict)
    except ValueError:
        return False


def make_floor_judge(Judge):
    class FloorJudge(Judge):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.gpt5 = str(self.model).startswith("gpt-5")
            self.floor = None if self.gpt5 else JUDGE_MIN_TOKENS
            self.n_empty = self.n_unparseable = self.n_retries = self.n_retry_failed = self.n_errors = 0
            self._init_args, self._init_kw = a, dict(kw)
            self._retry = None
            self._lock = threading.Lock()

        def chat_raw(self, system, user, max_tokens):
            return super().chat(system, user, max_tokens)

        def _retry_judge(self):
            with self._lock:
                if self._retry is None:
                    kw = dict(self._init_kw)
                    if kw.get("cache_dir"):
                        kw["cache_dir"] = os.path.join(kw["cache_dir"], "retry_%d" % JUDGE_RETRY_TOKENS)
                    self._retry = FloorJudge.__bases__[0](*self._init_args, **kw)
                    self._retry.chat_raw = self._retry.chat
            return self._retry

        def chat(self, system, user, max_tokens=400):
            budget = max(max_tokens, self.floor) if self.floor else max_tokens
            try:
                out = super().chat(system, user, budget)
            except Exception:
                # a verdict lost to an exception would otherwise vanish if the caller swallows it
                with self._lock:
                    self.n_errors += 1
                _ep_count("judge_error")
                raise
            if not _parses_as_object(out) and not self.gpt5 and JUDGE_RETRY_TOKENS > budget:
                # one re-request at a larger budget (an answer cut by the budget is the usual cause), through a
                # judge with its OWN cache directory, so a cached bad answer is never replayed
                with self._lock:
                    self.n_retries += 1
                _ep_count("judge_retries")
                try:
                    out = self._retry_judge().chat_raw(system, user, JUDGE_RETRY_TOKENS)
                except Exception:                  # e.g. prompt + 8000 over the server context: keep the first answer
                    with self._lock:
                        self.n_retry_failed += 1
                    _ep_count("judge_retry_failed")
            if not (out or "").strip():
                with self._lock:
                    self.n_empty += 1
                _ep_count("judge_empty")
            elif not _parses_as_object(out):
                with self._lock:
                    self.n_unparseable += 1          # the Ledger credits nothing: the episode leaves the reward groups
                _ep_count("judge_unparseable")
            return out
    return FloorJudge


def setup_environment(arm):
    """Must run before sepsim / run_v2 are imported."""
    for k, v in V2FIX.items():
        os.environ[k] = v
    os.environ.pop("SEPSIM_REDRAW", None)
    os.environ["SEPSIM_ARM"] = "sepsim_v2fix" if arm in ("a0", "a2") else "e16d_" + arm
    os.environ["SEPSIM_PLANNER_END"] = "0"
    os.environ.pop("SEPSIM_INTENT_CACHE", None)
    for k in set().union(*ARM_ENV.values()):
        os.environ.pop(k, None)
    os.environ.update(ARM_ENV[arm])
    tree = tree_of(arm)
    if "sepsim" in sys.modules and not os.path.abspath(sys.modules["sepsim"].__file__).startswith(os.path.abspath(tree)):
        raise RuntimeError("sepsim already imported from another tree; one arm family per process")
    for p in (HERE, tree, os.path.join(tree, "scripts"), os.path.join(BENCH, "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)
    sys.modules.setdefault("torchvision", None)
    sys.modules.setdefault("torchaudio", None)


class PlannerLM:
    def __init__(self, path, gpu, nf4=False, dtype="bfloat16", adapter=None, trainable=False,
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
            return
        kw = {"low_cpu_mem_usage": True, "device_map": {"": gpu}}
        if nf4:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, dtype if dtype != "auto" else "bfloat16"),
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        else:
            kw.update(F.dtype_kwarg("auto" if dtype == "auto" else getattr(torch, dtype)))
        self.model = AutoModelForCausalLM.from_pretrained(path, **kw)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=trainable)
        if not trainable:
            self.model.eval()
        cfg = getattr(self.model, "config", None)
        ctx = getattr(cfg, "max_position_embeddings", None) or 32768
        self.context = int(ctx)
        self.budget = min(F.PLANNER_BUDGET, self.context - max_new - F.MARGIN)
        self.adapter = adapter
        self.n_calls = 0

    STOP_RE = None

    def stop_mask(self, gen_ids):
        """1 on the generated tokens that spell the end_session VALUE (true/false), else 0.
        Char span found in the decoded text, mapped to tokens by binary search over prefix decodes
        (exact for byte-level BPE, where single-token decodes need not concatenate to the text).
        Returns None when the field is absent (then no stop credit is given to that generation)."""
        import re
        if PlannerLM.STOP_RE is None:
            PlannerLM.STOP_RE = re.compile(r'"end_session"\s*:\s*"?(true|false)"?', re.I)
        ids = list(gen_ids)
        full = self.tok.decode(ids, skip_special_tokens=False)
        m = PlannerLM.STOP_RE.search(full)
        if not m:
            return None
        a, b = m.start(1), m.end(1)

        def first_token_covering(pos):
            lo, hi = 1, len(ids)          # smallest k with len(decode(ids[:k])) > pos -> token k-1
            while lo < hi:
                mid = (lo + hi) // 2
                if len(self.tok.decode(ids[:mid], skip_special_tokens=False)) > pos:
                    hi = mid
                else:
                    lo = mid + 1
            return lo - 1
        i, j = first_token_covering(a), first_token_covering(b - 1)
        mask = [0] * len(ids)
        for k in range(i, j + 1):
            mask[k] = 1
        return mask

    def eos_ids(self):
        if self.model is None:
            from transformers import GenerationConfig
            e = GenerationConfig.from_pretrained(self.path).eos_token_id
            return set(e if isinstance(e, (list, tuple)) else [e])
        gc = getattr(self.model, "generation_config", None)
        e = getattr(gc, "eos_token_id", None) if gc is not None else None
        if e is None:
            e = self.tok.eos_token_id
        return set(e if isinstance(e, (list, tuple)) else [e])

    def build_prompt(self, it):
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
        prompts = [self.build_prompt(it) for it in items]
        L = max(len(p[2]) for p in prompts)
        pad = self.tok.pad_token_id
        dev = next(self.model.parameters()).device
        input_ids = torch.tensor([[pad] * (L - len(p[2])) + p[2] for p in prompts], dtype=torch.long, device=dev)
        attn = torch.tensor([[0] * (L - len(p[2])) + [1] * len(p[2]) for p in prompts], dtype=torch.long, device=dev)
        kw = dict(max_new_tokens=self.max_new, pad_token_id=pad)
        if t0 and t0 > 0:
            torch.manual_seed(int(items[0]["seed"]))
            kw.update(do_sample=True, temperature=t0, top_p=p0)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, attention_mask=attn, **kw)
        eos = self.eos_ids()
        res = []
        for i, (user_fit, fit, ids) in enumerate(prompts):
            row = out[i][L:].tolist()
            cut = next((j for j, x in enumerate(row) if x in eos), None)
            gen = row[: cut + 1] if cut is not None else row
            while gen and cut is None and gen[-1] == pad and pad not in eos:
                gen.pop()
            self.n_calls += 1
            res.append({"raw": self.tok.decode(gen, skip_special_tokens=True), "prompt_text": user_fit,
                        "prompt_ids": list(ids), "gen_ids": gen,
                        "fit": {**fit, "prompt_tokens": len(ids), "budget": self.budget, "batched": len(items), "backend": "hf",
                                "batch_seed": int(items[0]["seed"]) if (t0 and t0 > 0) else None},
                        "hit_max_new": cut is None and len(gen) >= self.max_new})
        return res

    def field_mask(self, gen_ids, field):
        """1 on the generated tokens that spell the string VALUE of a JSON field (e.g. profile_note),
        located as in stop_mask; None when the field is absent."""
        import re
        ids = list(gen_ids)
        full = self.tok.decode(ids, skip_special_tokens=False)
        m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % re.escape(field), full)
        if not m or m.end(1) <= m.start(1):
            return None
        a, b = m.start(1), m.end(1)

        def first_token_covering(pos):
            lo, hi = 1, len(ids)
            while lo < hi:
                mid = (lo + hi) // 2
                if len(self.tok.decode(ids[:mid], skip_special_tokens=False)) > pos:
                    hi = mid
                else:
                    lo = mid + 1
            return lo - 1
        i, j = first_token_covering(a), first_token_covering(b - 1)
        return [1 if i <= k <= j else 0 for k in range(len(ids))]

    def stop_target(self, gen_ids, mask, want_end):
        """Supervision target for the end_session value: the policy's own prefix up to the value, and the
        value tokens re-encoded with the human's decision (true at the real last message, else false)."""
        import re
        if not mask or 1 not in mask:
            return None
        i = mask.index(1)
        j = len(mask) - 1 - mask[::-1].index(1)
        span = self.tok.decode(list(gen_ids[i:j + 1]), skip_special_tokens=False)
        new = re.sub(r"(?i)true|false", "true" if want_end else "false", span, count=1)
        if new == span and not re.search(r"(?i)true|false", span):
            return None
        ids = self.tok(new, add_special_tokens=False)["input_ids"]
        # gen_len: the length of the whole generation this value belongs to -- the aux loss is normalised per
        # generated token exactly like the GRPO loss (user decision 2026-09-26)
        return {"prefix_ids": list(gen_ids[:i]), "target_ids": list(ids), "want_end": bool(want_end),
                "gen_len": len(gen_ids)}

    def generate(self, system, user, temperature=0.0, top_p=1.0, seed=0):
        import torch
        import fit_prompts as F
        user_fit, fit = F.fit_planner_user(self.tok, system, user, budget=self.budget)
        text = self.tok.apply_chat_template([{"role": "system", "content": system},
                                             {"role": "user", "content": user_fit}],
                                            tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        n = int(enc["input_ids"].shape[1])
        if n > self.budget:
            raise AssertionError("Planner prompt %d > budget %d after fitting" % (n, self.budget))
        enc = enc.to(next(self.model.parameters()).device)
        kw = dict(max_new_tokens=self.max_new, pad_token_id=self.tok.pad_token_id)
        if temperature and temperature > 0:
            torch.manual_seed(seed)
            kw.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        gen = out[0][n:]
        self.n_calls += 1
        return {"raw": self.tok.decode(gen, skip_special_tokens=True), "prompt_text": user_fit,
                "prompt_ids": enc["input_ids"][0].tolist(), "gen_ids": gen.tolist(),
                "fit": {**fit, "prompt_tokens": n, "budget": self.budget},
                "hit_max_new": int(gen.shape[0]) >= self.max_new}


class Task2Env:
    def __init__(self, arm, gpu, planner, judge=None, ditto_path=DITTO, corpus="/home/mzjiang/v5-latency/data.jsonl",
                 batch=False, max_batch=8, implicit_profile=False, fewshot_pool=None, selector="length", task1_only=False):
        self.task1_only = bool(task1_only)     # Task 1 generation only: no R0 agent, no ledger judge
        if arm not in ARM_ENV:
            raise ValueError(arm)
        if arm in V3_ARMS and judge is None:
            raise ValueError("%s needs a goal judge" % arm)
        setup_environment(arm)
        from sepsim import acts, models, planner_prompt as PP
        import run_v2
        from r0_client import R0Client
        from metrics.judge import Judge
        import fit_prompts as F
        import planner_prompt_v3 as V3
        assert (run_v2.ANTILEAK, run_v2.NEARCOPY, run_v2.COPY_SCOPE, run_v2.REDRAW, run_v2.NSAMP,
                run_v2.SELECTOR, run_v2.LENGTH_SELECT, run_v2.POSITION, run_v2.GUARDRAILS, run_v2.END_PROBE) == \
               (True, True, "both", 4, 3, "length", True, "system", False, False), "v2fix did not bind"
        expected = frozenset({"nostopclobber"}) if arm in V3_ARMS + ("pend",) else frozenset()
        assert acts.prior_mode() == expected, "act prior mode wrong for arm %s" % arm
        self.e16 = arm in E16_ARMS
        if self.e16:
            from sepsim import pipeline as _pl
            assert os.path.abspath(models.__file__).startswith(os.path.abspath(TREE_E16)), models.__file__
            assert (models.ROLESTOP, models.INTENT_PROSE, models.NEXTSTEP_INLINE, models.ENDMASK_RETRY) == \
                   (True, False, False, False), "E1.6 speaker switches did not bind"
            assert (run_v2.SELF_JUDGE, run_v2.PLANNER_END, run_v2.T1_SAMPLE, run_v2.ENDGATE,
                    run_v2.AGENDA_CORROB) == (arm == "e16", True, True, False, False), "E1.6 runner switches"
            assert _pl.prior_annotations([{"annotations": {"x": 1}}], 2) == {}, "NO_ANN did not bind"
        self.arm, self.gpu, self.planner, self.judge = arm, gpu, planner, judge
        # Implicit Profile + Speaker few-shot (implicit_profile.py); pend arm only, off by default
        if (implicit_profile or fewshot_pool is not None) and arm != "pend":
            raise ValueError("the Implicit Profile is implemented for the pend arm")
        self.ip, self.fewshot = bool(implicit_profile), fewshot_pool
        if selector not in ("length", "borda"):
            raise ValueError(selector)
        if selector == "borda" and arm != "pend":
            raise ValueError("the length+style selector is implemented for the pend arm")
        self.selector = selector
        self.style_scorer = None
        if selector == "borda":
            import style_select
            self.style_scorer = style_select.StyleScorer()
        if arm == "pend":
            self.system = V3.system_prompt_pend(implicit_profile=self.ip)
            assert '"goal_met"' in self.system and '"last_reply_helpful"' not in self.system
        else:
            self.system = V3.system_prompt_v3() if arm in V3_ARMS else PP.system_prompt()
        # GPU calls (Planner, goal judge, Ditto) are serialised by this lock so episodes can run in
        # threads: each GPU call seeds and generates atomically; R0/ledger HTTP calls overlap freely.
        self.gpu_lock = threading.Lock()
        if arm == "e16":
            assert '"last_reply_helpful"' in self.system and "length_words\": <words if they made THIS move>" in self.system
        if arm == "final":
            assert '"last_reply_helpful"' not in self.system and "inside that move's band" not in self.system
        self.recs = {}
        for l in open(corpus, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                self.recs[r["conversation_id"]] = r
        self.reqs = json.load(open(os.path.join(BENCH, "data/req_shards_v1.json")))
        # Task Agent (R0) and requirement-ledger judge. Default = the benchmark's gpt-5-mini at
        # reasoning_effort=minimal. A substitute endpoint (e.g. gpt-oss-120b on vLLM) is set with
        # R0_BASE_URL/R0_MODEL and JUDGE_BASE_URL/JUDGE_MODEL; gpt-oss has no "minimal", so the
        # effort is set with R0_REASONING_EFFORT / JUDGE_REASONING_EFFORT. All of it is in describe().
        if arm == "pend" and not self.task1_only and os.environ.get("PEND_ALLOW_OTHER_ENDPOINTS") != "1":
            # user decision (option A): R0 and the ledger judge are our own gpt-oss-120b on vLLM
            for k in ("R0_BASE_URL", "JUDGE_BASE_URL"):
                u = os.environ.get(k) or ""
                if not any(h in u for h in ("127.0.0.1:%s" % PEND_PORT, "localhost:%s" % PEND_PORT)):
                    raise RuntimeError("%s must point at the local gpt-oss-120b server (port %s) for the pend arm, got %r"
                                       % (k, PEND_PORT, u))
            for k in ("R0_MODEL", "JUDGE_MODEL"):
                if os.environ.get(k) != "gpt-oss-120b":
                    raise RuntimeError("%s must be gpt-oss-120b for the pend arm (got %r)" % (k, os.environ.get(k)))
        self.r0_effort = os.environ.get("R0_REASONING_EFFORT", "minimal")
        self.judge_effort = os.environ.get("JUDGE_REASONING_EFFORT", "minimal")
        jkey = hashlib.sha256(("%s|%s" % (os.environ.get("JUDGE_MODEL", "default"),
                                          os.environ.get("JUDGE_BASE_URL", "default"))).encode()).hexdigest()[:12]
        self.judge_cache_dir = os.path.join(WORK, "judge_cache_%s" % jkey)      # one cache per judge model+endpoint
        self.ledger_judge = make_floor_judge(Judge)(reasoning_effort=self.judge_effort, verbose=False,
                                                    cache_dir=self.judge_cache_dir)
        self.r0 = make_tracking_r0(R0Client)(reasoning_effort=self.r0_effort)
        if self.e16:
            import ditto_e16
            base_cls = ditto_e16.DittoSpeaker
        else:
            base_cls = models.DittoSpeaker
        FitDitto = F.make_fit_ditto_speaker(base_cls)
        self.speaker = FitDitto(path=ditto_path, gpu=gpu, position=run_v2.POSITION, max_new=F.SPEAKER_MAX_NEW).load()
        assert self.speaker.max_new == F.SPEAKER_MAX_NEW and PLANNER_MAX_NEW == F.PLANNER_MAX_NEW, "generation caps disagree"
        # E1.6 Z1 (T1_SAMPLE): turn 1 is sampled at the speaker checkpoint's own card values
        self.t1_sampling = self.speaker.card_sampling() if self.e16 else None
        # cross-episode dynamic batching (threads submit, one GPU call per batch); off = one by one
        self.planner_batcher = self.speaker_batcher = None
        if batch:
            import batching
            if not hasattr(self.speaker, "say_batch"):
                raise RuntimeError("batching needs the E1.6 Ditto speaker (say_batch)")
            if getattr(planner, "remote", None) is None:
                self.planner_batcher = batching.Batcher(planner.generate_batch, self.gpu_lock,
                                                        key=lambda it: (it["temperature"], it["top_p"]),
                                                        max_batch=max_batch, name="planner-batcher")
            self.speaker_batcher = batching.Batcher(self.speaker.say_batch, self.gpu_lock,
                                                    max_batch=2 * max_batch, name="speaker-batcher")

    def _plan(self, system, user, temperature, top_p, seed):
        item = {"system": system, "user": user, "temperature": temperature, "top_p": top_p, "seed": seed}
        if self.planner_batcher is not None:
            return self.planner_batcher(item)
        if getattr(self.planner, "remote", None) is not None:
            return self.planner.generate_batch([item])[0]     # vLLM server: no local GPU, no lock
        # unbatched: the SAME function as the batched path, one item at a time (identical code, no padding)
        with self.gpu_lock:
            return self.planner.generate_batch([item])[0]

    def _plan_many(self, items):
        if self.planner_batcher is not None:
            return self.planner_batcher.map(items)
        if getattr(self.planner, "remote", None) is not None:
            return self.planner.generate_batch(items)         # concurrent requests to the vLLM server
        return [self._plan(it["system"], it["user"], it["temperature"], it["top_p"], it["seed"]) for it in items]

    def _say_many(self, reqs):
        """-> [(text, ended, fit, hit_max_new)] in request order."""
        if self.speaker_batcher is not None:
            return self.speaker_batcher.map(reqs)
        if hasattr(self.speaker, "say_batch"):
            # unbatched: the SAME function as the batched path, one request at a time (cap hits recorded)
            out = []
            for r in reqs:
                with self.gpu_lock:
                    out.append(self.speaker.say_batch([r])[0])
            return out
        out = []                       # legacy arms (v2fix tree Ditto): cap hit not measurable
        for r in reqs:
            with self.gpu_lock:
                txt, e = self.speaker.say(r["scenario_text"], r["block"], r["hist_u"], r["hist_a"], r["turn"],
                                          seed=r["seed"], temperature=r["temperature"], top_p=r["top_p"],
                                          avoid=r["avoid"], reject_template=r["reject_template"],
                                          reject_reuse=r["reject_reuse"])
                fit = self.speaker.last_fit
            out.append((txt, e, fit, None))
        return out

    def describe(self):
        import fit_prompts as F
        return {"arm": self.arm, "planner": self.planner.path, "planner_adapter": self.planner.adapter,
                "planner_backend": "vllm" if getattr(self.planner, "remote", None) is not None else "hf",
                "planner_vllm_url": getattr(getattr(self.planner, "remote", None), "url", None),
                "planner_budget": self.planner.budget, "speaker_budget": F.SPEAKER_BUDGET,
                "system_prompt_sha256": hashlib.sha256(self.system.encode()).hexdigest(),
                "goal_judge": (self.judge.model_path, self.judge.adapter) if self.judge else None,
                "r0_model": self.r0.model, "ledger_judge_model": self.ledger_judge.model,
                "r0_base_url": os.environ.get("R0_BASE_URL", "default(api.openai.com)"),
                "judge_base_url": os.environ.get("JUDGE_BASE_URL", "default(api.openai.com)"),
                # what is actually SENT: the benchmark clients only send reasoning_effort in the gpt-5
                # dialect; for other models (gpt-oss on vLLM) the server's default effort applies
                "r0_reasoning_effort": self.r0_effort if getattr(self.r0, "gpt5_dialect", True) else "server default (not sent)",
                "judge_reasoning_effort": self.judge_effort if self.ledger_judge.gpt5 else "server default (not sent)",
                "ledger_judge_min_tokens": self.ledger_judge.floor, "ledger_judge_retry_tokens": JUDGE_RETRY_TOKENS,
                "act_prior": os.environ["SEPSIM_ACT_PRIOR"], "v2fix": V2FIX, "t_max": T_MAX,
                "tree": tree_of(self.arm), "arm_env": ARM_ENV[self.arm], "t1_sampling": self.t1_sampling,
                "implicit_profile": self.ip, "fewshot": self.fewshot.describe() if self.fewshot else None,
                "selector": self.selector, "speaker_max_new": self.speaker.max_new, "planner_max_new": self.planner.max_new,
                "judge_cache_dir": getattr(self, "judge_cache_dir", None), "task1_only": self.task1_only,
                "endpoint_bypass": os.environ.get("PEND_ALLOW_OTHER_ENDPOINTS") == "1",
                "batching": None if self.planner_batcher is None else {
                    "planner": self.planner_batcher.max_batch, "speaker": self.speaker_batcher.max_batch}}

    def _session(self, conversation_id, seed, replicate=0, planner_temperature=0.0, planner_top_p=1.0,
                 record_generation=False, with_ledger=True):
        """Per-conversation state and the speak/respond closures, shared by Task 2 (run_episode) and
        Task 1 (task1_generate) so both run exactly the same Planner + Ditto + Selector code."""
        from sepsim import agenda as AG, persona as P, pipeline, planner_prompt as PP, state, stopping
        import run_v2
        from r0_client import Ledger
        import planner_prompt_v3 as V3
        from task2_episode import run_episode
        arm, planner, speaker = self.arm, self.planner, self.speaker
        rec = self.recs[conversation_id]
        rid = rec["record_id"]
        scenario = rec["scenario"]
        sc_text = pipeline.scenario_text(rec)
        sid = "%s#s%d" % (rid, seed)
        rng = random.Random(pipeline.seed_for(sid, 0))
        led = stopping.StoppingLedger(scenario)
        ag = AG.Agenda(AG.terms_of(" ".join([str((scenario.get("goal") or {}).get("topic", "")),
                                             str((scenario.get("goal") or {}).get("context", ""))]), 8))
        ledger = Ledger(self.reqs[conversation_id]["req"], judge=self.ledger_judge) if with_ledger else None
        S = {"hist_u": [], "hist_a": [], "prev_block": state.d0(P.initial_stage(scenario.get("goal"))),
             "block": None, "cov": [], "mode": "task2", "ip_notes": [], "ip_ctx": None, "ip_last_note": None, "resp_t": {}}
        import implicit_profile as IP

        def speak(t):
            hist_u, hist_a = S["hist_u"], S["hist_a"]
            gs = None
            v3 = arm in V3_ARMS
            if v3:
                if hist_a:
                    with self.gpu_lock:
                        gs = self.judge.assess(sc_text, hist_u, hist_a)
                else:
                    gs = {"status": "NOT ASSESSED", "unmet": []}
                up = V3.user_prompt_v3(scenario, S["prev_block"], hist_u, hist_a, t, led,
                                       {"status": gs["status"], "unmet": gs.get("unmet", [])}, prev_ann={})
            elif arm == "pend":
                ip_sec = ""
                if self.ip:
                    ctx = S["ip_ctx"] if S["mode"] == "task1" else ({"mode": "task2"} if t >= 2 else None)
                    ip_sec = IP.render_planner_sections(S["ip_notes"], ctx)
                up = V3.user_prompt_pend(scenario, S["prev_block"], hist_u, hist_a, t, led, prev_ann={},
                                         ip_sections=ip_sec)
            else:
                up = PP.user_prompt(scenario, S["prev_block"], hist_u, hist_a, t, ledger=led,
                                    prev_ann={}, agenda_view=ag.render(), p_end=None)
            pseed = int(hashlib.sha256(("%s|%d|%d|%d" % (sid, t, replicate, 7)).encode()).hexdigest()[:8], 16)
            _t0 = time.time()
            g = self._plan(self.system, up, planner_temperature, planner_top_p, pseed)
            planner_s = time.time() - _t0            # wall time incl. waiting for a batch / the GPU lock
            raw = g["raw"]
            self_judge = None
            if arm == "e16" and t >= 2:
                # E1.6 SEPSIM_SELF_JUDGE, line for line (run_v2.py @ cf19400): the Planner's own
                # rating of the last reply feeds the ledger BEFORE read_plan, so STOP_LEDGER sees it
                _d = state.json_of(raw) or {}
                _h = _d.get("last_reply_helpful")
                _q = _d.get("last_reply_quality")
                self_judge = {"helpful": _h if isinstance(_h, bool) else None,
                              "dataset_quality": _q if _q in stopping.QUALITY_ORDER else None}
                led.judge(self_judge["helpful"], self_judge["dataset_quality"])
            if v3:
                fields, diag, end_session = V3.read_plan_v3(raw, t, scenario, rng, led)
            elif arm == "pend" and g["hit_max_new"]:
                # a cut output is not a plan: it is not read at all (no stopping-ledger gain, no act-RNG draw)
                fields, diag, end_session = None, {"cut_by_max_new": True}, False
            elif arm == "pend":
                fields, diag, end_session = V3.read_plan_pend(raw, t, scenario, rng, led)
            elif arm == "e16":
                fields, diag = PP.read_plan(raw, t, scenario, rng, led, agenda_open=False)
                end_session = None
            else:
                fields, diag = PP.read_plan(raw, t, scenario, rng, led)
                end_session = None
            if g["hit_max_new"] and fields is not None:
                # a generation cut by the token cap is not a decision: treated as unparsed (no end, no note)
                diag = dict(diag or {}, cut_by_max_new=True)
                fields, end_session = None, False
            unparsed = fields is None
            if unparsed:
                fields = {"move": "Other", "act": "other"}
            ended = bool(end_session) if (v3 or arm == "pend") else bool(state.ends_session(fields))
            note, entry = "", ""
            if self.ip and t >= 2:
                note = "" if unparsed else IP.clean_note(fields.get("profile_note"))
                if note and S["ip_last_note"] is not None and IP.norm_text(S["ip_last_note"]) == IP.norm_text(note):
                    note = ""                           # an exact repeat of the last note adds nothing
                elif note:
                    S["ip_last_note"] = note
                ctx = S["ip_ctx"] if S["mode"] == "task1" else None
                measured = IP.measured_diff(ctx["pred_prev"], ctx["gold_prev"]) if ctx else ""
                entry = IP.profile_entry(t - 1, measured, note)
                if entry:
                    S["ip_notes"].append(entry)
            base = {"planner_prompt": g["prompt_text"], "planner_fit": g["fit"], "planner_raw": raw,
                    "planner_hit_max_new": g["hit_max_new"], "planner_diag": diag, "planner_s": round(planner_s, 2),
                    "planner_unparsed": unparsed, "ended_planner": ended,
                    "move": fields.get("move", ""), "act": fields.get("act", ""),
                    "stop_rule": fields.get("stop_rule", "none"), "goal_status": gs, "self_judge": self_judge,
                    "goal_met": fields.get("goal_met"), "still_wanted": fields.get("still_wanted"),
                    "profile_note": note if self.ip else None, "profile_entry": entry if self.ip else None,
                    "ip_notes_n": len(S["ip_notes"]) if self.ip else None,
                    "ledger_before": {"turns": led.turns, "gain_trace": list(led.gain_trace)}}
            if record_generation:
                sm, nm = rl_masks(planner, g, unparsed, diag, t)
                base["planner_gen"] = {"prompt_ids": g["prompt_ids"], "gen_ids": g["gen_ids"], "stop_mask": sm,
                                       "note_mask": nm, "hit_max_new": g["hit_max_new"],
                                       "gen_logprobs": g.get("gen_logprobs"), "gen_adapter": g.get("gen_adapter"),
                                       "temperature": planner_temperature, "top_p": planner_top_p, "seed": pseed}
            if v3 and ended:
                return {**base, "planner_stop": True, "user": ""}
            if unparsed:
                block = S["prev_block"]
            elif v3:
                block = V3.speaker_block_v3(fields, gs)
            elif arm == "pend":
                block = V3.speaker_block_pend(fields, last_line=False)
            else:
                block = PP.render_block(fields, ag.render())
            # S["block"] is the Planner's own state for the next turn: never the Speaker-only additions
            # (the last-message line, the notes, the examples)
            S["block"] = block
            if arm == "pend":
                spk = block if unparsed else V3.speaker_block_pend(fields, last_line=True)
                _t0 = time.time()
                res = self._pend_generate(base, block, fields, S, t, sid, sc_text, hist_u, hist_a,
                                          conversation_id, scenario, unparsed, spk_block=spk)
                res["speaker_s"] = round(time.time() - _t0, 2)   # all candidates, redraws, selector
                return res
            examples = []
            # ---- generation block: line for line the same as rollout_stop_sft.py / run_v2 ----
            ANTILEAK, NEARCOPY = run_v2.ANTILEAK, run_v2.NEARCOPY
            avoid = hist_u if run_v2.GUARDRAILS else None
            prior = list(hist_u) + (list(hist_a) if run_v2.COPY_SCOPE == "both" else [])
            if run_v2.GUARDS_ON:
                avoid = prior
            # E1.6 Z1 (T1_SAMPLE): on turn 1 every draw, the first included, is sampled at the
            # speaker card's values and the pick is uniform among guard survivors (seeded)
            z1 = self.e16 and t == 1
            T_G, P_G = self.t1_sampling if z1 else (0.0, 1.0)
            T_S, P_S = self.t1_sampling if z1 else (0.7, 0.9)
            def req(seed, temp, top_p):
                return {"scenario_text": sc_text, "block": block, "hist_u": list(hist_u), "hist_a": list(hist_a),
                        "turn": t, "seed": seed, "temperature": temp, "top_p": top_p, "avoid": avoid,
                        "reject_template": ANTILEAK, "reject_reuse": NEARCOPY}
            first = self._say_many([req(pipeline.seed_for(sid, t), T_G, P_G)] +
                                   [req(pipeline.seed_for(sid, t, k + 1), T_S, P_S) for k in range(run_v2.NSAMP)])
            fit0 = first[0][2]
            cands, flags = [x[0] for x in first], [x[1] for x in first]
            reasons, n_extra, eligible = None, 0, None
            if run_v2.GUARDS_ON:
                def reason_of(c):
                    r = run_v2.guard_reason(c, prior)
                    if not r and examples and IP.copies_example(c, examples):
                        r = "fewshot_copy"      # a run of >= COPY_NGRAM words from another person's message
                    if not r and (self.ip or examples) and IP.leaks_scaffold(c):
                        r = "template"          # the new block lines leaked into the message
                    return r
                reasons = [reason_of(c) for c in cands]
                while all(reasons) and n_extra < run_v2.REDRAW:
                    k = run_v2.NSAMP + n_extra
                    sx, ex, _, _ = self._say_many([req(pipeline.seed_for(sid, t, k + 1), T_S, P_S)])[0]
                    cands.append(sx)
                    flags.append(ex)
                    reasons.append(reason_of(sx))
                    n_extra += 1
                eligible = [i for i, r in enumerate(reasons) if not r] or None
            idx = run_v2.choose(cands, fields.get("length_words"), None, eligible)
            if z1:
                pool = eligible if eligible else list(range(len(cands)))
                idx = random.Random(pipeline.seed_for(sid, t, 97)).choice(pool)
            return {**base, "user": cands[idx], "ended_speaker": bool(flags[idx]), "block": block, "t1_sampled": z1,
                    "guard_reasons": reasons, "guard_extra": n_extra,
                    "no_survivor": reasons is not None and eligible is None,
                    "selected_index": idx, "n_candidates": len(cands), "speaker_fit": fit0,
                    "candidates": list(cands), "candidates_ended": [bool(f) for f in flags]}

        def respond(t, text):
            S["hist_u"].append(text)
            msgs = []
            for i, u in enumerate(S["hist_u"]):
                msgs.append({"role": "user", "content": u})
                if i < len(S["hist_a"]):
                    msgs.append({"role": "assistant", "content": S["hist_a"][i]})
            _t0 = time.time()
            reply = self.r0.reply(msgs)
            _t1 = time.time()
            prev_reply = S["hist_a"][-1] if S["hist_a"] else ""
            S["hist_a"].append(reply)
            ledger.update(t, text, reply)
            S["resp_t"][t] = {"r0_s": round(_t1 - _t0, 2), "ledger_s": round(time.time() - _t1, 2)}
            led.observe({}, reply, prev_reply)
            ag.retire_satisfied(reply, {})
            if AG.looks_like_new_offer(reply, prev_reply):
                ag.reset_on_new_offer()
            S["prev_block"] = S["block"]
            S["cov"].append((t, (round(ledger.coverage(), 4), bool(ledger.complete()))))
            return reply

        return {"S": S, "speak": speak, "respond": respond, "ledger": ledger, "led": led, "ag": ag,
                "rid": rid, "scenario": scenario}


    def _pend_generate(self, base, block, fields, S, t, sid, sc_text, hist_u, hist_a, cid, scenario, unparsed,
                       spk_block=None):
        """pend generation stage. Same guards and schedule as the other arms (greedy + NSAMP samples at
        (0.7, 0.9); on turn 1 all sampled at the Ditto card, E1.6 Z1; up to REDRAW extra draws when every
        candidate fails), plus (user design, 2026-09-25):
          * each candidate slot gets its OWN few-shot examples (variant = slot), so the Speaker prompts differ;
          * a candidate whose text equals an earlier candidate of the turn is 'duplicate' and that slot alone
            is redrawn (new seed, new examples) up to DUP_REDRAW times (exact identity; no threshold);
          * copying >= COPY_NGRAM words of an example is 'fewshot_copy'; our block headers in the text is
            'template';
          * selection: length + style Borda (selector='borda') or the E1.6 rule (selector='length').
        The Planner's state (S['block']) is `block`; the Speaker-only lines go to the Speaker prompts only."""
        import implicit_profile as IP
        import style_select as SS
        from sepsim import pipeline
        import run_v2
        assert run_v2.GUARDS_ON, "v2fix guards must be on"
        ANTILEAK, NEARCOPY = run_v2.ANTILEAK, run_v2.NEARCOPY
        prior = list(hist_u) + (list(hist_a) if run_v2.COPY_SCOPE == "both" else [])
        avoid = prior
        z1 = t == 1
        T_G, P_G = self.t1_sampling if z1 else (0.0, 1.0)
        T_S, P_S = self.t1_sampling if z1 else (0.7, 0.9)
        notes = list(S["ip_notes"]) if self.ip else []
        persona = scenario.get("persona")
        state_block, block = block, (spk_block if spk_block is not None else block)   # block = Speaker base

        def examples_for(variant):
            return self.fewshot.select(cid, persona, t, variant=variant) if self.fewshot else []

        def block_for(ex):
            # an unparsed turn keeps the Planner's previous block, and still gets the notes and examples
            if not (notes or ex):
                return block
            return block + IP.speaker_lines(notes, ex)

        def req(seed, temp, top_p, blk):
            return {"scenario_text": sc_text, "block": blk, "hist_u": list(hist_u), "hist_a": list(hist_a),
                    "turn": t, "seed": seed, "temperature": temp, "top_p": top_p, "avoid": avoid,
                    "reject_template": ANTILEAK, "reject_reuse": NEARCOPY}

        n0 = 1 + run_v2.NSAMP
        slot_ex = [examples_for(j) for j in range(n0)]
        slot_blk = [block_for(e) for e in slot_ex]
        outs = self._say_many([req(pipeline.seed_for(sid, t), T_G, P_G, slot_blk[0])] +
                              [req(pipeline.seed_for(sid, t, j), T_S, P_S, slot_blk[j]) for j in range(1, n0)])
        cands = [o[0] for o in outs]
        flags = [o[1] for o in outs]
        fits = [o[2] for o in outs]
        hits = [o[3] for o in outs]

        def all_examples():
            seen, out = set(), []
            for ex in slot_ex:
                for e in ex:
                    if e["text"] not in seen:
                        seen.add(e["text"])
                        out.append(e)
            return out

        def reason_of(i):
            c = cands[i]
            if hits[i]:
                return "max_new"             # cut by the Speaker's token cap: never emitted
            r = run_v2.guard_reason(c, prior)
            exs = all_examples()
            if not r and exs and IP.copies_example(c, exs):
                r = "fewshot_copy"
            if not r and (self.ip or self.fewshot) and IP.leaks_scaffold(c):
                r = "template"
            if not r and IP.duplicate_of(c, cands[:i]):
                r = "duplicate"
            return r

        reasons = [reason_of(i) for i in range(len(cands))]
        dup_redraws = 0
        for rnd in range(1, run_v2.REDRAW + 1):          # the same redraw budget as E1.6's all-fail rule
            dups = [i for i, x in enumerate(reasons) if x == "duplicate"]
            if not dups:
                break
            for i in dups:
                slot_ex[i] = examples_for(i + n0 * rnd)
                slot_blk[i] = block_for(slot_ex[i])
            new = self._say_many([req(pipeline.seed_for(sid, t, 100 + 10 * rnd + i), T_S, P_S, slot_blk[i]) for i in dups])
            for i, o in zip(dups, new):
                cands[i], flags[i], fits[i], hits[i] = o[0], o[1], o[2], o[3]
            dup_redraws += len(dups)
            reasons = [reason_of(i) for i in range(len(cands))]
        n_extra = 0
        while all(reasons) and n_extra < run_v2.REDRAW:
            j = n0 + n_extra
            # every candidate failed: redraw WITHOUT examples, so a Speaker that copies the examples
            # (seen in the smoke: all 8 first-turn candidates copied a prompt-like example) can still pass
            ex = []
            blk = block_for(ex)
            o = self._say_many([req(pipeline.seed_for(sid, t, j + 1), T_S, P_S, blk)])[0]
            cands.append(o[0])
            flags.append(o[1])
            fits.append(o[2])
            hits.append(o[3])
            slot_ex.append(ex)
            slot_blk.append(blk)
            reasons.append(reason_of(len(cands) - 1))
            n_extra += 1
        eligible = [i for i, x in enumerate(reasons) if not x] or None
        # no survivor: choose among the candidates that were at least produced whole and are not blank;
        # a capped (cut) text is never emitted unless nothing else exists (then recorded, verify FAILs)
        pool = eligible
        fallback = None
        if eligible is None:
            whole = [i for i in range(len(cands)) if not hits[i]]
            pool = [i for i in whole if (cands[i] or "").strip()] or whole or None
            fallback = "whole_nonblank" if pool and any((cands[i] or "").strip() for i in pool) else \
                ("whole_blank" if pool else "all_capped")
        sel = None
        if self.selector == "borda":
            own = S["mode"] == "task1" and t >= 2
            refs = list(hist_u) if own else [e["text"] for e in all_examples()]
            idx, sel = SS.select(cands, pool, fields.get("length_words"), refs, self.style_scorer)
            sel["refs"] = "own_real_messages" if own else ("fewshot" if refs else "none")
        else:
            idx = run_v2.choose(cands, fields.get("length_words"), None, pool)
            if z1:
                idx = random.Random(pipeline.seed_for(sid, t, 97)).choice(pool or list(range(len(cands))))

        def fit_tokens(f):
            if not isinstance(f, dict):
                return -1
            return f.get("final_tokens") if f.get("compacted") else f.get("original_tokens", -1)
        worst = max(range(len(fits)), key=lambda i: fit_tokens(fits[i]))
        return {**base, "user": cands[idx], "ended_speaker": bool(flags[idx]), "block": state_block, "t1_sampled": z1,
                "speaker_block_base": block if block != state_block else None,
                "guard_reasons": reasons, "guard_extra": n_extra, "dup_redraws": dup_redraws,
                "no_survivor": eligible is None, "no_survivor_fallback": fallback,
                "emitted_capped": bool(hits[idx]), "selected_index": idx, "n_candidates": len(cands),
                "speaker_fit": fits[worst], "speaker_fits": list(fits), "speaker_hit_max_new": hits, "selection": sel,
                "candidates": list(cands), "candidates_ended": [bool(f) for f in flags],
                "fewshot": [[[e["cid"], e["t"]] for e in ex] for ex in slot_ex] if self.fewshot else None,
                "speaker_block_selected": slot_blk[idx] if slot_blk[idx] != block else None}

    def run_episode(self, conversation_id, seed, replicate=0, planner_temperature=0.0, planner_top_p=1.0,
                    record_generation=False):
        from task2_episode import run_episode
        arm, planner = self.arm, self.planner
        ss = self._session(conversation_id, seed, replicate, planner_temperature, planner_top_p, record_generation)
        S, speak, respond, ledger, rid = ss["S"], ss["speak"], ss["respond"], ss["ledger"], ss["rid"]
        # e16: E1.6 SEPSIM_PLANNER_END -- the Planner's Complete act ends the episode after the
        # closing message it asked for (emitted, no assistant reply). v3 arms exit silently instead.
        if self.task1_only:
            raise RuntimeError("this Task2Env was built for Task 1 only (no R0 / ledger judge)")
        orphans0 = orphan_incidents()
        episode_begin()
        try:
            ep = run_episode(T_MAX, None, speak, respond, planner_end=(arm in EMIT_END_ARMS))
            cov_final, complete_final, ledger_dict = round(ledger.coverage(), 4), ledger.complete(), ledger.as_dict()
        finally:
            counts = episode_end()
        if orphan_incidents() != orphans0:
            raise RuntimeError("an R0/judge incident was counted outside any episode thread while %s ran: per-episode "
                               "attribution is broken, so no episode can be marked clean" % conversation_id)
        capped = sum(1 for s in ep["trace"] if s.get("emitted_capped"))
        compacted = sum(1 for s in ep["trace"] if step_compacted(s))
        clean = episode_clean(counts) and capped == 0 and compacted == 0 and ep["emitted_user_turns"] > 0
        after, last = dict(S["cov"]), (0.0, False)
        for step in ep["trace"]:
            last = after.get(step["t"], last)
            step["coverage_after"], step["complete_after"] = last
            step.update(S["resp_t"].get(step["t"], {}))
        return {"conversation_id": conversation_id, "record_id": rid, "seed": seed, "arm": arm,
                "replicate": replicate, "speaker_kind": "ditto", "planner_path": planner.path,
                "planner_adapter": planner.adapter, "planner_temperature": planner_temperature,
                "emitted_user_turns": ep["emitted_user_turns"], "decision_steps": ep["decision_steps"],
                "end_kind": ep["end_kind"], "turns": ep["emitted_user_turns"], "stop_kind": ep["end_kind"],
                "ended_by_token": ep["end_kind"] != "t_max",
                "coverage": cov_final, "complete": complete_final,
                "n_req": len(self.reqs[conversation_id]["req"]), "ledger": ledger_dict, "trace": ep["trace"],
                "ledger_judge_empty_total": self.ledger_judge.n_empty,
                "ledger_judge_unparseable_total": self.ledger_judge.n_unparseable,
                "r0_empty_retries_total": getattr(self.r0, "n_empty_retries", None),
                "r0_len_retries_total": self.r0.n_len_retries, "r0_len_truncated_total": self.r0.n_len_truncated,
                "r0_ctx_fit_total": self.r0.n_ctx_fit,
                # this episode's own incidents; an unclean episode never enters a reward group
                "episode_counters": counts, "emitted_capped_steps": capped, "compacted_steps": compacted,
                "clean": clean,
                "human_turns": self.human_turns(conversation_id)}

    def task1_generate(self, conversation_id, seed=0, keep_prompts=False):
        """Task 1 (teacher-forced) generations for one REAL conversation, in the benchmark's generations
        schema (tools/score_method.py): one row per real user turn with greedy (the selected candidate),
        samples (the other candidates), greedy_ended under M2 (Speaker blank at t OR the Planner's end
        decided at t-1), samples_ended = the Speaker flags (E1.6 convention). The history at turn t is always the real one; the Planner's own state
        carries over, as in run_v2. Planner at temperature 0. No R0, no ledger."""
        from sepsim import agenda as AG, pipeline
        import implicit_profile as IP
        ss = self._session(conversation_id, seed, 0, 0.0, 1.0, False, with_ledger=False)
        S, speak, led, ag = ss["S"], ss["speak"], ss["led"], ss["ag"]
        S["mode"] = "task1"
        rec = self.recs[conversation_id]
        users, agents = pipeline.split_messages(rec)
        real = [u["text"] for u in users]
        preds = []
        goal = rec["scenario"].get("goal") or {}
        rows = []
        # M2 (user decision D7): the Planner's end at turn t makes message t the LAST one, i.e. the person
        # sends no message t+1 -> the benchmark END flag (the person stops INSTEAD of writing) is set at row
        # t+1, and at the K+1 probe for an end decided at the real last turn n. A blank Speaker message at
        # turn t is END at row t itself (the benchmark convention). The raw decisions stay in the rows.
        prev_decision = False
        for t in range(1, len(users) + 1):
            # only message t-1 (prediction and real text) may inform turn t
            S["ip_ctx"] = IP.task1_context(real[: t - 1], preds, t) if self.ip else None
            st = speak(t)
            if st.get("planner_stop"):
                raise RuntimeError("silent Planner exit has no Task 1 utterance; use an emit-end arm")
            if step_compacted(st):
                raise RuntimeError("Task 1 %s turn %d: a prompt was compacted (history dropped)" % (conversation_id, t))
            cands, ends, idx = st["candidates"], st["candidates_ended"], st["selected_index"]
            samples = [c for i, c in enumerate(cands) if i != idx]
            s_blank = [bool(e) for i, e in enumerate(ends) if i != idx]
            rows.append({"record_id": rec["record_id"], "conversation_id": conversation_id, "turn_index": t,
                         "is_first_turn": t == 1, "discipline": goal.get("discipline", "unknown"),
                         "intent_variant": "pend_" + os.path.basename(str(self.planner.path).rstrip("/")),
                         "greedy": st["user"],
                         "greedy_ended": bool(st["ended_speaker"]) or prev_decision,
                         "end_mapping": "M2", "ended_by_prev_decision": prev_decision,
                         "speaker_ended": bool(st["ended_speaker"]), "planner_ends_session": bool(st["ended_planner"]),
                         "samples": samples, "samples_ended": s_blank,
                         "samples_speaker_ended": s_blank, "profile": st.get("block"),
                         "move": st.get("move"), "act": st.get("act"), "goal_met": st.get("goal_met"),
                         "planner_unparsed": st.get("planner_unparsed"), "planner_hit_max_new": st.get("planner_hit_max_new"),
                         "planner_diag": st.get("planner_diag"),
                         "planner_fit": st.get("planner_fit"), "speaker_fit": st.get("speaker_fit"),
                         "speaker_fits": st.get("speaker_fits"), "emitted_capped": st.get("emitted_capped"),
                         "guard_no_survivor": st.get("no_survivor"), "no_survivor_fallback": st.get("no_survivor_fallback"),
                         "planner_adapter": self.planner.adapter,
                         "profile_note": st.get("profile_note"), "profile_entry": st.get("profile_entry"),
                         "ip_notes_n": st.get("ip_notes_n"),
                         "fewshot": st.get("fewshot"), "speaker_hit_max_new": st.get("speaker_hit_max_new"),
                         "guard_reasons": st.get("guard_reasons"), "dup_redraws": st.get("dup_redraws"),
                         "selected_index": st.get("selected_index"), "selection": st.get("selection"),
                         "planner_prompt_tokens": (st.get("planner_fit") or {}).get("prompt_tokens")})
            if keep_prompts:
                rows[-1]["planner_prompt"] = st["planner_prompt"]
            prev_decision = bool(st["ended_planner"])
            preds.append(st["user"])
            # teacher forcing: the REAL message and the REAL assistant reply enter the history
            S["hist_u"].append(users[t - 1]["text"])
            if t - 1 < len(agents):
                prev = S["hist_a"][-1] if S["hist_a"] else ""
                reply = agents[t - 1]["text"]
                S["hist_a"].append(reply)
                led.observe({}, reply, prev)
                ag.retire_satisfied(reply, {})
                if AG.looks_like_new_offer(reply, prev):
                    ag.reset_on_new_offer()
            S["prev_block"] = S["block"] if S["block"] is not None else S["prev_block"]
        # K+1 probe (the benchmark's termination measurement): after the real person's last message and
        # whatever reply followed it, one more turn. M2: ended = the Planner's decision at the real last
        # turn n (message n was the close) OR a blank Speaker message at K+1. The K+1 turn's own Planner
        # decision is recorded (ended_planner_k1) but is not the END flag.
        k = len(users)
        S["ip_ctx"] = IP.task1_context(real[:k], preds, k + 1) if self.ip else None
        st = speak(k + 1)
        if st.get("planner_stop"):
            raise RuntimeError("silent Planner exit has no Task 1 utterance; use an emit-end arm")
        if step_compacted(st):
            raise RuntimeError("Task 1 %s K+1: a prompt was compacted (history dropped)" % conversation_id)
        blank = bool(st["ended_speaker"])
        k1 = {"record_id": rec["record_id"], "conversation_id": conversation_id, "gold_k": k, "turn_index": k + 1,
              "intent_variant": rows[0]["intent_variant"] + "_k1" if rows else "k1",
              "end_mapping": "M2", "ended_by_decision_at_n": prev_decision,
              "ended_speaker": blank, "ended_planner_k1": bool(st["ended_planner"]),
              "ended_empty": not (st["user"] or "").strip(),
              "ended": prev_decision or blank,
              "greedy": st["user"], "greedy_ended": prev_decision or blank,
              "planner_move": st.get("move"), "planner_act": st.get("act"), "goal_met": st.get("goal_met"),
              "planner_hit_max_new": st.get("planner_hit_max_new"), "planner_unparsed": st.get("planner_unparsed"),
              "planner_diag": st.get("planner_diag"),
              "planner_prompt_tokens": (st.get("planner_fit") or {}).get("prompt_tokens"),
              "planner_fit": st.get("planner_fit"),
              "speaker_fit": st.get("speaker_fit"), "speaker_fits": st.get("speaker_fits"),
              "speaker_hit_max_new": st.get("speaker_hit_max_new"), "emitted_capped": st.get("emitted_capped"),
              "guard_reasons": st.get("guard_reasons"), "selected_index": st.get("selected_index"),
              "fewshot": st.get("fewshot")}
        return rows, k1

    def human_turns(self, conversation_id):
        """Number of messages the real person sent in this conversation (the Task 2 turn target)."""
        from sepsim import pipeline
        users, _ = pipeline.split_messages(self.recs[conversation_id])
        return len(users)

    def task1_prompts(self, conversation_id):
        """Greedy teacher-forced pass (current policy) -> per turn the exact Planner user prompt, so a
        Task 1 training group can sample G decisions from the same state the Planner would be in. The pass
        is the full Task 1 generation (the Implicit Profile needs the Speaker's predictions), i.e. exactly
        what the Task 1 evaluation runs."""
        if self.arm != "pend":
            raise ValueError("task1_prompts is implemented for the pend arm")
        rows, _ = self.task1_generate(conversation_id, keep_prompts=True)
        n = len(rows)
        return [{"t": r["turn_index"], "n_real": n, "real_final": r["turn_index"] == n, "user_prompt": r["planner_prompt"],
                 "planner_fit": r.get("planner_fit"), "emitted_capped": bool(r.get("emitted_capped")),
                 "planner_hit_max_new": bool(r.get("planner_hit_max_new"))} for r in rows]

    def task1_sample(self, conversation_id, t, user_prompt, real_final, G, temperature, top_p, seed):
        """G sampled Planner decisions at one real turn t >= 2 (turn 1 cannot end, so it teaches nothing about
        stopping); reward 1 if end_session == (message t was the person's last), else 0. A sample whose plan
        was not parsed, was cut by the token cap, or has no valid end_session value is not a decision: reward
        0, no stop mask (so no stop credit), never used as the supervision example. Returned in the rollout
        schema (one step with planner_gen each)."""
        if self.arm != "pend":
            raise ValueError("task1_sample is implemented for the pend arm")
        if t < 2:
            raise ValueError("Task 1 stop groups start at turn 2 (turn 1 cannot end)")
        from sepsim import stopping
        import planner_prompt_v3 as V3
        scenario = self.recs[conversation_id]["scenario"]
        out = []
        seeds = [int(hashlib.sha256(("t1s|%s|%d|%d|%d" % (conversation_id, t, g, seed)).encode()).hexdigest()[:8], 16)
                 for g in range(G)]
        gens = self._plan_many([{"system": self.system, "user": user_prompt, "temperature": temperature,
                                 "top_p": top_p, "seed": sd} for sd in seeds])
        for g in range(G):
            pseed, gen = seeds[g], gens[g]
            if (gen.get("fit") or {}).get("compacted"):
                raise RuntimeError("Task 1 group %s t%d: Planner prompt compacted (history dropped)" % (conversation_id, t))
            if gen["hit_max_new"]:
                fields, diag, end = None, {"cut_by_max_new": True}, False
            else:
                fields, diag, end = V3.read_plan_pend(gen["raw"], t, scenario, random.Random(pseed),
                                                      stopping.StoppingLedger(scenario))
            unparsed = fields is None
            sm, nm = rl_masks(self.planner, gen, unparsed, diag, t)
            # a decision = parsed, not capped, a valid end_session value, t >= 2 (the reward does not depend on
            # whether the stop mask could be located; without a mask the sample just gets no stop credit)
            valid = (not unparsed and not gen["hit_max_new"] and (diag or {}).get("end_session_valid") is True
                     and not (diag or {}).get("end_session_t1_ignored"))
            out.append({"t": t, "replicate": g, "real_final": bool(real_final), "ended_planner": bool(end),
                        "planner_unparsed": unparsed, "planner_hit_max_new": gen["hit_max_new"],
                        "decision_valid": valid, "planner_diag": diag, "planner_fit": gen.get("fit"),
                        "reward": float(valid and bool(end) == bool(real_final)),
                        "planner_gen": {"prompt_ids": gen["prompt_ids"], "gen_ids": gen["gen_ids"], "stop_mask": sm,
                                        "note_mask": nm, "hit_max_new": gen["hit_max_new"],
                                        "gen_logprobs": gen.get("gen_logprobs"), "gen_adapter": gen.get("gen_adapter"),
                                        "temperature": temperature, "top_p": top_p, "seed": pseed}})
        # one stop-supervision example per position: the first VALID sample with a located stop mask
        for x in out:
            if not x["decision_valid"] or x["planner_gen"]["stop_mask"] is None:
                continue
            tgt = self.planner.stop_target(x["planner_gen"]["gen_ids"], x["planner_gen"]["stop_mask"], real_final)
            if tgt is not None:
                out[0]["aux"] = dict(tgt, prompt_ids=list(x["planner_gen"]["prompt_ids"]))
                break
        return out

    def run_task1(self, conversation_id, seed=0, keep_prompts=False):
        """Task 1 (teacher-forced) stop decisions of the Planner on a REAL conversation (pend arm), read
        off the full Task 1 generation (the same code as the evaluation; the Implicit Profile also needs
        the Speaker's predictions). Each turn carries the raw decision and the Speaker blank, and the
        conversation the K+1 Speaker blank, so task1_stop can apply the M2 mapping."""
        if self.arm != "pend":
            raise ValueError("run_task1 is implemented for the pend arm")
        rows, k1 = self.task1_generate(conversation_id, seed=seed, keep_prompts=keep_prompts)
        n = len(rows)
        turns = []
        for r in rows:
            x = {"t": r["turn_index"], "n_real": n, "real_final": r["turn_index"] == n,
                 "ended_planner": bool(r["planner_ends_session"]), "speaker_blank": bool(r["speaker_ended"]),
                 "planner_unparsed": bool(r.get("planner_unparsed")),
                 "planner_hit_max_new": bool(r.get("planner_hit_max_new")),
                 "goal_met": r.get("goal_met"), "planner_diag": r.get("planner_diag"),
                 "planner_fit": r.get("planner_fit"), "speaker_fits": r.get("speaker_fits"),
                 "speaker_hit_max_new": r.get("speaker_hit_max_new"), "emitted_capped": r.get("emitted_capped")}
            if keep_prompts:
                x["user_prompt"] = r["planner_prompt"]
            turns.append(x)
        return {"conversation_id": conversation_id, "record_id": self.recs[conversation_id]["record_id"],
                "arm": self.arm, "n_real": n, "planner_path": self.planner.path,
                "planner_adapter": self.planner.adapter, "end_mapping": "M2",
                "k1_speaker_blank": bool(k1["ended_speaker"]), "k1_ended": bool(k1["ended"]), "turns": turns}
