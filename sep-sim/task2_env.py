# -*- coding: utf-8 -*-
"""Task2Env: one reusable Task 2 environment for evaluation rollouts AND RL (Planner is the policy).

Everything that decides behaviour lives here, once, so evaluation and RL cannot drift apart:
  * PlannerLM   any causal LM (32B NF4, or a smaller 7-9B/20B model), optional LoRA adapter
                (trainable for RL). Prompts are fitted (fit_prompts, never truncated) to
                min(fit_prompts.PLANNER_BUDGET, model context - max_new - margin). Tokenized with
                add_special_tokens=False (the chat template carries any BOS).
                generate(system, user, temperature, top_p, seed) -> dict(raw, prompt_ids, gen_ids, fit)
  * Speaker     FitDittoSpeaker (D5 fix), frozen, v2fix guards and selector line for line.
  * Judge       goal_judge.GoalJudge (a2 only), frozen.
  * arms        a0 = original Planner prompt/read_plan (override on); Planner end logged only.
                a2 = v3 system/user prompt, read_plan_v3 (stop = end_session, no length clamp),
                     judge GOAL STATUS, silent Planner exit.
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


def make_floor_judge(Judge):
    class FloorJudge(Judge):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.gpt5 = str(self.model).startswith("gpt-5")
            self.floor = None if self.gpt5 else JUDGE_MIN_TOKENS
            self.n_empty = 0
            self._lock = threading.Lock()

        def chat(self, system, user, max_tokens=400):
            out = super().chat(system, user, max(max_tokens, self.floor) if self.floor else max_tokens)
            if not (out or "").strip():
                with self._lock:
                    self.n_empty += 1
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
                 max_new=PLANNER_MAX_NEW):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import fit_prompts as F
        self.path, self.gpu, self.max_new = path, gpu, max_new
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
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
        gc = getattr(self.model, "generation_config", None)
        e = getattr(gc, "eos_token_id", None) if gc is not None else None
        if e is None:
            e = self.tok.eos_token_id
        return set(e if isinstance(e, (list, tuple)) else [e])

    def generate_batch(self, items):
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
            prompts.append((user_fit, fit, ids))
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
                        "fit": {**fit, "prompt_tokens": len(ids), "budget": self.budget, "batched": len(items)},
                        "hit_max_new": cut is None and len(gen) >= self.max_new})
        return res

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
        return {"prefix_ids": list(gen_ids[:i]), "target_ids": list(ids), "want_end": bool(want_end)}

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
                 batch=False, max_batch=8):
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
        if arm == "pend":
            self.system = V3.system_prompt_pend()
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
        self.r0_effort = os.environ.get("R0_REASONING_EFFORT", "minimal")
        self.judge_effort = os.environ.get("JUDGE_REASONING_EFFORT", "minimal")
        self.ledger_judge = make_floor_judge(Judge)(reasoning_effort=self.judge_effort, verbose=False,
                                                    cache_dir=os.path.join(WORK, "judge_cache"))
        self.r0 = R0Client(reasoning_effort=self.r0_effort)
        if self.e16:
            import ditto_e16
            base_cls = ditto_e16.DittoSpeaker
        else:
            base_cls = models.DittoSpeaker
        FitDitto = F.make_fit_ditto_speaker(base_cls)
        self.speaker = FitDitto(path=ditto_path, gpu=gpu, position=run_v2.POSITION).load()
        # E1.6 Z1 (T1_SAMPLE): turn 1 is sampled at the speaker checkpoint's own card values
        self.t1_sampling = self.speaker.card_sampling() if self.e16 else None
        # cross-episode dynamic batching (threads submit, one GPU call per batch); off = one by one
        self.planner_batcher = self.speaker_batcher = None
        if batch:
            import batching
            if not hasattr(self.speaker, "say_batch"):
                raise RuntimeError("batching needs the E1.6 Ditto speaker (say_batch)")
            self.planner_batcher = batching.Batcher(planner.generate_batch, self.gpu_lock,
                                                    key=lambda it: (it["temperature"], it["top_p"]),
                                                    max_batch=max_batch, name="planner-batcher")
            self.speaker_batcher = batching.Batcher(self.speaker.say_batch, self.gpu_lock,
                                                    max_batch=2 * max_batch, name="speaker-batcher")

    def _plan(self, system, user, temperature, top_p, seed):
        if self.planner_batcher is not None:
            return self.planner_batcher({"system": system, "user": user, "temperature": temperature,
                                         "top_p": top_p, "seed": seed})
        with self.gpu_lock:
            return self.planner.generate(system, user, temperature, top_p, seed)

    def _plan_many(self, items):
        if self.planner_batcher is not None:
            return self.planner_batcher.map(items)
        return [self._plan(it["system"], it["user"], it["temperature"], it["top_p"], it["seed"]) for it in items]

    def _say_many(self, reqs):
        """-> [(text, ended, fit)] in request order."""
        if self.speaker_batcher is not None:
            return self.speaker_batcher.map(reqs)
        out = []
        for r in reqs:
            with self.gpu_lock:
                txt, e = self.speaker.say(r["scenario_text"], r["block"], r["hist_u"], r["hist_a"], r["turn"],
                                          seed=r["seed"], temperature=r["temperature"], top_p=r["top_p"],
                                          avoid=r["avoid"], reject_template=r["reject_template"],
                                          reject_reuse=r["reject_reuse"])
                fit = self.speaker.last_fit
            out.append((txt, e, fit))
        return out

    def describe(self):
        import fit_prompts as F
        return {"arm": self.arm, "planner": self.planner.path, "planner_adapter": self.planner.adapter,
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
                "ledger_judge_min_tokens": self.ledger_judge.floor, "ledger_judge_empty": self.ledger_judge.n_empty,
                "act_prior": os.environ["SEPSIM_ACT_PRIOR"], "v2fix": V2FIX, "t_max": T_MAX,
                "tree": tree_of(self.arm), "arm_env": ARM_ENV[self.arm], "t1_sampling": self.t1_sampling,
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
             "block": None, "cov": []}

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
                up = V3.user_prompt_pend(scenario, S["prev_block"], hist_u, hist_a, t, led, prev_ann={})
            else:
                up = PP.user_prompt(scenario, S["prev_block"], hist_u, hist_a, t, ledger=led,
                                    prev_ann={}, agenda_view=ag.render(), p_end=None)
            pseed = int(hashlib.sha256(("%s|%d|%d|%d" % (sid, t, replicate, 7)).encode()).hexdigest()[:8], 16)
            g = self._plan(self.system, up, planner_temperature, planner_top_p, pseed)
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
            elif arm == "pend":
                fields, diag, end_session = V3.read_plan_pend(raw, t, scenario, rng, led)
            elif arm == "e16":
                fields, diag = PP.read_plan(raw, t, scenario, rng, led, agenda_open=False)
                end_session = None
            else:
                fields, diag = PP.read_plan(raw, t, scenario, rng, led)
                end_session = None
            unparsed = fields is None
            if unparsed:
                fields = {"move": "Other", "act": "other"}
            ended = bool(end_session) if (v3 or arm == "pend") else bool(state.ends_session(fields))
            base = {"planner_prompt": g["prompt_text"], "planner_fit": g["fit"], "planner_raw": raw,
                    "planner_hit_max_new": g["hit_max_new"], "planner_diag": diag,
                    "planner_unparsed": unparsed, "ended_planner": ended,
                    "move": fields.get("move", ""), "act": fields.get("act", ""),
                    "stop_rule": fields.get("stop_rule", "none"), "goal_status": gs, "self_judge": self_judge,
                    "goal_met": fields.get("goal_met"), "still_wanted": fields.get("still_wanted"),
                    "ledger_before": {"turns": led.turns, "gain_trace": list(led.gain_trace)}}
            if record_generation:
                base["planner_gen"] = {"prompt_ids": g["prompt_ids"], "gen_ids": g["gen_ids"], "stop_mask": planner.stop_mask(g["gen_ids"]),
                                       "temperature": planner_temperature, "top_p": planner_top_p, "seed": pseed}
            if v3 and ended:
                return {**base, "planner_stop": True, "user": ""}
            if unparsed:
                block = S["prev_block"]
            elif v3:
                block = V3.speaker_block_v3(fields, gs)
            elif arm == "pend":
                block = V3.speaker_block_pend(fields)
            else:
                block = PP.render_block(fields, ag.render())
            S["block"] = block
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
                reasons = [run_v2.guard_reason(c, prior) for c in cands]
                while all(reasons) and n_extra < run_v2.REDRAW:
                    k = run_v2.NSAMP + n_extra
                    sx, ex, _ = self._say_many([req(pipeline.seed_for(sid, t, k + 1), T_S, P_S)])[0]
                    cands.append(sx)
                    flags.append(ex)
                    reasons.append(run_v2.guard_reason(sx, prior))
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
            reply = self.r0.reply(msgs)
            prev_reply = S["hist_a"][-1] if S["hist_a"] else ""
            S["hist_a"].append(reply)
            ledger.update(t, text, reply)
            led.observe({}, reply, prev_reply)
            ag.retire_satisfied(reply, {})
            if AG.looks_like_new_offer(reply, prev_reply):
                ag.reset_on_new_offer()
            S["prev_block"] = S["block"]
            S["cov"].append((t, (round(ledger.coverage(), 4), bool(ledger.complete()))))
            return reply

        return {"S": S, "speak": speak, "respond": respond, "ledger": ledger, "led": led, "ag": ag,
                "rid": rid, "scenario": scenario}

    def run_episode(self, conversation_id, seed, replicate=0, planner_temperature=0.0, planner_top_p=1.0,
                    record_generation=False):
        from task2_episode import run_episode
        arm, planner = self.arm, self.planner
        ss = self._session(conversation_id, seed, replicate, planner_temperature, planner_top_p, record_generation)
        S, speak, respond, ledger, rid = ss["S"], ss["speak"], ss["respond"], ss["ledger"], ss["rid"]
        # e16: E1.6 SEPSIM_PLANNER_END -- the Planner's Complete act ends the episode after the
        # closing message it asked for (emitted, no assistant reply). v3 arms exit silently instead.
        ep = run_episode(T_MAX, None, speak, respond, planner_end=(arm in EMIT_END_ARMS))
        after, last = dict(S["cov"]), (0.0, False)
        for step in ep["trace"]:
            last = after.get(step["t"], last)
            step["coverage_after"], step["complete_after"] = last
        return {"conversation_id": conversation_id, "record_id": rid, "seed": seed, "arm": arm,
                "replicate": replicate, "speaker_kind": "ditto", "planner_path": planner.path,
                "planner_adapter": planner.adapter, "planner_temperature": planner_temperature,
                "emitted_user_turns": ep["emitted_user_turns"], "decision_steps": ep["decision_steps"],
                "end_kind": ep["end_kind"], "turns": ep["emitted_user_turns"], "stop_kind": ep["end_kind"],
                "ended_by_token": ep["end_kind"] != "t_max",
                "coverage": round(ledger.coverage(), 4), "complete": ledger.complete(),
                "n_req": len(self.reqs[conversation_id]["req"]), "ledger": ledger.as_dict(), "trace": ep["trace"],
                "ledger_judge_empty_total": self.ledger_judge.n_empty,
                "r0_empty_retries_total": getattr(self.r0, "n_empty_retries", None),
                "human_turns": self.human_turns(conversation_id)}

    def task1_generate(self, conversation_id, seed=0):
        """Task 1 (teacher-forced) generations for one REAL conversation, in the benchmark's generations
        schema (tools/score_method.py): one row per real user turn with greedy (the selected candidate),
        samples (the other candidates), greedy_ended (Speaker end OR the Planner's end, as E1.6's
        PLANNER_END records it). The history at turn t is always the real one; the Planner's own state
        carries over, as in run_v2. Planner at temperature 0. No R0, no ledger."""
        from sepsim import agenda as AG, pipeline
        ss = self._session(conversation_id, seed, 0, 0.0, 1.0, False, with_ledger=False)
        S, speak, led, ag = ss["S"], ss["speak"], ss["led"], ss["ag"]
        rec = self.recs[conversation_id]
        users, agents = pipeline.split_messages(rec)
        goal = rec["scenario"].get("goal") or {}
        rows = []
        for t in range(1, len(users) + 1):
            st = speak(t)
            if st.get("planner_stop"):
                raise RuntimeError("silent Planner exit has no Task 1 utterance; use an emit-end arm")
            cands, ends, idx = st["candidates"], st["candidates_ended"], st["selected_index"]
            samples = [c for i, c in enumerate(cands) if i != idx]
            s_ends = [e for i, e in enumerate(ends) if i != idx]
            rows.append({"record_id": rec["record_id"], "conversation_id": conversation_id, "turn_index": t,
                         "is_first_turn": t == 1, "discipline": goal.get("discipline", "unknown"),
                         "intent_variant": "pend_" + os.path.basename(str(self.planner.path).rstrip("/")),
                         "greedy": st["user"], "greedy_ended": bool(st["ended_speaker"]) or bool(st["ended_planner"]),
                         "speaker_ended": bool(st["ended_speaker"]), "planner_ends_session": bool(st["ended_planner"]),
                         "samples": samples, "samples_ended": s_ends, "profile": st.get("block"),
                         "move": st.get("move"), "act": st.get("act"), "goal_met": st.get("goal_met"),
                         "planner_unparsed": st.get("planner_unparsed"), "planner_hit_max_new": st.get("planner_hit_max_new"),
                         "planner_fit": st.get("planner_fit"), "speaker_fit": st.get("speaker_fit"),
                         "guard_no_survivor": st.get("no_survivor"), "planner_adapter": self.planner.adapter})
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
        # K+1 probe (the benchmark's termination measurement): after the real person's last message
        # and whatever reply followed it, one more turn; ended = Speaker end OR Planner end (E1 semantics)
        k = len(users)
        st = speak(k + 1)
        if st.get("planner_stop"):
            raise RuntimeError("silent Planner exit has no Task 1 utterance; use an emit-end arm")
        k1 = {"record_id": rec["record_id"], "conversation_id": conversation_id, "gold_k": k, "turn_index": k + 1,
              "intent_variant": rows[0]["intent_variant"] + "_k1" if rows else "k1",
              "ended_speaker": bool(st["ended_speaker"]), "ended_planner": bool(st["ended_planner"]),
              "ended_empty": not (st["user"] or "").strip(),
              "ended": bool(st["ended_speaker"]) or bool(st["ended_planner"]),
              "greedy": st["user"], "greedy_ended": bool(st["ended_speaker"]) or bool(st["ended_planner"]),
              "planner_move": st.get("move"), "planner_act": st.get("act"), "goal_met": st.get("goal_met")}
        return rows, k1

    def human_turns(self, conversation_id):
        """Number of messages the real person sent in this conversation (the Task 2 turn target)."""
        from sepsim import pipeline
        users, _ = pipeline.split_messages(self.recs[conversation_id])
        return len(users)

    def task1_prompts(self, conversation_id):
        """Greedy teacher-forced pass (current policy) -> per turn the exact Planner user prompt, so a
        Task 1 training group can sample G decisions from the same state the Planner would be in."""
        res = self.run_task1(conversation_id, keep_prompts=True)
        return [{"t": r["t"], "n_real": r["n_real"], "real_final": r["real_final"], "user_prompt": r["user_prompt"]}
                for r in res["turns"]]

    def task1_sample(self, conversation_id, t, user_prompt, real_final, G, temperature, top_p, seed):
        """G sampled Planner decisions at one real turn; reward 1 if end_session == (message t was the
        person's last), else 0. Returned in the rollout schema (one step with planner_gen each)."""
        if self.arm != "pend":
            raise ValueError("task1_sample is implemented for the pend arm")
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
            fields, diag, end = V3.read_plan_pend(gen["raw"], t, scenario, random.Random(pseed),
                                                  stopping.StoppingLedger(scenario))
            out.append({"t": t, "replicate": g, "real_final": bool(real_final), "ended_planner": bool(end),
                        "planner_unparsed": fields is None, "planner_hit_max_new": gen["hit_max_new"],
                        "reward": float(bool(end) == bool(real_final)),
                        "planner_gen": {"prompt_ids": gen["prompt_ids"], "gen_ids": gen["gen_ids"], "stop_mask": self.planner.stop_mask(gen["gen_ids"]),
                                        "temperature": temperature, "top_p": top_p, "seed": pseed}})
        # one stop-supervision example per position: the first sample whose end_session value was found
        for x in out:
            tgt = self.planner.stop_target(x["planner_gen"]["gen_ids"], x["planner_gen"]["stop_mask"], real_final)
            if tgt is not None:
                out[0]["aux"] = dict(tgt, prompt_ids=list(x["planner_gen"]["prompt_ids"]))
                break
        return out

    def run_task1(self, conversation_id, seed=0, keep_prompts=False):
        """Task 1 (teacher-forced) stop decisions of the Planner on a REAL conversation (pend arm).

        Mirrors the E1.6 run_v2 loop: at turn t the history is the real person's first t-1 messages and
        the real assistant replies; the Planner writes its state (temperature 0) and its end_session is
        recorded against whether the real person's message t was their last. The Planner's own previous
        state carries over as in run_v2. Only the Planner runs (no Ditto, no R0): the stop decision is the
        Planner's alone in this architecture."""
        if self.arm != "pend":
            raise ValueError("run_task1 is implemented for the pend arm")
        from sepsim import persona as P, pipeline, state, stopping
        import planner_prompt_v3 as V3
        rec = self.recs[conversation_id]
        rid, scenario = rec["record_id"], rec["scenario"]
        users, agents = pipeline.split_messages(rec)
        n = len(users)
        rng = random.Random(pipeline.seed_for(rid, 0))
        led = stopping.StoppingLedger(scenario)
        prev_block = state.d0(P.initial_stage(scenario.get("goal")))
        rows = []
        for t in range(1, n + 1):
            if t >= 2 and t - 2 < len(agents):
                at = agents[t - 2]["text"]
                pt = agents[t - 3]["text"] if t >= 3 and t - 3 < len(agents) else ""
                led.observe({}, at, pt)
            hist_u = [u["text"] for u in users[: t - 1]]
            hist_a = [a["text"] for a in agents[: t - 1]]
            up = V3.user_prompt_pend(scenario, prev_block, hist_u, hist_a, t, led, prev_ann={})
            pseed = int(hashlib.sha256(("t1|%s|%d|%d" % (rid, t, seed)).encode()).hexdigest()[:8], 16)
            g = self._plan(self.system, up, 0.0, 1.0, pseed)
            fields, diag, end = V3.read_plan_pend(g["raw"], t, scenario, rng, led)
            unparsed = fields is None
            block = prev_block if unparsed else V3.speaker_block_pend(fields)
            rows.append({"t": t, "n_real": n, "real_final": t == n, "ended_planner": bool(end),
                         "planner_unparsed": unparsed, "planner_hit_max_new": g["hit_max_new"],
                         "goal_met": None if unparsed else fields.get("goal_met"),
                         "planner_fit": g["fit"], "planner_diag": diag})
            if keep_prompts:
                rows[-1]["user_prompt"] = up
            prev_block = block
        return {"conversation_id": conversation_id, "record_id": rid, "arm": self.arm, "n_real": n,
                "planner_path": self.planner.path, "planner_adapter": self.planner.adapter, "turns": rows}
