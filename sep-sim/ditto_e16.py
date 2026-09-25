# -*- coding: utf-8 -*-
"""Ditto-8B speaker for the E1.6 tree (Sep-1st-Simulator-e1r @ cf19400).

The E1.6 tree branched before Ditto support was added, so its sepsim.models has only the
UserLM Speaker. This is the v2fix `DittoSpeaker` (Sep-Simulator sepsim/models.py) copied
verbatim -- load, build_prompt, say, end_probability -- as a subclass of the E1.6 Speaker, with
the attributes the E1.6 `Speaker.say` reads set for Ditto's template grammar:

  header   E1.6 SEPSIM_ROLESTOP ends a draw at the token that opens a new role. In the Llama-3
           template of UserLM that is <|start_header_id|>; in Ditto's Qwen chat template it is
           <|im_start|>. Same rule (a stop token named by the template grammar), other tokenizer.
  endconv  None: Ditto has no conversation-end token; a blank message is its end signal. So
           SEPSIM_ENDGATE / SEPSIM_ENDMASK_RETRY have nothing to mask and must stay off
           (asserted in load, never silently ignored).

SEPSIM_INTENT_PROSE / NEXTSTEP_INLINE / T1_INTENT_ONLY live in the UserLM build_prompt, which
this class overrides; they are asserted off for the same reason.

`card_sampling()` returns the sampling parameters from the checkpoint's own
generation_config.json -- the Ditto counterpart of E1.6 Z1 (T1_SAMPLE), which used UserLM-8b's
model-card T=1.0 / top_p=0.8. Nothing is chosen by us.
"""
from __future__ import annotations

import json
import os

from sepsim import models

UNSUPPORTED = ("SEPSIM_ENDGATE", "SEPSIM_ENDMASK_RETRY", "SEPSIM_INTENT_PROSE",
               "SEPSIM_NEXTSTEP_INLINE", "SEPSIM_T1_INTENT_ONLY", "SEPSIM_KEEPEND", "SEPSIM_ENDSCORE")


class DittoSpeaker(models.Speaker):
    """Ditto's user simulation interface with role-flipped dialogue history.

    Ditto is Qwen3-VL based and does not use UserLM's conversation-end token.
    A blank message is the benchmark's model-agnostic end signal.
    """

    def __init__(self, path=None, **kwargs):
        ditto_path = path or os.environ.get("DITTO_PATH")
        if not ditto_path:
            raise ValueError("DITTO_PATH must name a local Ditto-8B checkpoint")
        super().__init__(path=ditto_path, **kwargs)

    def load(self):
        bad = [k for k in UNSUPPORTED if os.environ.get(k, "0") not in ("", "0")]
        if bad or os.environ.get("SEPSIM_INTENT_CACHE"):
            raise RuntimeError("UserLM-only switches set for Ditto: %s" % (bad or ["SEPSIM_INTENT_CACHE"]))
        import sys
        sys.modules["torchvision"] = None  # text-only Ditto; avoid mismatched optional torchvision
        sys.modules["torchaudio"] = None  # text-only Ditto; avoid mismatched optional torchaudio
        import torch
        from transformers import AutoTokenizer, AutoModelForImageTextToText

        self._tok = AutoTokenizer.from_pretrained(self.path)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map=({"": self.gpu} if self.gpu is not None else "auto")).eval()
        self.eot = self._tok.eos_token_id
        self.header = self._tok.convert_tokens_to_ids("<|im_start|>")
        if self.header is None or self.header == self._tok.unk_token_id:
            raise RuntimeError("Ditto tokenizer has no <|im_start|>")
        self.endconv = None
        return self

    def build_prompt(self, scenario_text, block, hist_u, hist_a, turn):
        instruction = ("You are simulating a real human USER in a conversation "
                       "with an AI dataset-search assistant. Stay in character; "
                       "write ONLY the user's next message (no explanations, "
                       "no assistant-style answers).")
        system = instruction + "\n\n" + scenario_text + "\n\n" + block
        msgs = [{"role": "system", "content": system}]
        for i, utterance in enumerate(hist_u):
            msgs.append({"role": "assistant", "content": utterance})
            if i < len(hist_a):
                msgs.append({"role": "user", "content": hist_a[i]})
        try:
            return self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def say(self, *args, **kwargs):
        if kwargs.get("mask_end") or kwargs.get("allow_end"):
            raise RuntimeError("Ditto has no end token to mask or keep")
        txt, _ = super().say(*args, **kwargs)
        return txt, not bool(txt.strip())

    def end_probability(self, scenario_text, hist_u, hist_a):
        return None

    def say_batch(self, reqs):
        """Batched Speaker.say (E1.6 models.py @ cf19400), same attempt schedule and the same guards:
        attempt k samples at (temperature, top_p) -- or, for a greedy request, greedy first and then
        (RETRY_TEMPERATURE, RETRY_TOP_P) -- and a draw is redrawn while it is empty, seen before, our
        own template, or a verbatim reuse (as enabled per request). Requests whose draw passes leave the
        batch; the rest go to the next attempt together. Only the random stream differs from calling
        say() one by one (one manual_seed per generate call instead of per request).
        reqs: dicts scenario_text, block, hist_u, hist_a, turn, seed, temperature, top_p, avoid,
        reject_template, reject_reuse. -> list of (text, ended(blank), fit)."""
        import re
        import torch
        M = models
        tok = getattr(self, "_raw_tok", self._tok)
        budget = getattr(self, "budget", None)
        eos = [self.eot, self.header] if M.ROLESTOP else [self.eot]
        pad = self._tok.pad_token_id
        dev = "cuda:%d" % self.gpu if self.gpu is not None else next(self._model.parameters()).device
        st = []
        for r in reqs:
            prompt = self.build_prompt(r["scenario_text"], r["block"], r["hist_u"], r["hist_a"], r["turn"])
            fit = getattr(self, "last_fit", None)
            ids = tok(prompt)["input_ids"]
            if budget is not None and len(ids) > budget:
                raise AssertionError("Speaker prompt %d > budget %d" % (len(ids), budget))
            sampled = bool(r["temperature"] and r["temperature"] > 0)
            extra = bool(r["reject_template"] or r["reject_reuse"])
            avoid = r.get("avoid") or []
            st.append({"ids": ids, "fit": fit, "sampled": sampled, "k": 0, "txt": "", "done": False,
                       "tries": M.MAX_REGEN if (self.guardrails and (sampled or extra)) else 1,
                       "seen": [(x or "").strip().lower() for x in avoid if (x or "").strip()],
                       "prior": [x for x in avoid if (x or "").strip()], "r": r})
        while True:
            active = [s for s in st if not s["done"] and s["k"] < s["tries"]]
            if not active:
                break
            groups = {}
            for s in active:
                r = s["r"]
                tp = (r["temperature"], r["top_p"]) if s["sampled"] else (
                    (0.0, 1.0) if s["k"] == 0 else (M.RETRY_TEMPERATURE, M.RETRY_TOP_P))
                groups.setdefault(tp, []).append(s)
            for (t_k, p_k), grp in groups.items():
                L = max(len(s["ids"]) for s in grp)
                x = torch.tensor([[pad] * (L - len(s["ids"])) + s["ids"] for s in grp], dtype=torch.long, device=dev)
                a = torch.tensor([[0] * (L - len(s["ids"])) + [1] * len(s["ids"]) for s in grp], dtype=torch.long, device=dev)
                torch.manual_seed(min(s["r"]["seed"] + 1000 * s["k"] for s in grp))
                kw = dict(max_new_tokens=self.max_new, eos_token_id=eos, pad_token_id=pad)
                kw.update(dict(do_sample=True, temperature=t_k, top_p=p_k) if t_k and t_k > 0 else dict(do_sample=False))
                with torch.no_grad():
                    out = self._model.generate(input_ids=x, attention_mask=a, **kw)
                for i, s in enumerate(grp):
                    row = out[i][L:].tolist()
                    cut = next((j for j, v in enumerate(row) if v in eos), None)
                    row = row[: cut + 1] if cut is not None else row
                    raw = self._tok.decode(row, skip_special_tokens=False)
                    txt = re.sub(r"\s+", " ", M.TAG_RE.sub(" ", raw)).strip()
                    s["txt"], s["k"] = txt, s["k"] + 1
                    if not self.guardrails:
                        s["done"] = True
                        continue
                    low = txt.lower()
                    r = s["r"]
                    if not low or low in s["seen"]:
                        self.n_regen += 1
                        continue
                    if r["reject_template"] and M.looks_like_template(txt):
                        self.n_reject_template += 1
                        self.n_regen += 1
                        continue
                    if r["reject_reuse"] and M.is_verbatim_reuse(txt, s["prior"]):
                        self.n_reject_reuse += 1
                        self.n_regen += 1
                        continue
                    s["done"] = True
        return [(s["txt"], not bool(s["txt"].strip()), s["fit"]) for s in st]

    def card_sampling(self):
        """(temperature, top_p) from the checkpoint's generation_config.json."""
        p = os.path.join(self.path, "generation_config.json")
        cfg = json.load(open(p, encoding="utf-8"))
        t, tp = cfg.get("temperature"), cfg.get("top_p")
        if t is None or tp is None or not cfg.get("do_sample", True):
            raise RuntimeError("%s has no sampling temperature/top_p" % p)
        return float(t), float(tp)
