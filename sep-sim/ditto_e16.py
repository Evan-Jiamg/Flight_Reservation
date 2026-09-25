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

    def card_sampling(self):
        """(temperature, top_p) from the checkpoint's generation_config.json."""
        p = os.path.join(self.path, "generation_config.json")
        cfg = json.load(open(p, encoding="utf-8"))
        t, tp = cfg.get("temperature"), cfg.get("top_p")
        if t is None or tp is None or not cfg.get("do_sample", True):
            raise RuntimeError("%s has no sampling temperature/top_p" % p)
        return float(t), float(tp)
