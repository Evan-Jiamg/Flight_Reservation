"""Goal-satisfaction judge: one prompt builder shared by the LABEL judge and the TRAINED judge.

Input (both judges, training and inference): the scenario text the Speaker also reads
(sepsim.pipeline.scenario_text: persona + goal context + goal topic + familiarity + known
datasets) and the conversation so far. NO turn count, NO ledger or agenda output.
Output: {"status": "SATISFIED"|"PARTIAL"|"NOT", "unmet": [<= 3 short phrases]}.

Over-long inputs drop the OLDEST whole exchanges (same budget for both judges, so the
trained judge sees what its labels were produced from). A reply that does not parse is
recorded as status UNKNOWN and counted -- never silently mapped to any status.
"""
from __future__ import annotations

import json
import re

STATUSES = ("SATISFIED", "PARTIAL", "NOT")
JUDGE_BUDGET = 16384
MAX_NEW = 160
MAX_UNMET = 3

SYSTEM = (
    "You judge how far a person searching for research datasets has been satisfied by the "
    "assistant so far.\n"
    "Use only two things: what this person wants, as described below, and the conversation.\n"
    "A need counts as satisfied only when the assistant has pointed to concrete datasets or "
    "sources that plausibly meet it. Needs that are part of what this person wants count even "
    "if they have not raised them yet.\n"
    "SATISFIED: everything they want has been adequately addressed.\n"
    "PARTIAL: some of what they want has been adequately addressed, some has not.\n"
    "NOT: nothing they want has been adequately addressed yet.\n"
    "Reply with JSON only, in this form:\n"
    '{"status": "SATISFIED" | "PARTIAL" | "NOT", "unmet": ["<short phrase>", ...]}\n'
    "List at most three unmet needs, most important first, each under 12 words; "
    "use an empty list when the status is SATISFIED."
)
OMISSION = "[Earlier exchanges are omitted; the most recent ones follow.]\n\n"


def conversation_text(hist_u, hist_a, drop=0):
    lines = []
    for i, u in enumerate(hist_u):
        if i < drop:
            continue
        lines.append("USER: " + u)
        if i < len(hist_a):
            lines.append("ASSISTANT: " + hist_a[i])
    body = "\n\n".join(lines) if lines else "(nothing has been said yet)"
    return (OMISSION if drop else "") + body


def user_content(scenario_text, hist_u, hist_a, drop=0):
    return ("WHAT THIS PERSON WANTS\n" + " ".join(scenario_text.split())
            + "\n\nTHE CONVERSATION SO FAR\n" + conversation_text(hist_u, hist_a, drop))


def messages(scenario_text, hist_u, hist_a, drop=0):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_content(scenario_text, hist_u, hist_a, drop)}]


def fit_messages(tok, scenario_text, hist_u, hist_a, budget=JUDGE_BUDGET):
    """Return (messages, info) within budget, dropping oldest exchanges if needed."""
    def n_tokens(msgs):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return len(tok(text, add_special_tokens=False)["input_ids"])
    msgs = messages(scenario_text, hist_u, hist_a)
    n = n_tokens(msgs)
    if n <= budget:
        return msgs, {"prompt_tokens": n, "dropped_exchanges": 0}
    for drop in range(1, len(hist_u)):
        msgs = messages(scenario_text, hist_u, hist_a, drop)
        m = n_tokens(msgs)
        if m <= budget:
            return msgs, {"prompt_tokens": m, "original_tokens": n, "dropped_exchanges": drop}
    raise ValueError("cannot fit judge input even with one exchange")


def parse(raw):
    """-> (result dict, ok). Canonical form; UNKNOWN when unreadable."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    d = None
    if m:
        for cand in (m.group(0), re.sub(r",\s*([}\]])", r"\1", m.group(0))):
            try:
                d = json.loads(cand)
                break
            except Exception:
                continue
    if not isinstance(d, dict):
        return {"status": "UNKNOWN", "unmet": []}, False
    st = str(d.get("status", "")).strip().upper()
    if st not in STATUSES:
        return {"status": "UNKNOWN", "unmet": []}, False
    unmet = d.get("unmet") or []
    if not isinstance(unmet, list):
        unmet = [str(unmet)]
    unmet = [" ".join(str(u).split())[:120] for u in unmet if str(u).strip()][:MAX_UNMET]
    if st == "SATISFIED":
        unmet = []
    return {"status": st, "unmet": unmet}, True


def canonical(result):
    """Training target text (exactly what the trained judge learns to emit)."""
    return json.dumps({"status": result["status"], "unmet": result["unmet"]}, ensure_ascii=False)


class GoalJudge:
    """HF judge (label model or trained Qwen3-4B + LoRA). Greedy, deterministic."""

    def __init__(self, model_path, adapter=None, gpu=0, dtype="bfloat16", load_4bit=False,
                 device_map=None):
        self.model_path, self.adapter, self.gpu = model_path, adapter, gpu
        self.dtype, self.load_4bit, self.device_map = dtype, load_4bit, device_map
        self.tok = self.model = None
        self.n_calls = self.n_unparsed = 0

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from fit_prompts import dtype_kwarg
        dt = getattr(torch, self.dtype)
        kw = {"low_cpu_mem_usage": True, "device_map": self.device_map or {"": self.gpu}, **dtype_kwarg(dt)}
        if self.load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dt, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True)
        self.tok = AutoTokenizer.from_pretrained(self.model_path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kw)
        if self.adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter)
        self.model.eval()
        return self

    def assess(self, scenario_text, hist_u, hist_a):
        import torch
        if self.model is None:
            self.load()
        msgs, info = fit_messages(self.tok, scenario_text, hist_u, hist_a)
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # the chat template already carries any BOS (Llama-3.1 does); never add a second one
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        enc.pop("token_type_ids", None)
        dev = next(self.model.parameters()).device
        enc = enc.to(dev)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        raw = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        res, ok = parse(raw)
        self.n_calls += 1
        self.n_unparsed += int(not ok)
        return {**res, "parse_ok": ok, "raw": raw, **info}
