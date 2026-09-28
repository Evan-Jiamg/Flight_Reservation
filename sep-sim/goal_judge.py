"""Goal-satisfaction judge: one prompt builder shared by the LABEL judge and the TRAINED judge.

Input (both judges, training and inference): the scenario text the Speaker also reads
(sepsim.pipeline.scenario_text: persona + goal context + goal topic + familiarity + known
datasets) and the conversation so far. NO turn count, NO ledger or agenda output.
Output: {"status": "SATISFIED"|"PARTIAL"|"NOT", "unmet": [<= 3 short phrases]}.

Over-long inputs drop the OLDEST whole exchanges (same budget for both judges, so the
trained judge sees what its labels were produced from). A reply that does not parse is
recorded as status UNKNOWN and counted -- never silently mapped to any status.

GoalJudge.status_probs (and assess()["status_probs"]) gives P(SATISFIED/PARTIAL/NOT) by
teacher-forced scoring of the canonical status prefix after the same fitted generation prompt,
normalized over the three (see status_token_plan for the exact scored tokens).
"""
from __future__ import annotations

import copy
import json
import math
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


# ---- status probabilities (teacher-forced scoring of the canonical status prefix) ----------------
# The scored continuation for status X is STATUS_HEAD + X, i.e. '{"status": "X' -- the canonical
# prefix '{"status": "X"' WITHOUT its closing quote: byte-level BPE pre-tokenizers (Qwen, Llama-3)
# merge that quote with the following comma into one token ('",'), so a sequence ending in a bare
# '"' is a tokenization the trained judge never saw. status_token_plan asserts that the scored ids
# are an exact prefix of the ids of the full canonical target (the training tokenization); it
# raises instead of scoring anything else.
STATUS_HEAD = '{"status": "'


def common_prefix_len(seqs):
    seqs = [list(s) for s in seqs]
    n = min(len(s) for s in seqs) if seqs else 0
    for i in range(n):
        if any(s[i] != seqs[0][i] for s in seqs[1:]):
            return i
    return n


def status_token_plan(tok, prompt_text):
    """-> (shared_ids, {status: branch_ids}, n_prompt_tokens).

    shared_ids = prompt + the part of the status prefix common to all three statuses;
    branch_ids[X] = the remaining ids of '{"status": "X'. Tokenized exactly as the training
    target (prompt text + canonical JSON), with add_special_tokens=False (single BOS)."""
    p_ids = list(tok(prompt_text, add_special_tokens=False)["input_ids"])
    seqs = {}
    for st in STATUSES:
        pre = list(tok(prompt_text + STATUS_HEAD + st, add_special_tokens=False)["input_ids"])
        full = list(tok(prompt_text + canonical({"status": st, "unmet": []}),
                        add_special_tokens=False)["input_ids"])
        if full[:len(pre)] != pre:
            raise ValueError("status prefix %r does not tokenize as a prefix of the canonical target" % st)
        if pre[:len(p_ids)] != p_ids:
            raise ValueError("prompt ids are not a prefix of prompt+status ids (template boundary)")
        seqs[st] = pre
    c = common_prefix_len(seqs.values())
    if c < len(p_ids):
        raise ValueError("status continuations diverge inside the prompt")
    branches = {st: s[c:] for st, s in seqs.items()}
    if any(not b for b in branches.values()):
        raise ValueError("a status continuation has no distinguishing token")
    return seqs[STATUSES[0]][:c], branches, len(p_ids)


def normalize_logps(logps):
    """{status: log P(prefix)} -> ({status: p normalized over the three}, total unnormalized mass)."""
    m = max(logps.values())
    z = sum(math.exp(v - m) for v in logps.values())
    return {k: math.exp(v - m) / z for k, v in logps.items()}, math.exp(m) * z


class GoalJudge:
    """HF judge (label model or trained Qwen3-4B + LoRA). Greedy, deterministic."""

    def __init__(self, model_path, adapter=None, gpu=0, dtype="bfloat16", load_4bit=False,
                 device_map=None, with_probs=True):
        self.model_path, self.adapter, self.gpu = model_path, adapter, gpu
        self.dtype, self.load_4bit, self.device_map = dtype, load_4bit, device_map
        self.with_probs = with_probs       # assess() also scores status_probs (one extra prefill)
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

    def prompt_text(self, scenario_text, hist_u, hist_a):
        """The fitted generation prompt (same for assess and status_probs) and its fit info."""
        if self.model is None:
            self.load()
        msgs, info = fit_messages(self.tok, scenario_text, hist_u, hist_a)
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), info

    def _score_prompt(self, text):
        """-> ({status: p}, mass): teacher-forced log P of each status prefix, normalized."""
        import torch
        shared, branches, n_prompt = status_token_plan(self.tok, text)
        if n_prompt > JUDGE_BUDGET:
            raise AssertionError("judge prompt %d > budget %d" % (n_prompt, JUDGE_BUDGET))
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([shared], device=dev), use_cache=True, logits_to_keep=1)
            lp0 = torch.log_softmax(out.logits[0, -1].float(), -1)
            cache, c = out.past_key_values, len(shared)
            logps = {}
            for st, br in branches.items():
                lp = float(lp0[br[0]])
                if len(br) > 1:
                    can_crop = hasattr(cache, "crop")
                    cb = cache if can_crop else copy.deepcopy(cache)
                    o = self.model(input_ids=torch.tensor([br[:-1]], device=dev), past_key_values=cb,
                                   use_cache=True)
                    lq = torch.log_softmax(o.logits[0].float(), -1)
                    lp += sum(float(lq[k, br[k + 1]]) for k in range(len(br) - 1))
                    if can_crop:
                        cache.crop(c)
                logps[st] = lp
        return normalize_logps(logps)

    def status_probs(self, scenario_text, hist_u, hist_a):
        """{"SATISFIED": p, "PARTIAL": p, "NOT": p}: P('{"status": "X' | fitted prompt), normalized
        over the three. Same fitted prompt as assess()."""
        text, _ = self.prompt_text(scenario_text, hist_u, hist_a)
        return self._score_prompt(text)[0]

    def assess(self, scenario_text, hist_u, hist_a):
        import torch
        text, info = self.prompt_text(scenario_text, hist_u, hist_a)
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
        extra = {}
        if self.with_probs:
            probs, mass = self._score_prompt(text)
            extra = {"status_probs": probs, "status_probs_mass": mass}
        return {**res, "parse_ok": ok, "raw": raw, **info, **extra}
