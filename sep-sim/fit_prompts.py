"""Fix for audit defect D5: right-side truncation in sepsim.models.

sepsim.models tokenizes with truncation=True and a hard cap (Planner.raw_with: 12000,
Speaker.say: 7000). Hugging Face truncates from the RIGHT, so an over-long prompt loses the
newest dialogue and the generation header. The frozen sepsim code is not edited (run
provenance); instead:

  * TokProxy wraps a tokenizer. A call carrying one of the legacy caps is performed WITHOUT
    truncation and asserts the prompt fits the new budget. Every other attribute and call
    passes through unchanged. For any prompt that already fit under the legacy cap the
    token ids are identical to the legacy path (there was no truncation there either).
  * Prompts are fitted BEFORE generation so that they are within the budget:
      - Planner: fit_planner_user keeps the static prefix (everything before
        "THE CONVERSATION SO FAR"), inserts an omission marker, and keeps the most recent
        dialogue tokens -- the same algorithm as stop_prompt.encode_stop_prompt.
      - Ditto Speaker: FitDittoSpeaker.build_prompt drops the OLDEST whole exchanges
        (user turn + assistant reply) and notes the omission in the system message.
  * Budgets = model context - generation length - margin; for the observed data no prompt
    needs fitting at all (max Planner prompt 14245 tokens vs 32104).
Every fit is recorded (original tokens, removed tokens / dropped exchanges).
"""
from __future__ import annotations

MARKER = "THE CONVERSATION SO FAR\n"
OMITTED = "\n[Earlier dialogue omitted; latest context follows]\n"
PLANNER_CONTEXT, PLANNER_MAX_NEW = 32768, 600
SPEAKER_CONTEXT, SPEAKER_MAX_NEW = 32768, 200
MARGIN = 64
PLANNER_BUDGET = PLANNER_CONTEXT - PLANNER_MAX_NEW - MARGIN
SPEAKER_BUDGET = SPEAKER_CONTEXT - SPEAKER_MAX_NEW - MARGIN
LEGACY_CAPS = (7000, 8000, 12000)
SPEAKER_OMISSION_NOTE = "\n\n[Earlier turns of this conversation are omitted; the most recent ones follow.]"


class TokProxy:
    """Tokenizer wrapper that disables the legacy right-side truncation caps."""

    def __init__(self, tok, budget):
        object.__setattr__(self, "_tok", tok)
        object.__setattr__(self, "_budget", budget)
        object.__setattr__(self, "last_len", None)
        object.__setattr__(self, "n_legacy_calls", 0)

    def __call__(self, text, **kw):
        if kw.get("truncation") and kw.get("max_length") in LEGACY_CAPS:
            kw = {k: v for k, v in kw.items() if k not in ("truncation", "max_length")}
            object.__setattr__(self, "n_legacy_calls", self.n_legacy_calls + 1)
            enc = self._tok(text, **kw)
            n = int(enc["input_ids"].shape[-1]) if hasattr(enc["input_ids"], "shape") else len(enc["input_ids"])
            object.__setattr__(self, "last_len", n)
            if n > self._budget:
                raise ValueError("prompt of %d tokens exceeds the budget %d; fit it before generation"
                                 % (n, self._budget))
            return enc
        return self._tok(text, **kw)

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __setattr__(self, name, value):
        setattr(self._tok, name, value)


def _chat_len(tok, system, user):
    text = tok.apply_chat_template([{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)
    return len(tok(text)["input_ids"])      # same call shape as Planner.raw_with


def fit_planner_user(tok, system, user, budget=PLANNER_BUDGET):
    """Return (user_prompt_that_fits, info). Unchanged when it already fits."""
    n = _chat_len(tok, system, user)
    if n <= budget:
        return user, {"original_tokens": n, "removed_tokens": 0, "compacted": False}
    if MARKER not in user:
        raise ValueError("Planner prompt has no conversation marker; cannot fit safely")
    static, history = user.split(MARKER, 1)
    context = static + MARKER + OMITTED
    static_n = _chat_len(tok, system, context)
    if static_n >= budget - 64:
        raise ValueError("Planner static prefix alone exceeds the budget")
    hist_ids = tok(history, add_special_tokens=False)["input_ids"]
    keep = budget - static_n - 8
    while keep > 32:
        cand = context + tok.decode(hist_ids[-keep:], skip_special_tokens=False)
        m = _chat_len(tok, system, cand)
        if m <= budget:
            return cand, {"original_tokens": n, "removed_tokens": n - m, "compacted": True}
        keep -= max(8, m - budget + 8)
    raise ValueError("cannot fit recent Planner history")


def make_fit_ditto_speaker(DittoSpeaker):
    """Build the subclass lazily so this module imports without sepsim on the path."""

    class FitDittoSpeaker(DittoSpeaker):
        budget = SPEAKER_BUDGET

        def load(self):
            super().load()
            self._raw_tok = self._tok
            self._tok = TokProxy(self._raw_tok, self.budget)
            self.last_fit = None
            return self

        def build_prompt(self, scenario_text, block, hist_u, hist_a, turn):
            raw = getattr(self, "_raw_tok", self._tok)
            text = super().build_prompt(scenario_text, block, hist_u, hist_a, turn)
            n = len(raw(text)["input_ids"])     # same call shape as Speaker.say
            if n <= self.budget:
                self.last_fit = {"original_tokens": n, "dropped_exchanges": 0, "compacted": False}
                return text
            # drop the oldest whole exchanges; the note tells the model something was omitted
            for drop in range(1, len(hist_u)):
                text = super().build_prompt(scenario_text + SPEAKER_OMISSION_NOTE, block,
                                            hist_u[drop:], hist_a[drop:], turn)
                m = len(raw(text)["input_ids"])
                if m <= self.budget:
                    self.last_fit = {"original_tokens": n, "dropped_exchanges": drop,
                                     "final_tokens": m, "compacted": True}
                    return text
            raise ValueError("cannot fit Speaker prompt even with one exchange")

    return FitDittoSpeaker
