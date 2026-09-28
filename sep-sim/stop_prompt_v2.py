"""Stop prompt encoding v2 = stop_prompt.encode_stop_prompt + optional masking of lines that
state turn counts or rule-based stopping outputs (Addendum D). The same function must be
used in training, evaluation and rollout; mask=False reproduces v1 byte for byte."""
import re

from stop_prompt import encode_stop_prompt, DECISION_SYSTEM, MARKER  # noqa: F401

# Lines of the TREC Planner prompt that carry position or sepsim StoppingLedger outputs.
MASK_PATTERNS = [
    r"^\s*- turns so far:",
    r"^\s*- useful replies:",
    r"^\s*- no stopping condition is met",
    r"^\s*- stopping condition",
    r"^\s*- last reply repeated the previous offer:",
    r"^THEY HAVE SENT \d+ MESSAGES? SO FAR",
]
_MASK = [re.compile(p) for p in MASK_PATTERNS]
# Persona trait stays; only the rule-derived parenthetical ("so about 3 unhelpful replies ...") goes.
_GIVES_UP = re.compile(r"^(\s*- gives up: [^(]*?)\s*\(.*\)\s*$")


def mask_rules(user):
    """Remove masked lines from the static prefix only (never from the dialogue history)."""
    if MARKER in user:
        static, history = user.split(MARKER, 1)
        kept = []
        for line in static.split("\n"):
            if any(m.search(line) for m in _MASK):
                continue
            kept.append(_GIVES_UP.sub(r"\1", line))
        return "\n".join(kept) + MARKER + history
    return user


def encode_stop_prompt_v2(tokenizer, user, max_length, mask=False):
    return encode_stop_prompt(tokenizer, mask_rules(user) if mask else user, max_length)
