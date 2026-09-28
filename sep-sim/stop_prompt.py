"""Deterministic stop prompt encoding shared by SFT, evaluation and rollout.

For TREC prompts, preserve the full goal/state/ledger prefix and the most recent
conversation tokens. PRISM prompts retain their original head/tail policy.
"""
DECISION_SYSTEM = (
    "You decide whether a specific human user should stop talking now. "
    "Answer YES if the user would end the session now, before writing another "
    "message. Answer NO if the user still has a message to send. "
    "Return exactly YES or NO."
)
MARKER = "THE CONVERSATION SO FAR\n"
OMITTED = "\n[Earlier dialogue omitted; latest context follows]\n"


def encode_stop_prompt(tokenizer, user, max_length):
    def encode(body):
        chat = tokenizer.apply_chat_template([
            {"role": "system", "content": DECISION_SYSTEM},
            {"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True)
        return tokenizer(chat, add_special_tokens=False,
                         truncation=False)["input_ids"]

    ids = encode(user)
    original = len(ids)
    if original <= max_length:
        return ids, {"original_tokens": original, "removed_tokens": 0,
                     "trec_compacted": False}
    if MARKER not in user:
        keep_head = min(256, max_length // 4)
        kept = ids[:keep_head] + ids[-(max_length - keep_head):]
        return kept, {"original_tokens": original,
                      "removed_tokens": original - len(kept),
                      "trec_compacted": False}
    static, history = user.split(MARKER, 1)
    static += MARKER
    context = static + OMITTED
    static_ids = encode(context)
    if len(static_ids) >= max_length - 64:
        raise ValueError("TREC goal/state prefix exceeds context budget")
    history_ids = tokenizer(history, add_special_tokens=False,
                            truncation=False)["input_ids"]
    budget = max_length - len(static_ids) - 8
    while True:
        tail = tokenizer.decode(history_ids[-budget:],
                                skip_special_tokens=False)
        kept = encode(context + tail)
        if len(kept) <= max_length:
            return kept, {"original_tokens": original,
                          "removed_tokens": original - len(kept),
                          "trec_compacted": True}
        budget -= max(8, len(kept) - max_length + 8)
        if budget < 32:
            raise ValueError("cannot fit recent TREC history")

