#!/usr/bin/env python3
"""How often did the Speaker prompt exceed models.Speaker's 7000-token right-side truncation?

Speaker prompts are not logged, so this computes a LOWER BOUND: the role-flipped dialogue
history alone (Ditto chat template, as in DittoSpeaker.build_prompt) at each step t, plus the
Planner prompt length (logged gate_prompt, same history + goal/state) as a close proxy.
Right-side truncation drops the newest turns AND the generation header.
"""
import glob
import json
import sys

from transformers import AutoTokenizer

DITTO = "/tmp2/mzjiang_usersim/models/Ditto-8B"


def main(pattern):
    tok = AutoTokenizer.from_pretrained(DITTO)
    eps = [json.loads(l) for p in sorted(glob.glob(pattern)) for l in open(p)]
    hist_len, plan_len, by_t = [], [], {}
    for e in eps:
        hu, ha = [], []
        for s in e["trace"]:
            msgs = [{"role": "system", "content": "x"}]
            for i, u in enumerate(hu):
                msgs.append({"role": "assistant", "content": u})
                if i < len(ha):
                    msgs.append({"role": "user", "content": ha[i]})
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            n = len(tok(text)["input_ids"])
            p = len(tok(s["gate_prompt"])["input_ids"])
            hist_len.append(n)
            plan_len.append(p)
            by_t.setdefault(s["t"], []).append((n, p))
            if s["decision"] == "continue":
                hu.append(s["user"])
                ha.append(s["agent"])
    over = lambda xs, k: sum(x > k for x in xs)
    print(json.dumps({"steps": len(hist_len),
                      "history_only_over_7000 (lower bound)": over(hist_len, 7000),
                      "planner_prompt_over_7000 (proxy)": over(plan_len, 7000),
                      "by_turn": {t: {"n": len(v), "hist>7000": sum(a > 7000 for a, _ in v),
                                      "planner>7000": sum(b > 7000 for _, b in v),
                                      "hist_median": sorted(a for a, _ in v)[len(v) // 2]}
                                  for t, v in sorted(by_t.items())}}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
