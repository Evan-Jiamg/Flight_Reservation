#!/usr/bin/env python3
"""Server checks for the judge pipeline (tokenizers + Qwen3-4B weights; run by the coordinator).

1. Tokenizer level (no weights):
   - Qwen3-4B-Instruct-2507 and Llama-3.1-70B: status_token_plan succeeds on real fitted prompts
     (the scored '{"status": "X' ids are an exact prefix of the canonical training target ids),
     one BOS at most.
   - gpt-oss-120b: harmony special tokens are single ids, the chat template renders with
     reasoning_effort, and parse_harmony works on a decode(skip_special_tokens=False) round trip.
2. Model level (Qwen3-4B on --gpu, optional --adapter): for --n samples
   - status_probs has the three keys, each in [0, 1], summing to 1;
   - repeated calls are identical (KV-cache crop leaves no state behind);
   - cached scoring agrees with an uncached full-forward reference;
   - assess() returns the same status_probs; whenever the greedy output starts with the canonical
     head '{"status": "', argmax(status_probs) equals the parsed status.
"""
import argparse
import glob
import json

import goal_judge as GJ
import label_crosscheck as LC

SNAP = "/tmp2/hf_shared/hub/models--%s/snapshots/*/"


def snap(name):
    hits = glob.glob(SNAP % name)
    assert hits, "missing snapshot " + name
    return hits[0]


def load_samples(path, n):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    out = []
    while len(out) < n and any(by_src.values()):          # round-robin over sources, longest first
        for src in sorted(by_src):
            if by_src[src] and len(out) < n:
                lst = sorted(by_src[src], key=lambda r: -len(r["hist_u"]))
                out.append(lst[0])
                by_src[src] = [r for r in by_src[src] if r["id"] != lst[0]["id"]]
    return out


def tokenizer_checks(samples):
    from transformers import AutoTokenizer
    for name in ("Qwen--Qwen3-4B-Instruct-2507", "meta-llama--Meta-Llama-3.1-70B-Instruct"):
        tok = AutoTokenizer.from_pretrained(snap(name))
        for s in samples:
            msgs, info = GJ.fit_messages(tok, s["scenario_text"], s["hist_u"], s["hist_a"])
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            shared, br, n = GJ.status_token_plan(tok, text)
            assert n == info["prompt_tokens"] <= GJ.JUDGE_BUDGET
            if tok.bos_token_id is not None:
                assert shared.count(tok.bos_token_id) <= 1
        print(name, "status plan ok; branch tokens", {k: tok.convert_ids_to_tokens(v) for k, v in br.items()})

    tok = AutoTokenizer.from_pretrained(snap("openai--gpt-oss-120b"))
    for t in ("<|channel|>", "<|message|>", "<|return|>", "<|end|>", "<|start|>", "<|constrain|>"):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, (t, ids)
    s = samples[0]
    msgs, info = GJ.fit_messages(tok, s["scenario_text"], s["hist_u"], s["hist_a"])
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort="low")
    assert "Reasoning: low" in text and GJ.SYSTEM.split("\n")[0] in text, text[:600]
    fx = ('<|channel|>analysis<|message|>Thinking about {"status": "NOT"}.<|end|><|start|>assistant'
          '<|channel|>final<|message|>{"status": "PARTIAL", "unmet": ["license"]}<|return|>')
    rt = tok.decode(tok(fx, add_special_tokens=False)["input_ids"], skip_special_tokens=False)
    res, ok, hinfo = LC.parse_harmony(rt)
    assert ok and res == {"status": "PARTIAL", "unmet": ["license"]}, (rt, hinfo)
    print("gpt-oss tokenizer/template/harmony round trip ok; prompt tokens", info["prompt_tokens"],
          "tail", repr(text[-80:]))


def reference_probs(judge, s):
    """Uncached: one full forward per status over prompt + '{"status": "X'."""
    import torch
    text, _ = judge.prompt_text(s["scenario_text"], s["hist_u"], s["hist_a"])
    shared, br, _ = GJ.status_token_plan(judge.tok, text)
    dev = next(judge.model.parameters()).device
    lps = {}
    with torch.no_grad():
        for st, b in br.items():
            ids = shared + b
            lg = judge.model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0].float()
            lp = torch.log_softmax(lg, -1)
            lps[st] = sum(float(lp[len(shared) - 1 + k, b[k]]) for k in range(len(b)))
    return GJ.normalize_logps(lps)[0]


def model_checks(samples, adapter, gpu):
    judge = GJ.GoalJudge(snap("Qwen--Qwen3-4B-Instruct-2507"), adapter=adapter, gpu=gpu).load()
    n_head = n_agree = 0
    for s in samples:
        p = judge.status_probs(s["scenario_text"], s["hist_u"], s["hist_a"])
        assert set(p) == set(GJ.STATUSES) and all(0 <= v <= 1 for v in p.values())
        assert abs(sum(p.values()) - 1) < 1e-6, p
        p2 = judge.status_probs(s["scenario_text"], s["hist_u"], s["hist_a"])
        assert all(abs(p[k] - p2[k]) < 1e-5 for k in p), (p, p2)
        ref = reference_probs(judge, s)
        assert all(abs(p[k] - ref[k]) < 0.02 for k in p), ("cache vs reference", p, ref)
        a = judge.assess(s["scenario_text"], s["hist_u"], s["hist_a"])
        assert all(abs(a["status_probs"][k] - p[k]) < 1e-5 for k in p)
        head = a["raw"].lstrip().startswith(GJ.STATUS_HEAD)
        arg = max(p, key=p.get)
        if head and a["parse_ok"]:
            n_head += 1
            n_agree += int(arg == a["status"])
            assert arg == a["status"], ("argmax status_probs disagrees with greedy", p, a["raw"])
        print(json.dumps({"id": s["id"], "probs": {k: round(v, 4) for k, v in p.items()},
                          "mass": a["status_probs_mass"], "greedy": a["status"], "parse_ok": a["parse_ok"],
                          "canonical_head": head, "prompt_tokens": a["prompt_tokens"],
                          "raw": a["raw"][:120]}), flush=True)
    print("model checks ok; greedy outputs with canonical head %d/%d, argmax agreement %d/%d"
          % (n_head, len(samples), n_agree, n_head))
    if adapter:
        assert n_head == len(samples), "a trained judge should emit the canonical head on every sample"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="/tmp2/mzjiang_usersim/grpo_planner/v3_run1/judge/samples.jsonl")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()
    samples = load_samples(a.samples, a.n)
    tokenizer_checks(samples)
    if not a.skip_model:
        model_checks(samples, a.adapter or None, a.gpu)
    print("JUDGE SERVER TESTS OK")


if __name__ == "__main__":
    main()
