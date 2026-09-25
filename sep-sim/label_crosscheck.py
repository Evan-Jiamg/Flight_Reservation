#!/usr/bin/env python3
"""S1 label-quality cross-check: re-label a seeded, source-stratified subset with gpt-oss-120b.

1. Choose N sample ids (default 200) from the PARSED primary labels (Llama-3.1-70B), stratified by
   `source` (largest-remainder proportional allocation, seed declared). The id list is written to
   <out>.ids.txt; on resume it must be identical.
2. Label them with gpt-oss-120b using the SAME prompt builder (goal_judge.fit_messages ->
   apply_chat_template, add_special_tokens=False). gpt-oss speaks the harmony format: the decoded
   output (special tokens kept) holds an analysis channel and then a final channel. Only the text
   of the LAST final-channel message is parsed (goal_judge.parse). No final channel, or a final
   message that does not parse, is recorded as status UNKNOWN with the reason -- never guessed.
3. Rows use the primary label schema {id, status, unmet, parse_ok, raw, prompt_tokens,
   dropped_exchanges, model} plus harmony diagnostics. Resumable (append, skip done ids).
4. Agreement via label_agreement (load, kappa) on common parsed ids; gate printed:
   PASS if Cohen's kappa >= --kappa-gate (declared, default 0.4), else FAIL.

Declared generation config (logged in <out>.meta.json): greedy, reasoning_effort (default low),
max_new (default 4096). The gpt-oss chat template stamps the current date into its system
message; the rendered prompt's sha256 is stored per row so the exact input is traceable.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import goal_judge as GJ
import judge_metrics as JM
import label_agreement as LA

FINAL_HEADER = re.compile(r"<\|channel\|>\s*final\b[^<]*(?:<\|constrain\|>[^<]*)?<\|message\|>")
END_TOKENS = ("<|return|>", "<|end|>", "<|call|>", "<|start|>")
CHANNEL = re.compile(r"<\|channel\|>\s*([A-Za-z_]+)")


def extract_final(decoded):
    """Harmony output (decoded WITH special tokens) -> (final_text or None, info).

    Takes the last '<|channel|>final ... <|message|>' header and the text up to the next end
    token (<|return|>, <|end|>, <|call|>, <|start|>) or the end of the string."""
    decoded = decoded or ""
    heads = list(FINAL_HEADER.finditer(decoded))
    info = {"channels": CHANNEL.findall(decoded), "n_final": len(heads)}
    if not heads:
        info["reason"] = "no_final_channel"
        return None, info
    start = heads[-1].end()
    ends = [i for i in (decoded.find(t, start) for t in END_TOKENS) if i >= 0]
    stop = min(ends) if ends else len(decoded)
    info["final_terminated"] = bool(ends)
    return decoded[start:stop].strip(), info


def parse_harmony(decoded):
    """-> (result, ok, info): goal_judge.parse applied to the final channel only."""
    final, info = extract_final(decoded)
    if final is None:
        return {"status": "UNKNOWN", "unmet": []}, False, info
    res, ok = GJ.parse(final)
    if not ok:
        info["reason"] = "final_unparsed"
    return res, ok, info


def choose_ids(labels, samples, n, seed):
    """Seeded, source-stratified choice among samples that have a parsed primary label."""
    by_src = {}
    for s in samples:
        if s["id"] in labels:
            by_src.setdefault(s["source"], []).append(s["id"])
    return JM.stratified_sample(by_src, n, seed)


def agreement(primary_path, check_path, gate):
    A, B = LA.load(primary_path), LA.load(check_path)
    common = sorted(set(A) & set(B))
    pairs = [(A[i], B[i]) for i in common]
    k = LA.kappa(pairs)
    from collections import Counter
    rep = {"n_common": len(common), "agreement": sum(a == b for a, b in pairs) / max(1, len(pairs)),
           "cohen_kappa": k, "kappa_gate": gate,
           "confusion_primary_by_check": {"%s|%s" % x: v for x, v in sorted(Counter(pairs).items())},
           "dist_primary": dict(Counter(a for a, _ in pairs)), "dist_check": dict(Counter(b for _, b in pairs))}
    rep["gate"] = "PASS" if (k is not None and k >= gate) else "FAIL"
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--labels", required=True, help="primary labels (Llama-3.1-70B)")
    ap.add_argument("--model", required=True, help="gpt-oss-120b snapshot dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--reasoning-effort", default="low", choices=("low", "medium", "high"))
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--kappa-gate", type=float, default=0.4)
    ap.add_argument("--agreement-only", action="store_true", help="skip labelling; recompute agreement")
    args = ap.parse_args()

    labels, lstats = JM.load_labels(args.labels)
    samples = JM.load_samples(args.samples)
    ids = choose_ids(labels, samples, args.n, args.seed)
    ids_path = args.out + ".ids.txt"
    if os.path.exists(ids_path):
        old = [l.strip() for l in open(ids_path, encoding="utf-8") if l.strip()]
        if old != ids:
            raise SystemExit("existing %s differs from the seeded choice; refusing to mix subsets" % ids_path)
    else:
        with open(ids_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")

    if not args.agreement_only:
        run_labelling(args, samples, ids, lstats)
    rep = agreement(args.labels, args.out, args.kappa_gate)
    rows = [json.loads(l) for l in open(args.out, encoding="utf-8") if l.strip()]
    rep["check_rows"] = len(rows)
    rep["check_unparsed"] = sum(not r["parse_ok"] for r in rows)
    rep["check_fail_reasons"] = {}
    for r in rows:
        if not r["parse_ok"]:
            k = r.get("parse_fail_reason", "?")
            rep["check_fail_reasons"][k] = rep["check_fail_reasons"].get(k, 0) + 1
    rep["primary_label_stats"] = lstats
    rep["files_sha256"] = {"samples": JM.sha256_file(args.samples), "labels": JM.sha256_file(args.labels),
                           "check": JM.sha256_file(args.out), "ids": JM.sha256_file(ids_path)}
    json.dump(rep, open(args.out + ".agreement.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print("KAPPA GATE (>= %.2f): %s  kappa=%s n=%d" % (args.kappa_gate, rep["gate"], rep["cohen_kappa"],
                                                        rep["n_common"]))


def run_labelling(args, samples, ids, lstats):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fit_prompts import dtype_kwarg
    torch.manual_seed(args.seed)
    by_id = {s["id"]: s for s in samples}
    todo = [by_id[i] for i in ids]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8") if l.strip()}
    tok = AutoTokenizer.from_pretrained(args.model)
    dt = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device_map,
                                                 low_cpu_mem_usage=True, **dtype_kwarg(dt))
    model.eval()
    tmpl_kw = {"reasoning_effort": args.reasoning_effort}
    meta = {"model": args.model, "dtype": args.dtype, "device_map": args.device_map,
            "decode": "greedy", "reasoning_effort": args.reasoning_effort, "max_new": args.max_new,
            "n": args.n, "seed": args.seed, "stratified_by": "source",
            "samples_sha256": JM.sha256_file(args.samples), "labels_sha256": JM.sha256_file(args.labels),
            "primary_label_stats": lstats,
            "system_prompt_sha256": hashlib.sha256(GJ.SYSTEM.encode()).hexdigest(),
            "budget": GJ.JUDGE_BUDGET, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "argv": sys.argv}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=1)
    dev = next(model.parameters()).device
    n_fail, t0 = 0, time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        for i, s in enumerate(todo, 1):
            if s["id"] in done:
                continue
            msgs, info = GJ.fit_messages(tok, s["scenario_text"], s["hist_u"], s["hist_a"])
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **tmpl_kw)
            enc = tok(text, return_tensors="pt", add_special_tokens=False)
            enc.pop("token_type_ids", None)
            n_prompt = int(enc["input_ids"].shape[1])
            # fit_messages measured the template without reasoning_effort; re-check the real prompt
            assert n_prompt <= GJ.JUDGE_BUDGET, "prompt %d > budget %d" % (n_prompt, GJ.JUDGE_BUDGET)
            enc = enc.to(dev)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
            gen = out[0][n_prompt:]
            raw = tok.decode(gen, skip_special_tokens=False)
            res, ok, hinfo = parse_harmony(raw)
            n_fail += int(not ok)
            row = {"id": s["id"], **res, "parse_ok": ok, "raw": raw, "prompt_tokens": n_prompt,
                   "dropped_exchanges": info["dropped_exchanges"], "model": args.model,
                   "gen_tokens": int(gen.shape[0]), "hit_max_new": int(gen.shape[0]) >= args.max_new,
                   "harmony": hinfo, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}
            if not ok:
                row["parse_fail_reason"] = hinfo.get("reason", "?")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print("labelled %d/%d unparsed %d (%.0fs)" % (i, len(todo), n_fail, time.time() - t0), flush=True)
    print("DONE crosscheck labelling; unparsed this run %d" % n_fail, flush=True)


if __name__ == "__main__":
    main()
