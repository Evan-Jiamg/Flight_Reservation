#!/usr/bin/env python3
"""Smaller-Planner probe: the SAME logged Planner prompts, teacher-forced, through a candidate model.

Probe set: 20 Ditto rep0 no-gate episodes (all their steps), chosen by sha256 order of
(conversation_id) among distinct scenarios, seed 0 -- outcome-blind. Each step's exact Planner
user prompt (as logged) is sent with the ORIGINAL system prompt (the prompts were produced for it),
fitted with fit_prompts (no truncation), greedy, max_new 600 -- the production call shape.
Parsed with sepsim read_plan (override ON, as when the prompts were logged), fresh ledger.

Reported per candidate: JSON parse rate, valid act distribution rate, agreement with the logged
32B on move and on the session-ending signal, first end turn per episode (teacher-forced on the
32B trajectory, so it is a capability probe, not a free-running result) vs human K, stop_rule
distribution, latency and peak GPU memory. Caveat: models run on different hosts/dtypes where
stated; numbers compare capability, not bit-exact behaviour.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/mzjiang/Sep-Simulator")
os.environ["SEPSIM_ACT_PRIOR"] = "off"
sys.modules.setdefault("torchvision", None)
sys.modules.setdefault("torchaudio", None)


def build_probe_set(episodes_glob, corpus, out, n_episodes=20):
    import glob
    eps = [json.loads(l) for p in sorted(glob.glob(episodes_glob)) for l in open(p)]
    eps = [e for e in eps if e["seed"] == 0]
    eps.sort(key=lambda e: hashlib.sha256(e["conversation_id"].encode()).hexdigest())
    chosen = eps[:n_episodes]
    recs = {json.loads(l)["conversation_id"]: json.loads(l) for l in open(corpus) if l.strip()}
    with open(out, "w", encoding="utf-8") as f:
        for e in chosen:
            for s in e["trace"]:
                f.write(json.dumps({"key": "%s|s%d" % (e["conversation_id"], e["seed"]), "t": s["t"],
                                    "conversation_id": e["conversation_id"],
                                    "scenario": recs[e["conversation_id"]]["scenario"],
                                    "planner_prompt": s["gate_prompt"],
                                    "logged": {"move": s.get("move"), "act": s.get("act"),
                                               "stop_rule": s.get("stop_rule"),
                                               "ended_planner": s.get("ended_planner")}},
                                   ensure_ascii=False) + "\n")
    print("probe set:", len(chosen), "episodes", sum(len(e["trace"]) for e in chosen), "prompts")


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sepsim import planner_prompt as PP, state, stopping
    import fit_prompts as F
    rows = [json.loads(l) for l in open(args.probe, encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"low_cpu_mem_usage": True, "device_map": {"": args.gpu}}
    if args.nf4:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, args.dtype),
                                                       bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    elif args.dtype != "auto":
        kw.update(F.dtype_kwarg(getattr(torch, args.dtype)))
    else:
        kw.update(F.dtype_kwarg("auto"))
    torch.cuda.init()                         # CUDA must be initialised before resetting its stats
    torch.cuda.set_device(args.gpu)
    torch.cuda.reset_peak_memory_stats(args.gpu)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    SYS = PP.system_prompt()
    done = set()
    if os.path.exists(args.out):
        done = {(json.loads(l)["key"], json.loads(l)["t"]) for l in open(args.out)}
    with open(args.out, "a", encoding="utf-8") as f:
        for r in rows:
            if (r["key"], r["t"]) in done:
                continue
            up, info = F.fit_planner_user(tok, SYS, r["planner_prompt"],
                                          budget=min(F.PLANNER_BUDGET, args.context - 600 - 64))
            text = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": up}],
                                           tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda:%d" % args.gpu)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=600, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            dt = time.time() - t0
            new = out[0][enc["input_ids"].shape[1]:]
            raw = tok.decode(new, skip_special_tokens=True)
            rng = random.Random(int(hashlib.sha256(("%s|%d" % (r["key"], r["t"])).encode()).hexdigest()[:8], 16))
            fields, diag = PP.read_plan(raw, r["t"], r["scenario"], rng, stopping.StoppingLedger(r["scenario"]))
            d = state.json_of(raw) or {}
            dist_ok = isinstance(d.get("act_distribution"), list) and len(d.get("act_distribution")) > 0
            f.write(json.dumps({"key": r["key"], "t": r["t"], "parsed": fields is not None, "dist_ok": dist_ok,
                                "move": fields.get("move") if fields else None,
                                "act": fields.get("act") if fields else None,
                                "stop_rule": fields.get("stop_rule") if fields else None,
                                "ended": bool(state.ends_session(fields)) if fields else False,
                                "logged": r["logged"], "latency_s": round(dt, 2), "new_tokens": int(new.shape[0]),
                                "prompt_tokens": int(enc["input_ids"].shape[1]), "fit": info,
                                "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
    peak = torch.cuda.max_memory_allocated(args.gpu) / 2 ** 30
    json.dump({"model": args.model, "nf4": args.nf4, "dtype": args.dtype, "peak_gib": round(peak, 1),
               "host": os.uname().nodename}, open(args.out + ".meta.json", "w"), indent=1)
    print("done", args.model, "peak GiB %.1f" % peak, flush=True)


def summarize(args):
    from collections import Counter, defaultdict
    K = {}
    for l in open(args.corpus):
        r = json.loads(l)
        K[r["conversation_id"]] = sum(1 for m in r["chat_messages"] if m["participant_name"].lower() == "user")
    for path in args.results:
        rows = [json.loads(l) for l in open(path)]
        meta = json.load(open(path + ".meta.json")) if os.path.exists(path + ".meta.json") else {}
        n = len(rows)
        parsed = [r for r in rows if r["parsed"]]
        by_ep = defaultdict(list)
        for r in rows:
            by_ep[r["key"]].append(r)
        first_end, err = [], []
        for key, rs in by_ep.items():
            rs.sort(key=lambda r: r["t"])
            t = next((r["t"] for r in rs if r["ended"]), None)
            first_end.append(t)
            if t is not None:
                err.append(t - K[key.split("|")[0]])
        out = {"model": meta.get("model", path), "host": meta.get("host"), "peak_gib": meta.get("peak_gib"),
               "n": n, "parse_rate": len(parsed) / n, "dist_ok_rate": sum(r["dist_ok"] for r in rows) / n,
               "move_agree_logged32B": sum(r["move"] == r["logged"]["move"] for r in parsed) / max(1, len(parsed)),
               "end_agree_logged32B": sum(r["ended"] == bool(r["logged"]["ended_planner"]) for r in rows) / n,
               "end_rate": sum(r["ended"] for r in rows) / n,
               "episodes_with_end": sum(t is not None for t in first_end), "episodes": len(first_end),
               "first_end_minus_K_mean": sum(err) / len(err) if err else None,
               "first_end_abs_err_mean": sum(abs(e) for e in err) / len(err) if err else None,
               "stop_rules": dict(Counter(r["stop_rule"] for r in parsed).most_common(6)),
               "latency_s_mean": sum(r["latency_s"] for r in rows) / n}
        print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--episodes", required=True)
    b.add_argument("--corpus", default="/home/mzjiang/v5-latency/data.jsonl")
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--probe", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gpu", type=int, default=0)
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--nf4", action="store_true")
    r.add_argument("--context", type=int, default=32768)
    s = sub.add_parser("summary")
    s.add_argument("results", nargs="+")
    s.add_argument("--corpus", default="/home/mzjiang/v5-latency/data.jsonl")
    a = ap.parse_args()
    if a.cmd == "build":
        build_probe_set(a.episodes, a.corpus, a.out)
    elif a.cmd == "run":
        run(a)
    else:
        summarize(a)


if __name__ == "__main__":
    main()
