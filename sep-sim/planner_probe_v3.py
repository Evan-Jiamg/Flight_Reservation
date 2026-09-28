#!/usr/bin/env python3
"""Smaller-Planner probe under the NEW (v3) Planner architecture, teacher-forced on logged trajectories.

For each step t of the 20 probe episodes (Ditto rep0, seed 0; same selection as planner_probe.py):
  * system prompt  = planner_prompt_v3.system_prompt_v3()   (end_session, no band, no override)
  * user prompt    = planner_prompt_v3.user_prompt_v3(...) with
      - history      = the logged user/assistant turns before t
      - prev_block   = the logged previous state block, with its lexical "- pending:" line replaced by
                       the goal-status unmet list of step t-1 (what A2 would have rendered)
      - ledger       = replayed on the logged replies (turn count); no gain line, because the
                       candidate has no gain estimates of its own in a teacher-forced probe
      - GOAL STATUS  = the LABEL JUDGE's label (Llama-3.1-70B) for the history before t, standing in
                       for the trained judge; t = 1 -> NOT ASSESSED
  * parsed by read_plan_v3: stop = end_session; length = the model's own number.
Teacher forcing means history and previous states come from the logged 32B run, not from the
candidate itself, so this measures capability on identical inputs, not free-running behaviour.
Reference: the logged 32B (original prompt) end signal for the same steps ("32B kept data").
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/mzjiang/Sep-Simulator")
os.environ["SEPSIM_ACT_PRIOR"] = "nostopclobber"
sys.modules.setdefault("torchvision", None)
sys.modules.setdefault("torchaudio", None)

STATE_RE = re.compile(r"THE STATE YOU WROTE LAST TURN\n(.*?)\n\n", re.S)


def label_id(cid, t_after):
    return "ditto_rep0|%s|s0|r0|t%d" % (cid, t_after)


def build(args):
    """-> probe_v3 rows (one per step) with everything needed to render the v3 prompt."""
    from sepsim import stopping
    import planner_prompt_v3 as V3
    eps = [json.loads(l) for p in sorted(glob.glob(args.episodes)) for l in open(p)]
    eps = [e for e in eps if e["seed"] == 0]
    eps.sort(key=lambda e: hashlib.sha256(e["conversation_id"].encode()).hexdigest())
    chosen = eps[:20]
    recs = {json.loads(l)["conversation_id"]: json.loads(l) for l in open(args.corpus) if l.strip()}
    labels = {}
    for l in open(args.labels):
        r = json.loads(l)
        labels[r["id"]] = {"status": r["status"] if r.get("parse_ok") else "UNKNOWN", "unmet": r.get("unmet", [])}
    missing = []
    with open(args.out, "w", encoding="utf-8") as f:
        for e in chosen:
            cid = e["conversation_id"]
            scenario = recs[cid]["scenario"]
            hu, ha = [], []
            for s in e["trace"]:
                t = s["t"]
                gs = {"status": "NOT ASSESSED", "unmet": []} if t == 1 else labels.get(label_id(cid, t - 1))
                if gs is None:
                    missing.append(label_id(cid, t - 1))
                    gs = {"status": "UNKNOWN", "unmet": []}
                m = STATE_RE.search(s["gate_prompt"])
                prev_block = m.group(1)
                gs_prev = {"status": "NOT ASSESSED", "unmet": []} if t <= 2 else \
                    labels.get(label_id(cid, t - 2), {"status": "UNKNOWN", "unmet": []})
                lines = [l for l in prev_block.split("\n") if not l.startswith("- put aside:")]
                lines = [("- pending: " + V3.unmet_text(gs_prev)) if l.startswith("- pending:") else l for l in lines]
                prev_block = "\n".join(lines)
                f.write(json.dumps({"key": "%s|s0" % cid, "t": t, "conversation_id": cid, "scenario": scenario,
                                    "hist_u": list(hu), "hist_a": list(ha), "prev_block": prev_block,
                                    "goal_status": gs, "logged": {"ended_planner": s.get("ended_planner"),
                                                                  "move": s.get("move"), "act": s.get("act")}},
                                   ensure_ascii=False) + "\n")
                if s["decision"] != "continue":
                    break
                hu.append(s["user"])
                ha.append(s["agent"])
    print("probe_v3 rows written; missing labels:", len(missing), missing[:3])
    if missing:
        raise SystemExit("labels missing for the probe episodes; label them first")


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sepsim import stopping
    import fit_prompts as F
    import planner_prompt_v3 as V3
    rows = [json.loads(l) for l in open(args.probe, encoding="utf-8") if l.strip()]
    torch.cuda.init()
    torch.cuda.set_device(args.gpu)
    torch.cuda.reset_peak_memory_stats(args.gpu)
    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"low_cpu_mem_usage": True, "device_map": {"": args.gpu}}
    if args.nf4:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    else:
        kw.update(F.dtype_kwarg("auto" if args.dtype == "auto" else getattr(torch, args.dtype)))
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    SYS = V3.system_prompt_v3()
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            led = stopping.StoppingLedger(r["scenario"])
            prev = ""
            for a in r["hist_a"]:
                led.observe({}, a, prev)
                prev = a
            up = V3.user_prompt_v3(r["scenario"], r["prev_block"], r["hist_u"], r["hist_a"], r["t"], led,
                                   r["goal_status"], prev_ann={})
            up, info = F.fit_planner_user(tok, SYS, up, budget=min(F.PLANNER_BUDGET, args.context - 664))
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
            fields, diag, end = V3.read_plan_v3(raw, r["t"], r["scenario"], rng, led)
            f.write(json.dumps({"key": r["key"], "t": r["t"], "parsed": fields is not None,
                                "end_session": bool(end), "end_valid": bool(diag.get("end_session_valid")),
                                "length_words": (fields or {}).get("length_words"),
                                "patience": (fields or {}).get("patience"), "stop_rule": (fields or {}).get("stop_rule"),
                                "move": (fields or {}).get("move"), "goal_status": r["goal_status"]["status"],
                                "logged": r["logged"], "latency_s": round(dt, 2), "new_tokens": int(new.shape[0]),
                                "prompt_tokens": int(enc["input_ids"].shape[1]), "fit": info, "raw": raw},
                               ensure_ascii=False) + "\n")
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

    def episode_err(rows, flag):
        by = defaultdict(list)
        for r in rows:
            by[r["key"]].append(r)
        err, n_end = [], 0
        for key, rs in by.items():
            rs.sort(key=lambda r: r["t"])
            t = next((r["t"] for r in rs if flag(r)), None)
            if t is not None:
                n_end += 1
                err.append((t - 1) - K[key.split("|")[0]])     # silent exit: t-1 turns emitted
        return n_end, len(by), err

    for path in args.results:
        rows = [json.loads(l) for l in open(path)]
        meta = json.load(open(path + ".meta.json")) if os.path.exists(path + ".meta.json") else {}
        n = len(rows)
        n_end, n_ep, err = episode_err(rows, lambda r: r["end_session"])
        by_status = defaultdict(lambda: [0, 0])
        for r in rows:
            by_status[r["goal_status"]][0] += r["end_session"]
            by_status[r["goal_status"]][1] += 1
        out = {"model": meta.get("model", path).rstrip("/").split("/")[-3:], "peak_gib": meta.get("peak_gib"),
               "n": n, "parse_rate": sum(r["parsed"] for r in rows) / n,
               "end_valid_rate": sum(r["end_valid"] for r in rows) / n,
               "episodes_with_end": "%d/%d" % (n_end, n_ep),
               "first_end_emitted_minus_K_mean": sum(err) / len(err) if err else None,
               "first_end_abs_err_mean": sum(abs(e) for e in err) / len(err) if err else None,
               "P(end|status)": {k: "%d/%d" % tuple(v) for k, v in sorted(by_status.items())},
               "length_words_median": sorted(r["length_words"] for r in rows if r["length_words"])[
                   len([r for r in rows if r["length_words"]]) // 2] if any(r["length_words"] for r in rows) else None,
               "latency_s_mean": sum(r["latency_s"] for r in rows) / n}
        print(json.dumps(out))
    rows = [json.loads(l) for l in open(args.results[0])]
    n_end, n_ep, err = episode_err(rows, lambda r: bool(r["logged"]["ended_planner"]))
    print(json.dumps({"reference": "logged 32B, original prompt (kept data), silent-exit convention",
                      "episodes_with_end": "%d/%d" % (n_end, n_ep),
                      "first_end_abs_err_mean": sum(abs(e) for e in err) / len(err) if err else None,
                      "first_end_emitted_minus_K_mean": sum(err) / len(err) if err else None}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--episodes", required=True)
    b.add_argument("--labels", required=True)
    b.add_argument("--corpus", default="/home/mzjiang/v5-latency/data.jsonl")
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--probe", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gpu", type=int, default=1)
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--nf4", action="store_true")
    r.add_argument("--context", type=int, default=32768)
    s = sub.add_parser("summary")
    s.add_argument("results", nargs="+")
    s.add_argument("--corpus", default="/home/mzjiang/v5-latency/data.jsonl")
    a = ap.parse_args()
    {"build": build, "run": run, "summary": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
