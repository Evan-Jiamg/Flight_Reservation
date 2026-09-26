#!/usr/bin/env python3
"""Label per-turn samples with a LOCAL label judge (never the gpt-5-mini evaluation ledger).

Same prompt builder and budget as the trained judge (goal_judge.fit_messages), greedy decode.
Resumable; every row keeps the raw output; meta records model path, samples sha and the
sha of the judge system prompt, so a label set can be traced to exactly what produced it.
"""
import argparse
import hashlib
import json
import os
import time

import goal_judge as GJ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="", help="'auto' to spread over visible GPUs")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ids", default="", help="optional file with sample ids to label (cross-check subset)")
    args = ap.parse_args()
    samples = [json.loads(l) for l in open(args.samples, encoding="utf-8") if l.strip()]
    if args.ids:
        keep = {l.strip() for l in open(args.ids) if l.strip()}
        samples = [s for s in samples if s["id"] in keep]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out) if l.strip()}
    judge = GJ.GoalJudge(args.model, gpu=args.gpu, dtype=args.dtype, load_4bit=args.load_4bit,
                         device_map=args.device_map or None).load()
    meta = {"model": args.model, "load_4bit": args.load_4bit, "dtype": args.dtype,
            "samples_sha256": hashlib.sha256(open(args.samples, "rb").read()).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(GJ.SYSTEM.encode()).hexdigest(),
            "budget": GJ.JUDGE_BUDGET, "max_new": GJ.MAX_NEW, "n_target": len(samples),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=1)
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        for i, s in enumerate(samples, 1):
            if s["id"] in done:
                continue
            r = judge.assess(s["scenario_text"], s["hist_u"], s["hist_a"])
            f.write(json.dumps({"id": s["id"], **r, "model": args.model}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print("labelled %d/%d unparsed %d (%.0fs)" % (i, len(samples), judge.n_unparsed, time.time() - t0),
                      flush=True)
    print("DONE calls %d unparsed %d" % (judge.n_calls, judge.n_unparsed), flush=True)


if __name__ == "__main__":
    main()
