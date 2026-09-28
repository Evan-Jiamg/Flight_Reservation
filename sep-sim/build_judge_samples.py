#!/usr/bin/env python3
"""Per-turn samples for the goal-satisfaction judge (inputs only; labels come separately).

A sample = (conversation so far, ending with the assistant reply of exchange t). Sources:
  * logged no-gate Ditto episodes (continue steps only; the user text and R0 reply as logged),
  * the human TREC session of each scenario (real user and platform agent), exchanges 1..K.
Only scenarios in the Stage C union (inner train/validation of some outer fold) are used; the
per-(fold, group) restriction is applied at training time. The scenario text is
sepsim.pipeline.scenario_text (what the Speaker reads).
"""
import argparse
import glob
import hashlib
import json
import sys

sys.path.insert(0, "/home/mzjiang/Sep-Simulator")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", action="append", required=True, help="source=glob")
    ap.add_argument("--corpus", default="/home/mzjiang/v5-latency/data.jsonl")
    ap.add_argument("--union", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sepsim import pipeline
    recs = {}
    for l in open(a.corpus, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            recs[r["conversation_id"]] = r
    union = {s["conversation_id"] for s in json.load(open(a.union))["scenarios"]}
    counts = {}
    with open(a.out, "w", encoding="utf-8") as out:
        def emit(sid, source, cid, seed, rep, t, hu, ha):
            out.write(json.dumps({"id": sid, "source": source, "conversation_id": cid, "seed": seed,
                                  "replicate": rep, "t": t, "scenario_text": pipeline.scenario_text(recs[cid]),
                                  "hist_u": hu, "hist_a": ha}, ensure_ascii=False) + "\n")
            counts[source] = counts.get(source, 0) + 1
        for spec in a.episodes:
            source, pat = spec.split("=", 1)
            for p in sorted(glob.glob(pat)):
                for l in open(p, encoding="utf-8"):
                    e = json.loads(l)
                    cid = e["conversation_id"]
                    if cid not in union:
                        raise SystemExit("episode outside the union: %s" % cid)
                    hu, ha = [], []
                    for s in e["trace"]:
                        if s["decision"] != "continue":
                            break
                        hu.append(s["user"])
                        ha.append(s["agent"])
                        emit("%s|%s|s%d|r%d|t%d" % (source, cid, e["seed"], e.get("replicate", 0), s["t"]),
                             source, cid, e["seed"], e.get("replicate", 0), s["t"], list(hu), list(ha))
        for cid in sorted(union):
            users, agents = pipeline.split_messages(recs[cid])
            hu, ha = [], []
            for t, u in enumerate(users, 1):
                if t - 1 >= len(agents):
                    break
                hu.append(u["text"])
                ha.append(agents[t - 1]["text"])
                emit("human|%s|t%d" % (cid, t), "human", cid, -1, -1, t, list(hu), list(ha))
    meta = {"counts": counts, "union_scenarios": len(union), "episodes": a.episodes,
            "sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest()}
    json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
