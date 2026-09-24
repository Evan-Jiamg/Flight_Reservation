#!/usr/bin/env python3
"""Explicit, content-free stop policy: an empirical per-turn hazard.

For fold f, from the fold's nested INNER-TRAIN gold sessions only: every session contributes
positions 1..K+1 (K+1 = its stop). h_f(t) = (#sessions stopping at t + 0.5) /
(#sessions reaching t + 1). Positions beyond the longest train session use the last estimated
hazard. This is a transparent, inspectable parameter vector (no text is read), used as the
honest comparator for any learned gate. Written in the score_logged_gates.py row format for
every step of the given no-gate episodes whose scenario is inner train/val for that fold.
"""
import argparse
import json
import os
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nested-dir", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    manifest = json.load(open(os.path.join(args.nested_dir, "nested_manifest.json")))
    episodes = [json.loads(l) for l in open(args.episodes, encoding="utf-8") if l.strip()]
    params = {}
    with open(args.out, "w") as out:
        for f in manifest["folds"]:
            fold = f["fold"]
            stop_at = {}
            for l in open(os.path.join(args.nested_dir, "fold%d_inner_train.jsonl" % fold)):
                r = json.loads(l)
                if r["should_stop"]:
                    stop_at[r["record_id"]] = int(r["turn_index"])
            stops = Counter(stop_at.values())
            tmax = max(stop_at.values())
            h = {}
            for t in range(1, tmax + 1):
                reach = sum(1 for s in stop_at.values() if s >= t)
                h[t] = (stops.get(t, 0) + 0.5) / (reach + 1)
            params["f%dturn" % fold] = {"hazard": {str(t): round(v, 4) for t, v in h.items()},
                                        "n_sessions": len(stop_at)}
            side_of = {c: "inner_train" for c in f["inner_train_ids"]}
            side_of.update({c: "inner_validation" for c in f["inner_validation_ids"]})
            for e in episodes:
                side = side_of.get(e["conversation_id"])
                if side is None:
                    continue
                for s in e["trace"]:
                    out.write(json.dumps({"conversation_id": e["conversation_id"], "seed": e["seed"],
                                          "replicate": e.get("replicate", 0), "t": s["t"],
                                          "adapter": "f%dturn" % fold, "adapter_fold": fold,
                                          "scenario_side": side,
                                          "p_stop": h.get(s["t"], h[tmax])}) + "\n")
    json.dump(params, open(args.out + ".params.json", "w"), indent=1)
    print(json.dumps(params))


if __name__ == "__main__":
    main()
