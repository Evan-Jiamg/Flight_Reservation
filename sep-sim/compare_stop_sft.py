#!/usr/bin/env python3
"""Paired session bootstrap for two frozen stop models on identical positions."""
import argparse
import json
import math
import random
from collections import defaultdict


def read(path):
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    mapping = {(r["record_id"], int(r["turn_index"])): r for r in rows}
    if len(mapping) != len(rows):
        raise ValueError("duplicate evaluation positions")
    return mapping


def summarize(rows):
    cont = [r for r in rows if not r["target_stop"]]
    ends = [r for r in rows if r["target_stop"]]
    if not cont or not ends:
        raise ValueError("both continuation and K+1 stop examples are required")
    auc = sum((a["p_stop"] > b["p_stop"]) +
              0.5 * (a["p_stop"] == b["p_stop"])
              for a in ends for b in cont) / (len(ends) * len(cont))
    nll = -sum(math.log(max(1e-8, min(1 - 1e-8, r["p_stop"] if r["target_stop"]
                                                   else 1 - r["p_stop"])))
               for r in rows) / len(rows)
    return {"sessions": len({r["record_id"] for r in rows}),
            "n_cont": len(cont), "n_k1": len(ends),
            "false_stop": sum(r["stop"] for r in cont) / len(cont),
            "k1_end": sum(r["stop"] for r in ends) / len(ends),
            "auc": auc, "nll": nll}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    base, new = read(args.base), read(args.new)
    if base.keys() != new.keys():
        raise ValueError("paired evaluation positions differ")
    for key in base:
        if base[key]["target_stop"] != new[key]["target_stop"]:
            raise ValueError("paired labels differ")
    grouped = defaultdict(list)
    for key in sorted(base):
        grouped[key[0]].append(key)
    first, second = summarize(list(base.values())), summarize(list(new.values()))
    metrics = ("false_stop", "k1_end", "auc", "nll")
    draws = {m: [] for m in metrics}
    rng = random.Random(20260923)
    ids = sorted(grouped)
    for _ in range(args.bootstrap):
        sampled = [key for sid in rng.choices(ids, k=len(ids)) for key in grouped[sid]]
        a = summarize([base[k] for k in sampled])
        b = summarize([new[k] for k in sampled])
        for metric in metrics:
            draws[metric].append(b[metric] - a[metric])
    effects = {}
    for metric, values in draws.items():
        values.sort()
        effects[metric] = {"new_minus_base": second[metric] - first[metric],
                           "cluster_bootstrap_95": [values[int(.025 * len(values))],
                                                    values[int(.975 * len(values))]]}
    print(json.dumps({"base": first, "new": second, "paired": effects}, indent=2))


if __name__ == "__main__":
    main()
