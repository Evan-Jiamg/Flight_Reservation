#!/usr/bin/env python3
"""OUR diagnostic (not a benchmark metric): the F6 diversity members computed on the SELECTED output only.

The benchmark (docs/metric_specs.md; tools/metrics/surface.py) computes distinct1 / distinct2 /
first_turn_dup_rate over EVERY first-turn candidate (greedy + samples); that official definition is what
is compared with E1.6. This script copies those formulas line for line (lower-cased whitespace split,
bigrams over the flattened stream, duplicate rate over the non-empty strings) and reports them on the
single emitted first message per conversation, labelled as a diagnostic. The "official" column here is
a re-implementation for side-by-side reading; the number to cite is score_method.py's.
"""
import json
import sys


def f6(firsts):
    firsts = [s for s in firsts if s.strip()]
    uni = [w for s in firsts for w in s.lower().split()]
    big = list(zip(uni, uni[1:]))
    return {"distinct1": len(set(uni)) / max(1, len(uni)), "distinct2": len(set(big)) / max(1, len(big)),
            "first_turn_dup_rate": 1 - len(set(firsts)) / max(1, len(firsts)), "n": len(firsts)}


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    first = [r for r in rows if r.get("is_first_turn")]
    official = f6([s for r in first for s in ([r.get("greedy") or ""] + list(r.get("samples") or []))])
    selected = f6([r.get("greedy") or "" for r in first])
    print(json.dumps({"official_all_candidates": official, "diagnostic_selected_only": selected}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
