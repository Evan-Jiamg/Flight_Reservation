#!/usr/bin/env python3
"""How much of the stop gate's ranking is explained by the turn index alone?

For each prediction file (eval_stop_sft.py output), compare:
  gate AUC            p_stop ranking of K+1 vs continuation positions
  turn-only AUC       ranking by turn_index alone (later = more likely stop)
  within-turn AUC     gate AUC restricted to pairs with the SAME turn_index (content signal
                      that cannot come from position); pooled over turn indices
plus the Spearman correlation of p_stop with turn_index on continuation positions.
"""
import json
import sys
from collections import defaultdict


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** .5
    vb = sum((y - mb) ** 2 for y in rb) ** .5
    return cov / (va * vb) if va and vb else None


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pos = [r for r in rows if r["target_stop"]]
    neg = [r for r in rows if not r["target_stop"]]
    g = auc([r["p_stop"] for r in pos], [r["p_stop"] for r in neg])
    t = auc([int(r["turn_index"]) for r in pos], [int(r["turn_index"]) for r in neg])
    num = den = 0.0
    by_t = defaultdict(lambda: ([], []))
    for r in rows:
        by_t[int(r["turn_index"])][0 if r["target_stop"] else 1].append(r["p_stop"])
    for ti, (p, n) in by_t.items():
        if p and n:
            num += auc(p, n) * len(p) * len(n)
            den += len(p) * len(n)
    within = num / den if den else None
    rho = spearman([r["p_stop"] for r in neg], [int(r["turn_index"]) for r in neg])
    print(json.dumps({"file": path.split("/")[-1], "n_pos": len(pos), "n_neg": len(neg),
                      "gate_auc": round(g, 4), "turn_only_auc": round(t, 4),
                      "within_same_turn_auc": within and round(within, 4),
                      "within_pairs": int(den), "spearman_pstop_turn_on_continue": rho and round(rho, 4)}))
