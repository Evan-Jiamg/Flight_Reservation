#!/usr/bin/env python3
"""Task 1 (M2) metrics pooled over folds, and paired comparisons between arms (SPEC v16 item 8, user 2026-09-28).

One fold's test split has ~9 conversations, i.e. ~9 end events: too few for a Task 1 conclusion. This tool pools
the test generations of several folds (task1_v4.py outputs: FILE.jsonl rows per real user turn with greedy_ended,
FILE.jsonl.k1.jsonl the K+1 probe per conversation) and scores them with exactly the task1_stop definitions:
  TP = K+1 ended, FN = K+1 not ended, FP = END flags (greedy_ended) on the n real rows;
  term_f1 = 2TP / (2TP + FP + FN), premature_end_rate = FP / real rows, premature = share of conversations with
  an END flag on a real row, k1_end_rate = TP / conversations.
Rules: the files of one arm are disjoint folds (a conversation id appearing twice is an error); every conversation
must have turns 1..n and a K+1 row. A paired comparison needs the same conversation set in both arms; the bootstrap
resamples conversations (10000 draws, seed 0).

  task1_pooled.py --arm base=f0.jsonl,f1.jsonl,f2.jsonl --arm rl=g0.jsonl,g1.jsonl,g2.jsonl [--compare base rl]
                  [--json-out out.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random


def sha_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_arm(files):
    """-> {conversation_id: {"flags": [END on rows 1..n], "k1": bool}}"""
    convs = {}
    for fn in files:
        rows, k1 = {}, {}
        for l in open(fn, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                rows.setdefault(r["conversation_id"], {})[int(r["turn_index"])] = bool(r["greedy_ended"])
        kp = fn + ".k1.jsonl"
        if not os.path.exists(kp):
            raise SystemExit("%s: K+1 file %s missing" % (fn, kp))
        for l in open(kp, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                k1[r["conversation_id"]] = bool(r["ended"])
        for cid, turns in rows.items():
            if cid in convs:
                raise SystemExit("conversation %s appears in two files of one arm (folds must be disjoint)" % cid)
            n = len(turns)
            if sorted(turns) != list(range(1, n + 1)):
                raise SystemExit("%s: conversation %s has turns %s, not 1..n" % (fn, cid, sorted(turns)))
            if cid not in k1:
                raise SystemExit("%s: conversation %s has no K+1 row" % (fn, cid))
            convs[cid] = {"flags": [turns[t] for t in range(1, n + 1)], "k1": k1[cid]}
        extra = set(k1) - set(rows)
        if extra:
            raise SystemExit("%s: K+1 rows without generations: %s" % (fn, sorted(extra)[:3]))
    return convs


def metrics(convs, ids=None):
    ids = list(convs) if ids is None else ids
    tp = fp = fn = rows = prem = 0
    for cid in ids:
        c = convs[cid]
        tp += c["k1"]
        fn += not c["k1"]
        fp += sum(c["flags"])
        prem += any(c["flags"])
        rows += len(c["flags"])
    k = len(ids)
    den = 2 * tp + fp + fn
    return {"n_conversations": k, "n_turns": rows, "term_f1": (2 * tp / den) if den else 0.0,
            "premature_end_rate": fp / rows if rows else 0.0, "premature": prem / k if k else 0.0,
            "k1_end_rate": tp / k if k else 0.0}


KEYS = ("term_f1", "premature_end_rate", "premature", "k1_end_rate")
HIGHER_BETTER = {"term_f1": True, "k1_end_rate": True, "premature_end_rate": False, "premature": False}


def paired(a, b, n_boot=10000, seed=0):
    if set(a) != set(b):
        raise SystemExit("paired comparison needs the same conversations (only in one arm: %s)"
                         % sorted(set(a) ^ set(b))[:3])
    ids = sorted(a)
    ma, mb = metrics(a, ids), metrics(b, ids)
    rng = random.Random(seed)
    diffs = {k: [] for k in KEYS}
    for _ in range(n_boot):
        pick = [ids[rng.randrange(len(ids))] for _ in ids]
        xa, xb = metrics(a, pick), metrics(b, pick)
        for k in KEYS:
            diffs[k].append(xb[k] - xa[k])
    out = {}
    for k in KEYS:
        d = sorted(diffs[k])
        better = sum(1 for x in d if (x > 0 if HIGHER_BETTER[k] else x < 0)) / len(d)
        out[k] = {"a": ma[k], "b": mb[k], "diff": mb[k] - ma[k], "ci95": [d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]],
                  "p_b_better": better}
    return {"n_conversations": len(ids), "n_boot": n_boot, "seed": seed, "metrics": out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="NAME=FILE[,FILE...] (one file per fold)")
    ap.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    arms, files = {}, {}
    for spec in a.arm:
        name, _, fl = spec.partition("=")
        if not name or not fl:
            ap.error("--arm needs NAME=FILE[,FILE...]")
        fs = [f for f in fl.split(",") if f]
        arms[name], files[name] = load_arm(fs), {f: sha_file(f) for f in fs}
    res = {"arms": {n: metrics(c) for n, c in arms.items()}, "files_sha256": files, "comparisons": []}
    for n, m in res["arms"].items():
        print("%-12s conversations %3d turns %4d  term_f1 %.3f  premature_end_rate %.3f  premature %.3f  k1_end_rate %.3f"
              % (n, m["n_conversations"], m["n_turns"], m["term_f1"], m["premature_end_rate"], m["premature"], m["k1_end_rate"]))
    for x, y in a.compare:
        c = paired(arms[x], arms[y], a.n_boot)
        res["comparisons"].append({"a": x, "b": y, **c})
        print("%s vs %s (%d paired conversations)" % (y, x, c["n_conversations"]))
        for k, v in c["metrics"].items():
            print("   %-19s %.3f -> %.3f  diff %+.3f  95%% CI [%+.3f, %+.3f]  P(%s better) %.3f"
                  % (k, v["a"], v["b"], v["diff"], v["ci95"][0], v["ci95"][1], y, v["p_b_better"]))
    if a.json_out:
        json.dump(res, open(a.json_out, "w", encoding="utf-8"), indent=1)
    return res


if __name__ == "__main__":
    main()
