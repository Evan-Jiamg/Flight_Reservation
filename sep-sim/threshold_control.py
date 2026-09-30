#!/usr/bin/env python3
"""Threshold control (SPEC v17 §6 / B6, user 2026-09-30): Task 1 only, no training, pure CPU.

Question: does merely moving the untrained policy's decision threshold catch up with SFT (u0) / final (GRPO)? If so, that
is reported as it is.
  1. fit ONE scalar logit offset b on the base policy's teacher-forced end probabilities of the train_all decision points
     (RUN/base_pend_train.jsonl, written by the trainer's SFT stage), minimising the NLL of the human labels:
         P' = sigmoid(logit(P_end) + b),   P_end clipped to [1e-6, 1 - 1e-6];
     only the VALID points (a scored probability) enter the fit;
  2. apply b to the base policy's test end probabilities (fold 2: the v16 run's test.jsonl rows of update 0; folds 0 / 1:
     RUN/test_base.jsonl from eval_test_rl.py --include-base) and report nll (valid points), bal_p, auc (task1_stop
     .task1_prob_metrics) and, deciding "end" where P' > 0.5, the M2 stop metrics (task1_stop.task1_stop_metrics:
     term_f1, premature, ...). The same numbers at b = 0 (the base policy read at 0.5) and the logged greedy decisions
     are reported next to them.
  An unscored point (invalid plan -> p_end 0; valid decision without a located value -> its greedy decision) is not a
  probability: it keeps its p_end and its logged greedy decision (the shift applies to scored points only).

Leakage gate (fix round 1, C-N10): with --splits / --fold, every train point must be a train_all conversation (and
not forbidden) and every test conversation a test_all one; anything else is refused.

Usage: threshold_control.py --train RUN/base_pend_train.jsonl --test FILE --splits splits_v1.json --fold F
                            [--test-update 0|base] [--json-out OUT]
  FILE = a test.jsonl / test_base.jsonl of eval_test_rl.py (rows kind "task1"; the latest row per conversation of that
  update with the policy sha of its summary)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import task1_stop as T1  # noqa: E402

EPS = 1e-6


def _clip(p):
    return min(1 - EPS, max(EPS, float(p)))


def logit(p):
    p = _clip(p)
    return math.log(p / (1 - p))


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def nll(points, b=0.0):
    """Mean -log p(label) of the valid points under the offset b."""
    xs = [p for p in points if p.get("valid", True)]
    if not xs:
        return None
    return sum(-math.log(_clip(sigmoid(logit(p["p_end"]) + b)) if p["real_final"]
                         else _clip(1 - sigmoid(logit(p["p_end"]) + b)))
               for p in xs) / len(xs)


def fit_offset(points, lo=-50.0, hi=50.0, iters=200):
    """argmin_b NLL over the valid points: the derivative mean(sigmoid(z + b) - y) is increasing in b, so its root is
    found by bisection (the NLL is convex). A sample with only one label has no finite optimum: then the bound is
    returned and flagged by the caller."""
    xs = [(logit(p["p_end"]), 1.0 if p["real_final"] else 0.0) for p in points if p.get("valid", True)]
    if not xs:
        raise ValueError("no valid point to fit the offset on")

    def grad(b):
        return sum(sigmoid(z + b) - y for z, y in xs) / len(xs)
    if grad(lo) >= 0:
        return lo
    if grad(hi) <= 0:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if grad(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def shifted(points, b):
    out = []
    for p in points:
        q = dict(p)
        if p.get("valid", True):
            q["p_end"] = sigmoid(logit(p["p_end"]) + b)
        out.append(q)
    return out


def stop_rows(task1_rows, b, use_greedy=False):
    """M2 rows for task1_stop_metrics: at every scored decision point the Planner decision becomes P' > 0.5 (or, with
    use_greedy, the logged greedy decision); unscored points and the Speaker blanks keep their logged values."""
    out = []
    for r in task1_rows:
        pe = {p["t"]: p for p in r["end_probs"]}
        turns = []
        for x in r["task1"]["turns"]:
            y = dict(x)
            p = pe.get(x["t"])
            if not use_greedy and p is not None and p.get("valid", True):
                y["ended_planner"] = sigmoid(logit(p["p_end"]) + b) > 0.5
            turns.append(y)
        out.append({"turns": turns, "k1_speaker_blank": r["task1"].get("k1_speaker_blank", False)})
    return out


def load_test(path, update):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    key = update if update == "base" else int(update)
    summ = [r for r in rows if r.get("kind") == "summary" and r.get("update") == key]
    if not summ:
        raise SystemExit("%s has no summary of update %r" % (path, update))
    psha = summ[-1]["policy_sha"]
    t1 = {}
    for r in rows:
        if r.get("kind") == "task1" and r.get("update") == key and r.get("policy_sha") == psha:
            t1[r["conversation_id"]] = r
    if sorted(t1) != sorted(summ[-1].get("task1_ids") or t1):
        raise SystemExit("%s: Task 1 rows %s != the summary's ids" % (path, sorted(t1)))
    return [t1[c] for c in sorted(t1)], psha


def metrics(task1_rows, b):
    pts = shifted([p for r in task1_rows for p in r["end_probs"]], b)
    m = T1.task1_prob_metrics(pts)
    s = T1.task1_stop_metrics(stop_rows(task1_rows, b))
    return {"nll": m["nll"], "bal_p": m["bal_p"], "auc": m["auc"], "n_points": m["n_points"], "n_invalid": m["n_invalid"],
            "term_f1": s["term_f1"], "premature": s["premature"], "premature_end_rate": s["premature_end_rate"],
            "k1_end_rate": s["k1_end_rate"]}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True, help="RUN/base_pend_train.jsonl")
    ap.add_argument("--test", required=True, help="test.jsonl (update 0 = base of a v16 run) or test_base.jsonl")
    ap.add_argument("--test-update", default="0", help="the update of the base policy in --test: 0 (v16 u0) or base")
    ap.add_argument("--splits", required=True, help="the split file of the run (train_all / test_all gate)")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    train = [json.loads(l) for l in open(a.train, encoding="utf-8") if l.strip()]
    for p in train:
        if not (0.0 <= float(p["p_end"]) <= 1.0):
            raise SystemExit("train p_end %r outside [0, 1]" % p["p_end"])
    f = {int(x["fold"]): x for x in json.load(open(a.splits, encoding="utf-8"))["folds"]}[a.fold]
    train_all, test_all, forb = set(f["train_all"]), set(f["test_all"]), set(f["forbidden_for_training"])
    bad = sorted({p.get("conversation_id") for p in train} - train_all) + \
        sorted({p.get("conversation_id") for p in train} & forb)
    if bad:
        raise SystemExit("train points outside train_all / in forbidden: %s" % bad[:5])
    b = fit_offset(train)
    at_bound = abs(b) >= 50.0 - 1e-9
    rows, psha = load_test(a.test, a.test_update)
    bad = sorted({r["conversation_id"] for r in rows} - test_all)
    if bad:
        raise SystemExit("test conversations outside test_all: %s" % bad[:5])
    res = {"offset_b": b, "offset_at_bound": at_bound,
           "train": {"n_points": len(train), "n_valid": sum(1 for p in train if p.get("valid", True)),
                     "nll_b0": nll(train, 0.0), "nll_b": nll(train, b)},
           "test_update": a.test_update, "test_policy_sha": psha, "test_conversations": [r["conversation_id"] for r in rows],
           "base_logged_greedy": T1.task1_stop_metrics(stop_rows(rows, 0.0, use_greedy=True)),
           "base_at_0.5": metrics(rows, 0.0), "threshold_control": metrics(rows, b),
           "inputs_sha256": {a.train: sha(a.train), a.test: sha(a.test), a.splits: sha(a.splits)}, "fold": a.fold}
    print("offset b = %.4f%s (train nll %.4f -> %.4f on %d valid points)" % (
        b, " (AT THE BOUND: one label only?)" if at_bound else "", res["train"]["nll_b0"], res["train"]["nll_b"],
        res["train"]["n_valid"]))
    for name in ("base_at_0.5", "threshold_control"):
        m = res[name]
        print("%-18s nll %s bal_p %.4f auc %s term_f1 %.4f premature %.4f" % (
            name, None if m["nll"] is None else round(m["nll"], 4), m["bal_p"],
            None if m["auc"] is None else round(m["auc"], 4), m["term_f1"], m["premature"]))
    g = res["base_logged_greedy"]
    print("%-18s term_f1 %.4f premature %.4f (the logged greedy decisions)" % ("base_greedy", g["term_f1"], g["premature"]))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, sort_keys=True)
    return res


if __name__ == "__main__":
    main()
