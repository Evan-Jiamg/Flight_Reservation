#!/usr/bin/env python3
"""Gold-prefix stop metrics: position-wise and session first-stop timing.

Input: eval_stop_sft.py predictions (record_id, turn_index, target_stop, p_stop),
one K+1 row per session at its last position. For each threshold:

  position-wise   false_stop = P(p>=thr | continue), k1_end = P(p>=thr | K+1),
                  plus threshold-free AUC and NLL.
  session-wise    first position with p>=thr:
                    early      first stop before K+1 (premature, at a gold continue)
                    exact      first stop exactly at K+1
                    not_by_k1  no stop by K+1 -> RIGHT-CENSORED (we do not guess
                               the later stop turn)
                  plus mean positions-early among early sessions.

Session bootstrap (resample sessions) gives 95% intervals for the session rates
and position rates. Several prediction files can be passed (e.g. pooled folds);
record_ids must not collide.
"""
import argparse
import json
import math
import random
from collections import defaultdict

GRID = [round(0.1 * i, 1) for i in range(1, 10)]


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    sessions = defaultdict(list)
    for r in rows:
        sessions[r["record_id"]].append(r)
    for rid, rs in sessions.items():
        rs.sort(key=lambda r: int(r["turn_index"]))
        ends = [r for r in rs if r["target_stop"]]
        if len(ends) != 1 or ends[0] is not rs[-1]:
            raise ValueError("session %s must have exactly one K+1 row at the end" % rid)
    return sessions


def position_metrics(sessions, ids, thr):
    cont = [r for s in ids for r in sessions[s] if not r["target_stop"]]
    ends = [r for s in ids for r in sessions[s] if r["target_stop"]]
    return {"false_stop": sum(r["p_stop"] >= thr for r in cont) / len(cont) if cont else float("nan"),
            "k1_end": sum(r["p_stop"] >= thr for r in ends) / len(ends)}


def session_metrics(sessions, ids, thr):
    early = exact = censored = 0
    lead = []
    for s in ids:
        rs = sessions[s]
        first = next((i for i, r in enumerate(rs) if r["p_stop"] >= thr), None)
        if first is None:
            censored += 1
        elif first == len(rs) - 1:
            exact += 1
        else:
            early += 1
            lead.append(len(rs) - 1 - first)
    n = len(ids)
    return {"early": early / n, "exact": exact / n, "not_by_k1_censored": censored / n,
            "mean_turns_early": sum(lead) / len(lead) if lead else None}


def hazard_metrics(sessions, ids):
    """Sampled stop with probability p_stop at each position (session-level hazard)."""
    early = exact = censored = 0.0
    for s in ids:
        rs = sessions[s]
        survive = 1.0
        for r in rs[:-1]:
            survive *= 1 - r["p_stop"]
        h = rs[-1]["p_stop"]
        early += 1 - survive
        exact += survive * h
        censored += survive * (1 - h)
    n = len(ids)
    return {"early": early / n, "exact": exact / n, "not_by_k1_censored": censored / n}


def auc_nll(sessions, ids):
    rows = [r for s in ids for r in sessions[s]]
    cont = [r["p_stop"] for r in rows if not r["target_stop"]]
    ends = [r["p_stop"] for r in rows if r["target_stop"]]
    auc = (sum((e > c) + 0.5 * (e == c) for e in ends for c in cont) / (len(ends) * len(cont))
           if cont and ends else float("nan"))
    nll = -sum(math.log(min(1 - 1e-8, max(1e-8, r["p_stop"] if r["target_stop"] else 1 - r["p_stop"])))
               for r in rows) / len(rows)
    return auc, nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="+")
    ap.add_argument("--thresholds", default=",".join(map(str, GRID)))
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    sessions = load(args.predictions)
    ids = sorted(sessions)
    thresholds = [float(x) for x in args.thresholds.split(",")]
    auc, nll = auc_nll(sessions, ids)
    n_cont = sum(1 for s in ids for r in sessions[s] if not r["target_stop"])
    out = {"sessions": len(ids), "n_cont": n_cont, "n_k1": len(ids), "auc": auc, "nll": nll,
           "files": args.predictions, "by_threshold": {}}
    rng = random.Random(20260924)
    boots = [[ids[rng.randrange(len(ids))] for _ in ids] for _ in range(args.bootstrap)]
    for thr in thresholds:
        point = {**position_metrics(sessions, ids, thr), **session_metrics(sessions, ids, thr)}
        ci = {}
        for key in ("false_stop", "k1_end", "early", "exact", "not_by_k1_censored"):
            draws = []
            for b in boots:
                m = {**position_metrics(sessions, b, thr), **session_metrics(sessions, b, thr)}
                if not math.isnan(m[key]):
                    draws.append(m[key])
            draws.sort()
            ci[key] = [draws[int(.025 * len(draws))], draws[int(.975 * len(draws)) - 1]]
        out["by_threshold"][str(thr)] = {"point": point, "session_bootstrap_95": ci}
    hz = hazard_metrics(sessions, ids)
    hz_ci = {}
    for key in hz:
        draws = sorted(hazard_metrics(sessions, b)[key] for b in boots)
        hz_ci[key] = [draws[int(.025 * len(draws))], draws[int(.975 * len(draws)) - 1]]
    out["hazard"] = {"point": hz, "session_bootstrap_95": hz_ci}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
