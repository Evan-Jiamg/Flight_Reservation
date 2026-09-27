#!/usr/bin/env python3
"""Step 0 (user 2026-09-28): is the real person's stopping point predictable from how far their goal is met?

Diagnostic only (no training, no policy): for every TRAIN conversation of the fold (splits[fold]["train"], the
conversations with requirement shards; validation / test ids are refused), the real conversation is replayed turn by
turn through the SAME requirement ledger as Task 2 (r0_client.Ledger with Task2Env's floor judge on our gpt-oss
server, same judge cache): after the real assistant reply to user message t, coverage c_t (share of the person's
requirements addressed so far), the gain d_t = c_t - c_{t-1} and whether all are met.

Decision points follow Task 1: at turn t >= 2 the Planner writes message t and decides whether it is the last one,
having seen the assistant reply to message t-1. Features at a decision point t (available to it): c_{t-1}, d_{t-1},
complete_{t-1}, t; also the oracle c_t / d_t (the reply to message t itself, NOT available at decision time; shown only
to see whether the reply that follows the last message differs). For each feature: AUC of separating the real last
message (t = n) from earlier ones (both directions reported), and P(last | bucket) tables.

  step0_coverage.py --fold 2 --splits splits_v1.json --out step0_f2.jsonl [--workers 4]
Environment: JUDGE_BASE_URL (our gpt-oss server, port 8029), JUDGE_MODEL=gpt-oss-120b, JUDGE_REASONING_EFFORT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import task2_env as TE  # noqa: E402


def auc(pos, neg):
    if not pos or not neg:
        return None
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else 0.5 if a == b else 0.0
    return s / (len(pos) * len(neg))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--splits", default="/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json")
    ap.add_argument("--corpus", default="/home/mzjiang/v5-latency/data.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    for k in ("JUDGE_BASE_URL", "JUDGE_MODEL"):
        if not os.environ.get(k):
            raise SystemExit("%s must be set (our gpt-oss server)" % k)
    if os.environ["JUDGE_MODEL"] != "gpt-oss-120b" or (":%s" % TE.PEND_PORT) not in os.environ["JUDGE_BASE_URL"]:
        raise SystemExit("the ledger judge must be our gpt-oss-120b on port %s (as in Task 2)" % TE.PEND_PORT)
    sp = json.load(open(a.splits, encoding="utf-8"))
    f = {int(x["fold"]): x for x in sp["folds"]}[a.fold]
    ids = sorted(f["train"])
    forb = set(f["forbidden_for_training"])
    assert not set(ids) & forb, "a train id is forbidden"
    TE.setup_environment("pend")
    from r0_client import Ledger
    from metrics.judge import Judge
    from sepsim import pipeline
    effort = os.environ.get("JUDGE_REASONING_EFFORT", "minimal")
    jkey = hashlib.sha256(("%s|%s" % (os.environ["JUDGE_MODEL"], os.environ["JUDGE_BASE_URL"])).encode()).hexdigest()[:12]
    judge = TE.make_floor_judge(Judge)(reasoning_effort=effort, verbose=False,
                                       cache_dir=os.path.join(TE.WORK, "judge_cache_%s" % jkey))
    recs = {}
    for l in open(a.corpus, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r["conversation_id"] in ids:
                recs[r["conversation_id"]] = r
    reqs = json.load(open(os.path.join(TE.BENCH, "data/req_shards_v1.json"), encoding="utf-8"))
    missing = [c for c in ids if c not in recs or c not in reqs]
    if missing:
        raise SystemExit("train conversations without corpus record / requirement shards: %s" % missing[:3])

    def one(cid):
        users, agents = pipeline.split_messages(recs[cid])
        n = len(users)
        led = Ledger(reqs[cid]["req"], judge=judge)
        rows, prev = [], 0.0
        for t in range(1, n + 1):
            reply = agents[t - 1]["text"] if t - 1 < len(agents) else ""
            led.update(t, users[t - 1]["text"], reply)
            c = float(led.coverage())
            rows.append({"conversation_id": cid, "t": t, "n": n, "real_final": t == n, "has_reply": t - 1 < len(agents),
                         "n_req": len(reqs[cid]["req"]), "cov": c, "gain": c - prev, "complete": bool(led.complete())})
            prev = c
        return rows

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        allrows = [r for rows in ex.map(one, ids) for r in rows]
    with open(a.out, "w", encoding="utf-8") as fo:
        for r in allrows:
            fo.write(json.dumps(r) + "\n")
    by = {}
    for r in allrows:
        by.setdefault(r["conversation_id"], {})[r["t"]] = r
    pts = []                                   # decision points t >= 2
    for cid, rs in by.items():
        for t in range(2, len(rs) + 1):
            p_, c_ = rs[t - 1], rs[t]
            pts.append({"final": c_["real_final"], "t": t, "cov_prev": p_["cov"], "gain_prev": p_["gain"],
                        "complete_prev": float(p_["complete"]), "no_gain_prev": float(p_["gain"] <= 1e-9),
                        "cov_cur": c_["cov"], "gain_cur": c_["gain"]})
    res = {"fold": a.fold, "n_conversations": len(by), "n_decision_points": len(pts),
           "n_final": sum(p["final"] for p in pts), "judge_cache": judge.cache_dir if hasattr(judge, "cache_dir") else None,
           "judge_errors": getattr(judge, "n_errors", None), "seconds": round(time.time() - t0, 1), "auc": {}}
    for k in ("t", "cov_prev", "gain_prev", "complete_prev", "no_gain_prev", "cov_cur", "gain_cur"):
        pos = [p[k] for p in pts if p["final"]]
        neg = [p[k] for p in pts if not p["final"]]
        x = auc(pos, neg)
        res["auc"][k] = {"auc_higher_means_last": x, "auc_lower_means_last": None if x is None else 1 - x,
                         "mean_last": sum(pos) / len(pos) if pos else None, "mean_earlier": sum(neg) / len(neg) if neg else None}
    buckets = {}
    for p in pts:
        b = min(4, int(p["cov_prev"] * 5))
        buckets.setdefault(b, [0, 0])
        buckets[b][0] += p["final"]
        buckets[b][1] += 1
    res["p_last_by_cov_prev"] = {"%.1f-%.1f" % (b / 5, (b + 1) / 5): {"last": v[0], "n": v[1]} for b, v in sorted(buckets.items())}
    per = [{"conversation_id": cid[:10], "n": len(rs), "cov": [round(rs[t]["cov"], 2) for t in sorted(rs)]} for cid, rs in sorted(by.items())]
    res["trajectories"] = per
    json.dump(res, open(a.out + ".summary.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "trajectories"}, indent=1))
    for p_ in per:
        print("  %s n=%d cov %s" % (p_["conversation_id"], p_["n"], p_["cov"]))
    return res


if __name__ == "__main__":
    main()
