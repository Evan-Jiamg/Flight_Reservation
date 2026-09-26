#!/usr/bin/env python3
"""Cross-dataset check (evaluation only): does a trained judge's P(SATISFIED) locate the MultiWOZ
human's stop?

Input: subset_v1.jsonl (60 dialogues; never trained on). Goal text = " ".join(goal_message);
states after each assistant reply t = 1..K (K = number of turns with a non-empty assistant reply
counted as complete exchanges). Metrics, CIs and modes as eval_judge_timing.py. For MultiWOZ the
'prefinal' mode is the informative one when the final user message is a closing ("thanks, bye"):
in 'last' mode that closing text is part of the judged input. Both are reported; neither is
chosen after seeing results.

The judge prompt is the TREC one ("research datasets"); applying it unchanged to MultiWOZ is the
point of the generalisation test.
"""
import argparse
import json
import os

import eval_judge_timing as T
import judge_metrics as JM


def multiwoz_sessions(records):
    """-> ({cid: session_inputs}, stats). A turn with an empty assistant reply ends the usable
    exchanges (the state after it would have no reply to judge)."""
    sessions, stats = {}, {"dialogues": 0, "stopped_at_empty_reply": 0, "stopped_at_final_turn": 0}
    for r in records:
        stats["dialogues"] += 1
        users, assts = [], []
        for i, turn in enumerate(r["turns"]):
            u, a = str(turn.get("user", "")), str(turn.get("assistant", "") or "")
            users.append(u)
            if not a.strip():
                stats["stopped_at_empty_reply"] += 1
                stats["stopped_at_final_turn"] += int(i == len(r["turns"]) - 1)
                break
            assts.append(a)
        sessions[r["cid"]] = T.session_inputs(" ".join(r["goal_message"]), users, assts)
    return sessions, stats


def main():
    ap = argparse.ArgumentParser()
    T.common_args(ap)
    ap.add_argument("--data", default="/tmp2/mzjiang_usersim/xdom/data/multiwoz/subset_v1.jsonl")
    ap.add_argument("--tag", default="", help="e.g. fold0 (which judge adapter)")
    args = ap.parse_args()
    records = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    sessions, stats = multiwoz_sessions(records)
    os.makedirs(args.out, exist_ok=True)
    mpath = os.path.join(args.adapter, "train_manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else None
    judge = T.load_judge(args)
    tag = args.tag or "adapter"
    scored = T.score_sessions(judge, sessions, os.path.join(args.out, "scores_multiwoz_%s.jsonl" % tag))
    rep = {"dataset": "multiwoz_subset_v1", "data_sha256": JM.sha256_file(args.data), "stats": stats,
           "n_sessions": len(sessions), "K_dist": dict(sorted(
               {k: sum(s["K"] == k for s in sessions.values()) for k in {s["K"] for s in sessions.values()}}.items())),
           "adapter": os.path.abspath(args.adapter), "judge_manifest_fold": manifest and manifest["fold"],
           "n_boot": args.n_boot, "seed": args.seed, "metrics": T.report(scored, args.n_boot, args.seed)}
    path = os.path.join(args.out, "timing_multiwoz_%s.json" % tag)
    json.dump(rep, open(path, "w"), indent=1)
    for k, v in rep["metrics"].items():
        print(k, json.dumps(v["point"]), "CI95 within", v["ci95"]["within_turn_auc"],
              "overall", v["ci95"]["overall_auc"], flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
