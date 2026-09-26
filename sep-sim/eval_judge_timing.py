#!/usr/bin/env python3
"""Does the trained judge's P(SATISFIED) locate the human's stop? (TREC human sessions)

For every human session in the chosen split, the judge scores status_probs on the state after
each assistant reply t = 1..K (K = complete exchanges; input = scenario_text + conversation up to
reply t, the same builder as training/inference). Then, with the score P(SATISFIED) (and, as a
second score, the reward form P(SATISFIED) + 0.5 P(PARTIAL)):

  overall AUC       states after the human's LAST exchange (t = K) vs earlier states (t < K)
  within-turn AUC   the same ranking restricted to pairs with the SAME turn index, pooled with
                    pair weights (as shortcut_audit.py): content signal that cannot come from position
  turn-only AUC     baseline that ranks by the turn index alone
plus 'prefinal' mode (positive t = K-1, t = K dropped) and session-bootstrap 95% CIs.

Leakage: split ids come from splits_v1.json (--split validation|test, --fold). Test needs the
explicit --final flag. The judge's train_manifest.json must be for the same fold and split file
and must not contain any evaluated conversation. Default uses the '<split>_all' lists (human
sessions, shards not needed); --shards-only uses the shard-filtered lists.

This module also holds the shared scoring core used by eval_multiwoz_judge.py.
"""
import argparse
import json
import os
import sys

import judge_metrics as JM

SCORES = {"p_satisfied": lambda p: p["SATISFIED"],
          "p_sat_plus_half_partial": lambda p: p["SATISFIED"] + 0.5 * p["PARTIAL"]}
MODES = ("last", "prefinal")


def session_inputs(goal_text, users, assistants):
    """-> {"goal", "K", "states": [(t, hist_u, hist_a)]}: one state per complete exchange."""
    K = min(len(users), len(assistants))
    states = [(t, list(users[:t]), list(assistants[:t])) for t in range(1, K + 1)]
    return {"goal": goal_text, "K": K, "states": states,
            "n_user_messages": len(users), "unanswered_final_user_message": len(users) > len(assistants)}


def resolve_split_ids(splits, fold, split, final, shards_only):
    if split not in ("validation", "test"):
        raise SystemExit("--split must be validation or test")
    if split == "test" and not final:
        raise SystemExit("refusing to touch the test split without --final")
    fs = JM.fold_of(splits, fold)
    return list(fs[split if shards_only else split + "_all"])


def score_sessions(judge, sessions, cache_path):
    """Resumable: rows keyed by (session, t) in cache_path. -> {sid: {"K", "probs": {t: probs}}}."""
    done = {}
    if os.path.exists(cache_path):
        for l in open(cache_path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[(r["session"], r["t"])] = r
    with open(cache_path, "a", encoding="utf-8") as f:
        for sid in sorted(sessions):
            s = sessions[sid]
            for t, hu, ha in s["states"]:
                if (sid, t) in done:
                    continue
                text, info = judge.prompt_text(s["goal"], hu, ha)
                probs, mass = judge._score_prompt(text)
                r = {"session": sid, "t": t, "K": s["K"], "probs": probs, "mass": mass, **info}
                f.write(json.dumps(r) + "\n")
                f.flush()
                done[(sid, t)] = r
    out = {}
    for sid, s in sessions.items():
        out[sid] = {"K": s["K"], "probs": {t: done[(sid, t)]["probs"] for t, _, _ in s["states"]}}
    return out


def report(scored, n_boot, seed):
    rep = {}
    for sname, fn in SCORES.items():
        for mode in MODES:
            sess = {sid: {"K": v["K"], "scores": {t: fn(p) for t, p in v["probs"].items()}}
                    for sid, v in scored.items() if v["K"] >= 1}
            pts = JM.timing_points(sess, mode)
            rep["%s|%s" % (sname, mode)] = {"point": JM.timing_metrics(pts),
                                            "ci95": JM.session_bootstrap(pts, n_boot, seed)}
    return rep


def load_judge(args):
    import goal_judge as GJ
    return GJ.GoalJudge(args.base, adapter=args.adapter, gpu=args.gpu, dtype=args.dtype).load()


def common_args(ap):
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260925)


def main():
    ap = argparse.ArgumentParser()
    common_args(ap)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--split", required=True, choices=("validation", "test"))
    ap.add_argument("--final", action="store_true", help="required to evaluate the test split")
    ap.add_argument("--shards-only", action="store_true")
    ap.add_argument("--corpus", default="/home/mzjiang/v5-latency/data.jsonl")
    ap.add_argument("--sepsim", default="/home/mzjiang/Sep-Simulator")
    args = ap.parse_args()
    splits = json.load(open(args.splits))
    cids = resolve_split_ids(splits, args.fold, args.split, args.final, args.shards_only)
    split_sha = JM.sha256_file(args.splits)
    manifest = json.load(open(os.path.join(args.adapter, "train_manifest.json")))
    JM.check_manifest(manifest, args.fold, split_sha, cids)
    os.makedirs(args.out, exist_ok=True)

    sys.path.insert(0, args.sepsim)
    from sepsim import pipeline
    recs = {}
    for l in open(args.corpus, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            recs[r["conversation_id"]] = r
    sessions = {}
    for cid in cids:
        users, agents = pipeline.split_messages(recs[cid])
        sessions[cid] = session_inputs(pipeline.scenario_text(recs[cid]), [u["text"] for u in users],
                                       [a["text"] for a in agents])
    judge = load_judge(args)
    scored = score_sessions(judge, sessions, os.path.join(args.out, "scores_fold%d_%s.jsonl" % (args.fold, args.split)))
    rep = {"dataset": "trec_human", "fold": args.fold, "split": args.split,
           "list": args.split if args.shards_only else args.split + "_all", "n_sessions": len(sessions),
           "unanswered_final_user_message": sum(s["unanswered_final_user_message"] for s in sessions.values()),
           "split_file_sha256": split_sha, "adapter": os.path.abspath(args.adapter),
           "judge_manifest_fold": manifest["fold"], "corpus_sha256": JM.sha256_file(args.corpus),
           "n_boot": args.n_boot, "seed": args.seed, "metrics": report(scored, args.n_boot, args.seed)}
    path = os.path.join(args.out, "timing_fold%d_%s.json" % (args.fold, args.split))
    json.dump(rep, open(path, "w"), indent=1)
    for k, v in rep["metrics"].items():
        print(k, json.dumps(v["point"]), "CI95 within", v["ci95"]["within_turn_auc"],
              "overall", v["ci95"]["overall_auc"], flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
