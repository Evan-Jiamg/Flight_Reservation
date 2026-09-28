#!/usr/bin/env python3
"""Cross-fitting groups for the goal judge (audit defect D7).

For each outer fold f: S_f = (inner_train_ids U inner_validation_ids) of f that have requirement
shards. S_f is split into 3 groups by sorting on sha256(conversation_id) and dealing round-robin
(content-blind, outcome-blind, reproducible). A judge for (f, g) is trained ONLY on S_f minus group g
and evaluated ONLY on group g, so every scenario of S_f gets an out-of-fold evaluation and no fold's
outer test is ever used. Asserts: groups partition S_f; S_f disjoint from f's outer test; every
scenario was part of the Stage C no-gate union run (so logged trajectories exist).
"""
import argparse
import hashlib
import json

N_GROUPS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nested-manifest", required=True)
    ap.add_argument("--outer-folds", required=True, help="folds3_goal_persona_v1.json")
    ap.add_argument("--shards", required=True)
    ap.add_argument("--union", required=True, help="stageC_v1/scenarios_union.json")
    ap.add_argument("--corpus", default="/home/mzjiang/v5-latency/data.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    corpus_ids = {json.loads(l)["conversation_id"] for l in open(a.corpus) if l.strip()}
    nested = json.load(open(a.nested_manifest))["folds"]
    shards = json.load(open(a.shards))
    union = {s["conversation_id"] for s in json.load(open(a.union))["scenarios"]}
    outer = json.load(open(a.outer_folds))
    out = {"rule": "S_f = inner train U inner validation of outer fold f with requirement shards; "
                   "groups by sha256(conversation_id) round-robin; judge(f,g) trains on S_f minus g, "
                   "evaluates on g", "n_groups": N_GROUPS,
           "sources_sha256": {k: hashlib.sha256(open(p, "rb").read()).hexdigest()
                              for k, p in (("nested", a.nested_manifest), ("outer", a.outer_folds),
                                           ("shards", a.shards), ("union", a.union))},
           "folds": []}
    for f in nested:
        fold = f["fold"]
        S = sorted((set(f["inner_train_ids"]) | set(f["inner_validation_ids"])) & set(shards))
        test_ids, train_ids, dropped_ids = outer_fold(outer, fold)
        assert test_ids <= corpus_ids and train_ids <= corpus_ids, "outer ids are not conversation_ids"
        assert not set(S) & test_ids, "fold %d: inner scenario in outer test" % fold
        assert not set(S) & dropped_ids, "fold %d: inner scenario among dropped sessions" % fold
        assert set(S) <= train_ids, "fold %d: inner scenario outside outer train" % fold
        missing = [c for c in S if c not in union]
        assert not missing, "fold %d: no logged trajectory for %s" % (fold, missing[:3])
        order = sorted(S, key=lambda c: hashlib.sha256(c.encode()).hexdigest())
        groups = [order[g::N_GROUPS] for g in range(N_GROUPS)]
        assert sorted(sum(groups, [])) == S
        out["folds"].append({"fold": fold, "scenarios": S, "groups": groups,
                             "group_sizes": [len(g) for g in groups],
                             "inner_validation_in_S": sorted(set(f["inner_validation_ids"]) & set(S))})
        print("fold", fold, "scenarios", len(S), "groups", [len(g) for g in groups])
    json.dump(out, open(a.out, "w"), indent=1)


def outer_fold(outer, fold):
    """folds3_goal_persona_v1.json: per fold, `session_ids` = outer TEST sessions,
    `train_session_ids` = outer train, `dropped_session_ids` = goal/persona-mixed (unused)."""
    for f in outer["folds"]:
        if int(f["fold"]) == fold:
            return set(f["session_ids"]), set(f["train_session_ids"]), set(f["dropped_session_ids"])
    raise SystemExit("fold %d not in outer manifest" % fold)


if __name__ == "__main__":
    main()
