#!/usr/bin/env python3
"""The one split file every stage reads (leakage control). Per outer fold f:

  train       inner_train_ids(f)      with requirement shards  -> SFT / RL rollouts / reward stats / controller
  validation  inner_validation_ids(f) with requirement shards  -> checkpoint selection, gates only
  test        outer test session_ids(f) with requirement shards -> once, after method freeze
plus the same three sets WITHOUT the shard filter (human sessions, e.g. for gold-prefix timing),
and 'forbidden_for_training' = validation U test (both variants).

Asserts: train/validation/test pairwise disjoint per fold; train and validation inside the outer
train side; test is the outer test side; goal and persona of test sessions do not occur in train
(outer manifest property, re-checked here). Records source SHA256s.
"""
import argparse
import hashlib
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nested-manifest", required=True)
    ap.add_argument("--outer-folds", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--corpus", default="/home/mzjiang/v5-latency/data.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nested = {f["fold"]: f for f in json.load(open(a.nested_manifest))["folds"]}
    outer = json.load(open(a.outer_folds))
    shards = set(json.load(open(a.shards)))
    recs = {json.loads(l)["conversation_id"]: json.loads(l) for l in open(a.corpus) if l.strip()}
    goal_of, persona_of = outer.get("goal_of", {}), outer.get("persona_of", {})
    out = {"sources_sha256": {k: hashlib.sha256(open(p, "rb").read()).hexdigest() for k, p in
                              (("nested", a.nested_manifest), ("outer", a.outer_folds), ("shards", a.shards),
                               ("corpus", a.corpus))}, "folds": []}
    for f in outer["folds"]:
        k = int(f["fold"])
        tr_all = sorted(set(nested[k]["inner_train_ids"]))
        va_all = sorted(set(nested[k]["inner_validation_ids"]))
        te_all = sorted(set(f["session_ids"]))
        outer_train = set(f["train_session_ids"])
        for x, y, n in ((tr_all, va_all, "train/val"), (tr_all, te_all, "train/test"), (va_all, te_all, "val/test")):
            assert not set(x) & set(y), "fold %d: %s overlap" % (k, n)
        assert set(tr_all) <= outer_train and set(va_all) <= outer_train
        assert all(c in recs for c in tr_all + va_all + te_all)
        if goal_of and persona_of:
            tg = {goal_of.get(c) for c in te_all}
            tp = {persona_of.get(c) for c in te_all}
            assert not tg & {goal_of.get(c) for c in tr_all}, "fold %d: goal shared train/test" % k
            assert not tp & {persona_of.get(c) for c in tr_all}, "fold %d: persona shared train/test" % k
        d = {"fold": k,
             "train": [c for c in tr_all if c in shards], "validation": [c for c in va_all if c in shards],
             "test": [c for c in te_all if c in shards],
             "train_all": tr_all, "validation_all": va_all, "test_all": te_all}
        d["forbidden_for_training"] = sorted(set(va_all) | set(te_all))
        d["sizes"] = {x: len(d[x]) for x in ("train", "validation", "test", "train_all", "validation_all", "test_all")}
        out["folds"].append(d)
        print("fold", k, d["sizes"])
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
