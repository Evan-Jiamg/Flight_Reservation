#!/usr/bin/env python3
"""ID-only inner-VALIDATION Task 2 scenario lists (selection only; never shown to a reward editor).

Mirror of make_task2_train_scenarios.py for the inner_validation side:
conversation IDs in fold inner_validation_ids that also have requirement shards.
Writes to a separate directory so train-side lists stay untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nested-manifest", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    nested = json.loads(Path(args.nested_manifest).read_text(encoding="utf-8"))["folds"]
    corpus = [json.loads(x) for x in Path(args.corpus).read_text(encoding="utf-8").splitlines() if x]
    shards = json.loads(Path(args.shards).read_text(encoding="utf-8"))
    audit = {"purpose": "inner-validation Task 2 selection only; not for optimizer or reward editor",
             "source_sha256": {"nested": sha(args.nested_manifest), "corpus": sha(args.corpus),
                               "shards": sha(args.shards)}, "folds": []}
    for fold in nested:
        n = fold["fold"]
        allowed = set(fold["inner_validation_ids"])
        selected = [r for r in corpus if r["conversation_id"] in allowed
                    and r["conversation_id"] in shards]
        assert len({r["conversation_id"] for r in selected}) == len(selected)
        assert not ({r["conversation_id"] for r in selected} &
                    set(fold["inner_train_ids"] + fold["inner_mixed_ids"]))
        scenarios = [{"order": i, "conversation_id": r["conversation_id"],
                      "record_id": r["record_id"],
                      "n_req": len(shards[r["conversation_id"]]["req"])}
                     for i, r in enumerate(selected)]
        target = out / f"fold{n}_inner_validation_scenarios.json"
        target.write_text(json.dumps({"t_max": 10, "seeds": [0, 1], "scenarios": scenarios},
                                     indent=2) + "\n", encoding="utf-8")
        audit["folds"].append({"fold": n, "scenarios": len(scenarios),
                               "validation_sessions": len(allowed),
                               "episodes_per_arm": 2 * len(scenarios), "sha256": sha(target)})
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
