"""Dry-run fixtures of SPEC v18 (tests and the dry-run CLI): a fake init adapter (the v17 u0 stand-in), fake act labels
for train_all / validation_all, a fake reranker json. No GPU, no model."""
from __future__ import annotations

import json
import os
import random

import train_planner_rl as T

MOVES7 = ("Disclose", "Reveal", "Inquire", "Navigate", "Note", "Complete", "Other")


def make_init_adapter(d, theta=(-0.8, -1.2, -0.4, 0.3, -1.0)):
    """A fake checkpoint dir like ckpt/u00000 of a v17 dry run: fake_learner.json, the stub adapter, state.json with the
    policy sha. -> (dir, policy sha)."""
    os.makedirs(d, exist_ok=True)
    fl = T.FakeLearner("grpo", {}, 1e-5)
    fl.theta = list(theta)
    fl.save(d)
    sha = fl.policy_sha()
    T.write_json_atomic(os.path.join(d, "state.json"), {"update": 0, "policy_sha": sha, "spec_version": "v17"})
    return d, sha


def make_labels(splits_path, out, split="train_all", fold=0, seed=0, insufficient=()):
    """Fake label_acts.py rows for every message of splits[fold][split] (the fake env's message counts and word counts):
    3 votes over the 7 moves; (cid, t) in `insufficient` get one valid vote. Writes <out> and <out>.meta.json."""
    f = {int(x["fold"]): x for x in json.load(open(splits_path))["folds"]}[fold]
    env = T.FakeEnv(None)
    rows = []
    for cid in sorted(f[split]):
        hw = env.human_words(cid)
        for t in range(1, env.human_turns(cid) + 1):
            rng = random.Random(T.seed_of("fake-label", seed, cid, t))
            nv = 1 if (cid, t) in insufficient else 3
            votes = [rng.choice(MOVES7[:5]) if rng.random() < 0.85 else rng.choice(MOVES7[5:]) for _ in range(nv)]
            q7 = {m: votes.count(m) / nv for m in sorted(set(votes))}
            rows.append({"conversation_id": cid, "t": t, "q7": q7, "n_valid_votes": nv, "n_words": hw[t - 1],
                         "votes": [{"move": m} for m in votes]})
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    import hashlib
    json.dump({"fleiss_kappa_coarse": 0.62, "complete_agreement": 0.5, "n_votes_missing": 0, "n_retries": 0,
               "split": split, "fold": fold, "labels_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()},
              open(out + ".meta.json", "w"))
    return out


def make_reranker(out, passed=True):
    """A fake reranker json with the runtime fields (tiny PCA) and a recorded gate result."""
    d = {"version": "fake", "features": {"style": ["lowercase_start", "ends_with_punct", "n_question", "n_exclaim",
                                                   "upper_ratio"], "pca_dim": 2},
         "pca": {"mean": [0.0, 0.0, 0.0], "components": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]},
         "scaler": {"mean": [0.0] * 7, "std": [1.0] * 7}, "w": [0.5, -0.2, 0.3, -0.1, 0.0, 0.0, -0.4], "C": 0.1,
         "gate": {"passed": bool(passed), "loco_acc": 0.7 if passed else 0.5, "val_acc": 0.6}}
    json.dump(d, open(out, "w"))
    return out


def v18_args(d, splits, out, *extra, updates=3, margin=100.0, passed=True, init=None, labels=None, labels_val=None,
             reranker=None):
    """Dry-run v18 arguments (fake init adapter, labels, reranker created under d unless given)."""
    if init is None:
        init, sha = make_init_adapter(os.path.join(d, "init_u0"))
    else:
        sha = json.load(open(os.path.join(init, "state.json")))["policy_sha"]
    labels = labels or make_labels(splits, os.path.join(d, "labels_train.jsonl"))
    labels_val = labels_val or make_labels(splits, os.path.join(d, "labels_val.jsonl"), split="validation_all")
    reranker = reranker or make_reranker(os.path.join(d, "reranker.json"), passed=passed)
    return ["--dry-run", "--spec", "v18", "--fold", "0", "--splits", splits, "--out", out, "--updates", str(updates),
            "--G", "4", "--scenarios-per-update", "3", "--length-drift-margin", str(margin), "--ablation", "test",
            "--init-adapter", init, "--init-policy-sha", sha, "--act-labels", labels, "--act-labels-val", labels_val,
            "--reranker", reranker] + list(extra)
