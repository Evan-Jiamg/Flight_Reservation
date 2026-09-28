"""Checkpoint re-selection (--reselect-seeds): same validation procedure with more seeds, no side effect on the
training files, verifier checks (dry-run loop, no GPU)."""
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_planner_rl as T  # noqa: E402
import verify_pipeline as V  # noqa: E402
from test_rl_advantages import args, make_splits  # noqa: E402

SEEDS = ["--reselect-seeds", "0", "1", "2", "3", "4", "5", "6", "7"]


def tree_sha(d):
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(d)):
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, d).encode() + open(p, "rb").read())
    return h.hexdigest()


def _setup():
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    T.main(args(sp, out, controller="llm", updates=4))
    return sp, out


def test_reselect_no_side_effects_and_consistent():
    sp, out = _setup()
    before = {f: open(os.path.join(out, f), "rb").read() for f in ("validation.jsonl", "best.json", "updates.jsonl")}
    ck = tree_sha(os.path.join(out, "ckpt"))
    rec = T.main(args(sp, out, *SEEDS, controller="llm", updates=4))
    for f, b in before.items():
        assert open(os.path.join(out, f), "rb").read() == b, f
    assert tree_sha(os.path.join(out, "ckpt")) == ck
    rows = T.read_jsonl(os.path.join(out, "reselect.jsonl"))
    summ = {r["update"]: r for r in rows if r["kind"] == "summary"}
    assert sorted(summ) == [0, 2, 4] and all(s["reselect"] and s["val_seeds"] == list(range(8)) for s in summ.values())
    eps = [r for r in rows if r["kind"] == "episode"]
    assert {r["seed"] for r in eps} == set(range(8))
    # each candidate evaluated with ITS own policy (the checkpoint's recorded sha)
    for u, s in summ.items():
        st = json.load(open(os.path.join(out, "ckpt", "u%05d" % u, "state.json")))
        assert s["policy_sha"] == st["policy_sha"]
    ok = {u: s["selection_score"] for u, s in summ.items() if s["selection_score"] is not None}
    assert rec["best"]["update"] == max(ok, key=lambda u: (ok[u], -u))
    rep = V.Report()
    V.check_reselect_ids(out, set(json.load(open(sp))["folds"][0]["validation"]),
                         set(json.load(open(sp))["folds"][0]["train_all"]), rep)
    assert all(c["fail"] == 0 for c in rep.checks.values()), {k: v for k, v in rep.checks.items() if v["fail"]}
    # re-running is idempotent (summaries reused, nothing appended)
    n = len(rows)
    T.main(args(sp, out, *SEEDS, controller="llm", updates=4))
    assert len(T.read_jsonl(os.path.join(out, "reselect.jsonl"))) == n


def test_verify_catches_wrong_seed_and_best():
    sp, out = _setup()
    T.main(args(sp, out, *SEEDS, controller="llm", updates=4))
    val = set(json.load(open(sp))["folds"][0]["validation"])
    tra = set(json.load(open(sp))["folds"][0]["train_all"])
    p = os.path.join(out, "reselect.jsonl")
    rows = T.read_jsonl(p)
    for r in rows:
        if r["kind"] == "episode":
            r["seed"] = 99
            break
    with open(p, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    b = json.load(open(os.path.join(out, "reselect_best.json")))
    b["best"]["update"] = 99
    json.dump(b, open(os.path.join(out, "reselect_best.json"), "w"))
    rep = V.Report()
    V.check_reselect_ids(out, val, tra, rep)
    assert rep.checks["rl.reselect_seeds"]["fail"] == 1 and rep.checks["rl.reselect_best"]["fail"] == 1


def test_reselect_only_chosen_updates():
    import pytest
    sp, out = _setup()
    with pytest.raises(SystemExit):
        T.main(args(sp, out, *SEEDS, "--reselect-updates", "3", controller="llm", updates=4))   # never validated
    rec = T.main(args(sp, out, *SEEDS, "--reselect-updates", "0", "4", controller="llm", updates=4))
    rows = T.read_jsonl(os.path.join(out, "reselect.jsonl"))
    assert sorted(r["update"] for r in rows if r["kind"] == "summary") == [0, 4]
    assert sorted(int(u) for u in rec["candidates"]) == [0, 4]
    meta = T.read_jsonl(os.path.join(out, "reselect_meta.jsonl"))
    assert meta[-1]["candidates"] == [0, 4]
    rep = V.Report()
    sp_ = json.load(open(sp))["folds"][0]
    V.check_reselect_ids(out, set(sp_["validation"]), set(sp_["train_all"]), rep)
    assert all(c["fail"] == 0 for c in rep.checks.values())
