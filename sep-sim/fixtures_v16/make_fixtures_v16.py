"""Archived v16 dry-run runs, used by the tests of verify_pipeline's v16 branch (SPEC v17 B8/B9, 2026-09-30).

The v17 trainer no longer implements v16 (refill, D2, re-selection, best.json, the LLM controller's v16 settings), so
the v16 verify checks are tested on run directories that the v16 trainer wrote. This script produced them ONCE, at
commit 9afebdd (the last commit with the v16 train_planner_rl.py), with the dry-run fakes (no GPU); it cannot be
re-run with the v17 trainer. MANIFEST.json records the code sha of every file that wrote them.

  cd sep-sim && python fixtures_v16/make_fixtures_v16.py        (at commit 9afebdd only)
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEP = os.path.dirname(HERE)
sys.path.insert(0, SEP)
import train_planner_rl as T  # noqa: E402

IDS = ["c%02d" % i for i in range(30)]


def make_splits(d):
    s = {"folds": [{"fold": 0, "train": IDS[:14], "validation": IDS[14:15], "test": IDS[15:24],
                    "train_all": IDS[:14] + ["x_noshard"], "validation_all": IDS[14:15], "test_all": IDS[15:24],
                    "forbidden_for_training": IDS[14:24]}]}
    p = os.path.join(d, "splits.json")
    json.dump(s, open(p, "w"))
    return p


def make_big_splits(d, n_train=24, n_extra=6):
    ids = ["b%03d" % i for i in range(n_train + n_extra + 12)]
    tr, ex = ids[:n_train], ids[n_train:n_train + n_extra]
    val, te = ids[n_train + n_extra:n_train + n_extra + 2], ids[n_train + n_extra + 2:]
    sp = {"folds": [{"fold": 0, "train": tr, "validation": val, "test": te, "train_all": tr + ex,
                     "validation_all": val, "test_all": te, "forbidden_for_training": val + te}]}
    p = os.path.join(d, "splits.json")
    json.dump(sp, open(p, "w"))
    return p


def args(splits, out, *extra, controller="llm", updates=5):
    return ["--dry-run", "--fold", "0", "--splits", splits, "--out", out, "--updates", str(updates),
            "--G", "4", "--scenarios-per-update", "3", "--val-every", "2", "--val-seeds", "0", "1",
            "--controller", controller] + (["--ablation", "test-" + controller] if controller != "llm" else []) + list(extra)


def fresh_dir(name):
    d = os.path.join(HERE, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


def main():
    made = {}
    # main: 4 updates (validation at 0, 2, 4), then the 8-seed re-selection of every validated checkpoint
    d = fresh_dir("main")
    sp = make_splits(d)
    out = os.path.join(d, "run")
    T.main(args(sp, out, updates=4))
    T.main(args(sp, out, "--reselect-seeds", "0", "1", "2", "3", "4", "5", "6", "7", updates=4))
    made["main"] = "fresh(updates=4) + --reselect-seeds 0..7"
    # d2: margin -1 -> the D2 trigger at the second validation after the base (u4), annealing afterwards
    d = fresh_dir("d2")
    sp = make_splits(d)
    T.main(args(sp, os.path.join(d, "run"), "--t1-trigger-margin", "-1", "--ablation", "test-d2", updates=6))
    made["d2"] = "--t1-trigger-margin -1 --ablation test-d2, 6 updates"
    # cap: every Task 1 group degenerate (the fake policy never ends) -> refill up to the cap
    d = fresh_dir("cap")
    sp = make_big_splits(d)
    orig = T.FakeEnv.task1_sample

    def never_end(self, *a, **k):
        o = orig(self, *a, **k)
        for x in o:
            x["ended_planner"] = False
            x["reward"] = float(not x["real_final"])
        return o
    T.FakeEnv.task1_sample = never_end
    try:
        T.main(args(sp, os.path.join(d, "run"), updates=2))
    finally:
        T.FakeEnv.task1_sample = orig
    made["cap"] = "big splits, Task 1 samples never end, 2 updates (refill stop 'cap')"
    # iv: a user-approved intervention before update 3 (w_dist 1.0, bounds [1, 5])
    d = fresh_dir("iv")
    sp = make_splits(d)
    ivp = os.path.join(d, "iv.json")
    json.dump({"set_cfg": {"w_dist": 1.0}, "controller_bounds": {"w_dist": [1.0, 5.0]}, "reason": "test",
               "approved": "test"}, open(ivp, "w"))
    T.main(args(sp, os.path.join(d, "run"), updates=2))
    T.main(args(sp, os.path.join(d, "run"), "--resume", "--intervention", ivp, updates=4))
    made["iv"] = "2 updates, then --resume --intervention (w_dist 1.0, bounds [1, 5]) to 4 updates"
    shas = {f: hashlib.sha256(open(os.path.join(SEP, f), "rb").read()).hexdigest()
            for f in T.CODE_FILES + ("verify_pipeline.py",) if os.path.exists(os.path.join(SEP, f))}
    json.dump({"written_by": "fixtures_v16/make_fixtures_v16.py at commit 9afebdd (v16 trainer, dry run)",
               "runs": made, "code_sha256": shas}, open(os.path.join(HERE, "MANIFEST.json"), "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
