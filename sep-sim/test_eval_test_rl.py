"""Test-split evaluation (eval_test_rl.py) and its check / bootstrap (eval_test_boot.py): gates, no side effect on any run
file, ids = splits test / test_all, test once, resume after a crash, unclean episodes left out of the paired
comparison, the checker recomputes and catches tampering (dry-run loop, no GPU)."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_test_rl as E  # noqa: E402
import train_planner_rl as T  # noqa: E402
from test_rl_advantages import args, make_splits  # noqa: E402
from test_reselect import SEEDS  # noqa: E402

TEST = ["--final", "--test-seeds", "0", "1", "2"]


def _setup():
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    T.main(args(sp, out, controller="llm", updates=4))
    rec = T.main(args(sp, out, *SEEDS, controller="llm", updates=4))
    bu = rec["best"]["update"]
    if bu == 0:
        pytest.skip("the dry-run re-selection chose u0")
    return sp, out, sorted({0, bu})


def run_files_sha(out):
    """sha of every file of the run except the test outputs."""
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(out)):
        for f in sorted(files):
            if f.startswith("test"):
                continue
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, out).encode() + open(p, "rb").read())
    return h.hexdigest()


def ev(sp, out, ups):
    return E.main(TEST + ["--test-updates"] + [str(u) for u in ups] + args(sp, out, controller="llm", updates=4))


def boot(out):
    return subprocess.run([sys.executable, os.path.join(HERE, "eval_test_boot.py"), out], capture_output=True, text=True, cwd=HERE)


def test_eval_and_check():
    sp, out, ups = _setup()
    before = run_files_sha(out)
    ev(sp, out, ups)
    assert run_files_sha(out) == before                   # no run file (training, re-selection, checkpoints) touched
    rows = T.read_jsonl(os.path.join(out, "test.jsonl"))
    f0 = json.load(open(sp))["folds"][0]
    summ = {r["update"]: r for r in rows if r["kind"] == "summary"}
    assert sorted(summ) == ups
    assert {r["conversation_id"] for r in rows if r["kind"] == "episode"} == set(f0["test"])
    assert {r["conversation_id"] for r in rows if r["kind"] == "task1"} == set(f0["test_all"])
    assert all(r["split"] == "test" for r in rows)
    assert {r["seed"] for r in rows if r["kind"] == "episode"} == {0, 1, 2}
    for u, s in summ.items():
        assert s["policy_sha"] == json.load(open(os.path.join(out, "ckpt", "u%05d" % u, "state.json")))["policy_sha"]
        assert s["turn_stats_seeds01"]["n_episodes"] == 2 * len(f0["test"])
    meta = T.read_jsonl(os.path.join(out, "test_meta.jsonl"))[-1]
    assert set(meta["eval_code_sha256"]) == {"eval_test_rl.py", "eval_test_boot.py"}
    r = boot(out)
    assert r.returncode == 0 and "TEST CHECK PASSED" in r.stdout and "seeds 0/1" in r.stdout \
        and "Task 1:" in r.stdout and "WARNING" not in r.stdout, r.stdout + r.stderr
    # test once: a second run evaluates nothing again
    n = len(rows)
    ev(sp, out, ups)
    assert len(T.read_jsonl(os.path.join(out, "test.jsonl"))) == n
    # tampering is caught
    p = os.path.join(out, "test.jsonl")
    for r_ in rows:
        if r_["kind"] == "summary":
            r_["task1"]["bal_p"] += 0.1
            break
    open(p, "w").write("".join(json.dumps(x) + "\n" for x in rows))
    r = boot(out)
    assert r.returncode == 1 and "bal_p" in r.stdout


def test_resume_after_crash(monkeypatch):
    sp, out, ups = _setup()
    orig, calls = T.Trainer.task1_eval_row, {"n": 0}

    def crashing(self, u, psha, cid):
        if u == ups[1]:
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("simulated crash in the second update")
        return orig(self, u, psha, cid)

    monkeypatch.setattr(T.Trainer, "task1_eval_row", crashing)
    with pytest.raises(RuntimeError):
        ev(sp, out, ups)
    monkeypatch.setattr(T.Trainer, "task1_eval_row", orig)
    ev(sp, out, ups)
    rows = T.read_jsonl(os.path.join(out, "test.jsonl"))
    assert sorted(r["update"] for r in rows if r["kind"] == "summary") == ups
    f0 = json.load(open(sp))["folds"][0]
    for u in ups:        # no episode evaluated twice with the same policy (finished rows reused)
        k = [(r["conversation_id"], r["seed"]) for r in rows if r["kind"] == "episode" and r["update"] == u]
        assert len(k) == len(set(k)) == 3 * len(f0["test"])
    r = boot(out)
    assert r.returncode == 0 and "TEST CHECK PASSED" in r.stdout, r.stdout + r.stderr


def test_unclean_pair_left_out(monkeypatch):
    sp, out, ups = _setup()
    bad = sorted(json.load(open(sp))["folds"][0]["test"])[0]
    orig = T.FakeEnv.run_episode

    def unclean(self, conversation_id, seed, *a, **k):
        ep = orig(self, conversation_id, seed, *a, **k)
        if conversation_id == bad and seed == 1:
            ep["clean"] = False
        return ep

    monkeypatch.setattr(T.FakeEnv, "run_episode", unclean)
    ev(sp, out, ups)
    rows = T.read_jsonl(os.path.join(out, "test.jsonl"))
    summ = {r["update"]: r for r in rows if r["kind"] == "summary"}
    assert all(s["n_unclean_episodes"] == 1 and s["unclean_after_retries"] for s in summ.values())
    assert sum(1 for r in rows if r["kind"] == "episode" and r["conversation_id"] == bad and r["seed"] == 1) \
        == 2 * (T.VAL_RETRIES + 1)                        # re-run up to VAL_RETRIES times, per update
    r = boot(out)
    assert r.returncode == 0 and "WARNING: 1 (conversation, seed) pairs" in r.stdout, r.stdout + r.stderr


def test_gates():
    sp, out, ups = _setup()
    base = args(sp, out, controller="llm", updates=4)
    with pytest.raises(SystemExit):                                   # no --final
        E.main(["--test-seeds", "0", "--test-updates"] + [str(u) for u in ups] + base)
    with pytest.raises(SystemExit):                                   # not the re-selected checkpoint
        E.main(TEST + ["--test-updates", "0", "3"] + base)
    with pytest.raises(SystemExit):                                   # changed training argument (provenance)
        E.main(TEST + ["--test-updates"] + [str(u) for u in ups] + base + ["--seed", "7"])
    with pytest.raises(SystemExit):                                   # re-selection flags are not accepted
        E.main(TEST + ["--test-updates"] + [str(u) for u in ups] + base + ["--reselect-seeds", "0"])
    assert not os.path.exists(os.path.join(out, "test.jsonl"))


def test_test_ids_must_be_forbidden():
    d = tempfile.mkdtemp()
    sp = make_splits(d)
    s = json.load(open(sp))
    s["folds"][0]["test_all"] = s["folds"][0]["test_all"] + [s["folds"][0]["train"][0]]   # a training id in test_all
    json.dump(s, open(sp, "w"))
    a = T.parse_args(args(sp, os.path.join(d, "run"), controller="llm", updates=4))
    with pytest.raises(AssertionError):
        E.test_ids(a, T.load_split(sp, 0))
