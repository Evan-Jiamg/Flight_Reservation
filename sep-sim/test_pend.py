# -*- coding: utf-8 -*-
"""pend arm pieces that need no model: reward v3, Task 1 stop metrics, parallel rollouts in the trainer."""
import json
import os
import shutil
import tempfile

import pytest

import rl_reward as RR
import task1_stop as T1
import train_planner_rl as T
from test_rl_advantages import make_splits, args, strip, run_ok


def ep(turns, human, cov, steps=None, **step_flags):
    n = steps if steps is not None else max(1, turns)
    trace = [dict({"t": t + 1}, **step_flags) for t in range(n)]
    return {"emitted_user_turns": turns, "human_turns": human, "coverage": cov, "decision_steps": n,
            "trace": trace, "end_kind": "planner_end"}


def test_reward_v3_values():
    cfg = RR.default_cfg(version="v3")
    r = RR.reward(ep(4, 4, 0.5), cfg)
    assert r["total"] == pytest.approx(0.5) and r["components"]["len_err"] == 0
    r = RR.reward(ep(10, 4, 0.5), cfg)
    assert r["total"] == pytest.approx(0.5 - 0.6)
    r = RR.reward(ep(2, 5, 1.0), cfg)
    assert r["total"] == pytest.approx(1.0 - 0.3) and r["components"]["turn_diff"] == -3
    # constraints
    cfg2 = RR.default_cfg(version="v3", lambda_unparsed=1.0)
    r = RR.reward(ep(4, 4, 0.5, planner_unparsed=True), cfg2)
    assert r["total"] == pytest.approx(0.5 - 1.0)
    # v2 path unchanged by the dispatcher
    assert RR.reward.__name__ == "reward" and RR.validate(RR.default_cfg())["version"] == "v2"
    with pytest.raises(ValueError):
        RR.reward({**ep(4, 4, 0.5), "coverage": 1.5}, cfg)
    e = ep(4, 4, 0.5)
    del e["human_turns"]
    with pytest.raises(ValueError):
        RR.reward(e, cfg)
    with pytest.raises(ValueError):
        RR.default_cfg(version="v9")
    agg = RR.aggregate([(ep(4, 4, 0.5), RR.reward(ep(4, 4, 0.5), cfg)), (ep(10, 4, 0.1), RR.reward(ep(10, 4, 0.1), cfg))])
    assert agg["components_mean"]["turns"] == 7 and agg["components_mean"]["coverage"] == pytest.approx(0.3)


def conv(n, ends):
    return {"turns": [{"t": t, "real_final": t == n, "ended_planner": t in ends, "planner_unparsed": False}
                      for t in range(1, n + 1)]}


def test_task1_stop_metrics():
    m = T1.task1_stop_metrics([conv(3, {3}), conv(4, {2}), conv(2, set())])
    # tp=1 (conv1 t3), fp=1 (conv2 t2), fn=2 (conv2 t4, conv3 t2)
    assert m["term_f1"] == pytest.approx(2 / (2 + 1 + 2))
    assert m["premature"] == pytest.approx(1 / 3) and m["final_recall"] == pytest.approx(1 / 3)
    assert m["false_end_rate"] == pytest.approx(1 / 6)
    assert T1.within_tolerance(m, m, 0.0)
    worse = dict(m, premature=m["premature"] + 0.1)
    assert not T1.within_tolerance(worse, m, 0.05) and T1.within_tolerance(worse, m, 0.1 + 1e-9)
    with pytest.raises(ValueError):
        T1.task1_stop_metrics([{"turns": [{"t": 1, "real_final": False, "ended_planner": False}]}])


def test_parallel_rollouts_equal_serial():
    """Threaded episodes must give exactly the rows/updates of the serial run (order-free)."""
    d = tempfile.mkdtemp()
    try:
        sp = make_splits(d)
        a, b = os.path.join(d, "serial"), os.path.join(d, "par")
        run_ok(args(sp, a, updates=3, controller="fixed"))
        run_ok(args(sp, b, "--rollout-workers", "4", updates=3, controller="fixed"))
        key = lambda r: (r["update"], r["slot"], r["replicate"])
        ra = sorted(strip(T.read_jsonl(os.path.join(a, "rollouts.jsonl"))), key=key)
        rb = sorted(strip(T.read_jsonl(os.path.join(b, "rollouts.jsonl"))), key=key)
        assert ra == rb
        ua, ub = strip(T.read_jsonl(os.path.join(a, "updates.jsonl"))), strip(T.read_jsonl(os.path.join(b, "updates.jsonl")))
        assert ua == ub
        assert json.load(open(os.path.join(a, "best.json"))) == json.load(open(os.path.join(b, "best.json")))
        # pend default: reward v3 was used, and validation summaries carry the turn statistics
        assert ua[0]["cfg_used"]["version"] == "v3"
        summ = [v for v in T.read_jsonl(os.path.join(b, "validation.jsonl")) if v["kind"] == "summary"]
        assert summ and all(s["turn_stats"] and "abs_diff_mean" in s["turn_stats"] for s in summ)
    finally:
        shutil.rmtree(d, ignore_errors=True)
