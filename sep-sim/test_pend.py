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


class _Tok:
    """ids index into a vocabulary of strings; decode concatenates (a stand-in for BPE pieces)."""

    def __init__(self, vocab):
        self.vocab = vocab

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[i] for i in ids)


def _mask(pieces):
    import types
    import task2_env as T2
    tok = _Tok(pieces)
    obj = types.SimpleNamespace(tok=tok)
    return T2.PlannerLM.stop_mask(obj, list(range(len(pieces))))


def test_stop_mask_marks_only_the_value_tokens():
    pieces = ['{"goal_met": "yes",', ' "end', '_session', '":', ' tr', 'ue', ',', ' "next_step": "x"}']
    m = _mask(pieces)
    assert m == [0, 0, 0, 0, 1, 1, 0, 0]
    pieces = ['{"end_session": ', 'false', '}']
    assert _mask(pieces) == [0, 1, 0]
    pieces = ['{"end_session": "', 'tr', 'ue"', '}']            # string form, value split mid-token
    assert _mask(pieces) == [0, 1, 1, 0]
    assert _mask(['{"act": "x"}']) is None                       # field absent -> no stop credit


class _Tok2(_Tok):
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab.index(text)]}


def test_stop_target_swaps_the_value_only():
    import types
    import task2_env as T2
    pieces = ['{"end_session":', ' false', ',', ' true']
    obj = types.SimpleNamespace(tok=_Tok2(pieces))
    tgt = T2.PlannerLM.stop_target(obj, [0, 1, 2], [0, 1, 0], want_end=True)
    assert tgt == {"prefix_ids": [0], "target_ids": [3], "want_end": True, "gen_len": 3}
    tgt = T2.PlannerLM.stop_target(obj, [0, 1, 2], [0, 1, 0], want_end=False)
    assert tgt == {"prefix_ids": [0], "target_ids": [1], "want_end": False, "gen_len": 3}
    assert T2.PlannerLM.stop_target(obj, [0, 1, 2], None, want_end=True) is None


def test_stop_supervision_moves_the_stop_policy():
    """GRPO Task 1 groups are all-or-nothing at the start (zero spread -> skipped); the auxiliary
    stop supervision still moves p(end | last message) up and p(end | earlier message) down."""
    d = tempfile.mkdtemp()
    try:
        sp = make_splits(d)
        on, off = os.path.join(d, "on"), os.path.join(d, "off")
        run_ok(args(sp, on, updates=4, controller="fixed"))
        run_ok(args(sp, off, "--stop-sup-weight", "0", updates=4, controller="fixed"))
        th = lambda o: json.load(open(os.path.join(o, "ckpt", "u00004", "fake_learner.json")))["theta"]
        t_on, t_off = th(on), th(off)
        assert t_on[3] > t_off[3], "final-message end logit did not rise with stop supervision"
        assert t_on[1] < t_off[1], "earlier-message end logit did not fall with stop supervision"
        ups = T.read_jsonl(os.path.join(on, "updates.jsonl"))
        assert all(u["learner_stats"].get("aux_n", 0) > 0 for u in ups)
        assert all("aux_n" not in u["learner_stats"] for u in T.read_jsonl(os.path.join(off, "updates.jsonl")))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def conv(n, ends, blanks=(), k1_blank=False):
    return {"k1_speaker_blank": k1_blank,
            "turns": [{"t": t, "real_final": t == n, "ended_planner": t in ends, "speaker_blank": t in blanks,
                       "planner_unparsed": False} for t in range(1, n + 1)]}


def test_task1_stop_metrics():
    # M2: the decision at t is the END flag at row t+1 (K+1 for t = n)
    m = T1.task1_stop_metrics([conv(3, {3}), conv(4, {2}), conv(2, set())])
    # TP=1 (conv1: decision at n -> K+1), FP=1 (conv2: decision at 2 -> END at row 3), FN=2 (conv2, conv3)
    assert m["term_f1"] == pytest.approx(2 / (2 + 1 + 2)) and m["end_mapping"] == "M2"
    assert m["premature"] == pytest.approx(1 / 3) and m["k1_end_rate"] == pytest.approx(1 / 3)
    assert m["premature_end_rate"] == pytest.approx(1 / 9)
    assert m["decision_f1"] == pytest.approx(2 / 5)
    # a Speaker blank is END on its own row; a blank at the probe is a K+1 end
    m2 = T1.task1_stop_metrics([conv(3, set(), blanks={2}), conv(2, set(), k1_blank=True)])
    assert m2["premature_end_rate"] == pytest.approx(1 / 5) and m2["k1_end_rate"] == pytest.approx(1 / 2)
    assert m2["term_f1"] == pytest.approx(2 * 1 / (2 * 1 + 1 + 1))
    # the decision at the last row is NOT an END on that row (M2 shift)
    flags, k1 = T1.bench_flags(conv(3, {3})["turns"])
    assert flags == [False, False, False] and k1
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
        # Task 1 stop groups: same rows serial/parallel, TRAIN conversations only, every update
        ka = lambda r: (r["update"], r["conversation_id"], r["t"])
        ta = sorted(strip(T.read_jsonl(os.path.join(a, "rollouts_task1.jsonl"))), key=ka)
        tb = sorted(strip(T.read_jsonl(os.path.join(b, "rollouts_task1.jsonl"))), key=ka)
        assert ta == tb and ta
        sp_ = json.load(open(sp))["folds"][0]
        assert all(r["conversation_id"] in sp_["train_all"] and r["conversation_id"] not in sp_["forbidden_for_training"] for r in ta)
        assert sorted({r["update"] for r in ta}) == [1, 2, 3]
        for r in ta:
            assert r["real_final"] == (r["t"] == r["n_real"]) and len(r["samples"]) == 4
            assert all(x["reward"] == float(x["ended_planner"] == r["real_final"]) for x in r["samples"])
        assert all(u["train_aggregate"]["task1_train"]["n"] > 0 for u in ua)
        # pend default: reward v4 (D1(b)) with the train-only p_h, and validation summaries carry the turn statistics
        assert ua[0]["cfg_used"]["version"] == "v4" and ua[0]["reward_ctx"]["p_h"] and ua[0]["reward_ctx"]["q"]
        assert all(u["train_aggregate"]["shadow_reward_mean"] is not None and u["train_aggregate"]["turn_hist"] for u in ua)
        summ = [v for v in T.read_jsonl(os.path.join(b, "validation.jsonl")) if v["kind"] == "summary"]
        assert summ and all(s["turn_stats"] and "abs_diff_mean" in s["turn_stats"] and "turn_w1" in s["turn_stats"] for s in summ)
    finally:
        shutil.rmtree(d, ignore_errors=True)
