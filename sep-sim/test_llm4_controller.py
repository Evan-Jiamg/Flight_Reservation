# -*- coding: utf-8 -*-
import json

import pytest

import rl_controllers as RC


def hist(n, shadow=None, turns=5.0):
    out = []
    for u in range(1, n + 1):
        out.append({"update": u, "split": "train", "reward_version": "v4", "reward_mean": 0.1 * u,
                    "shadow_reward_mean": (shadow[u - 1] if shadow else 0.5),
                    "components_mean": {"coverage": 0.9, "dist": -0.2, "turns": turns, "rate_unparsed": 0.01,
                                        "rate_hit_max_new": 0.0},
                    "turn_hist": [0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 0], "p_h": [0.01] * 11, "kl": 0.01})
    return out


def reply(obj):
    return {"choices": [{"message": {"content": "thinking done. " + json.dumps(obj)}}]}


def make(transport, **kw):
    cfg0 = RC.initial_cfg(version="v4", w_cov=1.0, w_dist=1.0, lambda_unparsed=1.0, lambda_hit_max_new=1.0)
    return RC.make_controller("llm", cfg0, transport=transport, **kw)


def test_picks_v4_controller_and_acts_every_5():
    calls = []
    c = make(lambda req: calls.append(req) or reply({"factors": {"w_dist": 2.0, "w_cov": 0.8}, "rationale": "x"}))
    assert isinstance(c, RC.LLMFactorController)
    for n in range(1, 5):
        assert c.propose(hist(n))["w_dist"] == 1.0
    assert not calls
    cfg = c.propose(hist(5))
    assert len(calls) == 1 and cfg["w_dist"] == 2.0 and cfg["w_cov"] == pytest.approx(0.8)
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert "train_statistics" in payload and payload["current"]["w_dist"] == 1.0


def test_only_allowed_factors_and_bounds():
    c = make(lambda req: reply({"factors": {"w_dist": 3.0}}))
    cfg = c.propose(hist(5))
    assert cfg["w_dist"] == 1.0 and c.n_failures == 1          # 3.0 is not an allowed factor: nothing applied
    c2 = make(lambda req: reply({"factors": {"w_dist": 2.0}}))
    for _ in range(4):                                          # 1 -> 2 -> 4 -> 5 (bound) -> 5
        cfg = c2.propose(hist(5))
    assert cfg["w_dist"] == 5.0


def test_rollback_after_two_worse_decision_points():
    c = make(lambda req: reply({"factors": {"w_cov": 2.0}}))
    cfg = c.propose(hist(5, shadow=[0.5] * 5))                  # change applied, baseline 0.5
    assert cfg["w_cov"] == 2.0 and c.pending is not None
    c.transport = lambda req: reply({"factors": {}})
    cfg = c.propose(hist(10, shadow=[0.5] * 5 + [0.3] * 5))     # worse once
    assert cfg["w_cov"] == 2.0 and c.pending["bad"] == 1
    cfg = c.propose(hist(15, shadow=[0.5] * 5 + [0.3] * 10))    # worse twice -> roll back
    assert cfg["w_cov"] == 1.0 and c.n_rollbacks == 1 and c.pending is None


def test_rejects_validation_keys_and_state_roundtrip():
    c = make(lambda req: reply({"factors": {}}))
    bad = hist(5)
    bad[-1]["validation_turns"] = 3
    with pytest.raises(ValueError):
        c.propose(bad)
    c.propose(hist(5))
    d = c.state_dict()
    c2 = make(lambda req: reply({"factors": {}}))
    c2.load_state_dict(json.loads(json.dumps(d)))
    assert c2.cfg == c.cfg and c2.decisions == c.decisions
