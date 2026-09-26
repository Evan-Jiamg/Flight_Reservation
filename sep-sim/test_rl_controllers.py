"""Pure-python tests of the controllers. Run: python test_rl_controllers.py"""
import json
import math
import os
import tempfile

import rl_controllers as RC


def hist(u, rates, split="train", **extra):
    comp = {"goal": 0.5, "over_continue": 0.1, "early_stop": 0.0, "decision_steps": 5.0}
    comp.update({"rate_" + k: v for k, v in rates.items()})
    h = {"update": u, "split": split, "n_episodes": 8, "reward_mean": 0.3, "reward_std": 0.2,
         "components_mean": comp, "lr": 1e-5, "kl_coef": 0.04}
    h.update(extra)
    return h


R0 = {"unparsed": 0.0, "hit_max_new": 0.0, "no_survivor": 0.0, "judge_unknown": 0.0}


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    raise AssertionError("expected %s" % exc.__name__)


def test_fixed():
    c = RC.FixedController()
    cfg = c.propose([hist(1, R0)])
    assert cfg == RC.initial_cfg()
    c2 = RC.FixedController(RC.initial_cfg(lr=2e-5))
    c2.load_state_dict(c.state_dict())
    assert c2.cfg == c.cfg


def test_history_guard():
    c = RC.FixedController()
    raises(lambda: c.propose([hist(1, R0, split="validation")]))
    raises(lambda: c.propose([hist(1, R0, val_reward=1.0)]))
    raises(lambda: c.propose([hist(1, R0, test_reward=1.0)]))
    raises(lambda: c.propose([hist(1, R0, coverage=0.3)]))
    raises(lambda: RC.DualAscentController().propose([{"update": 1, "components_mean": {}}]))


def test_dual_ascent_math():
    c = RC.DualAscentController(eta=2.0, budget_unparsed=0.02, budget_hit_max_new=0.02, budget_no_survivor=0.05,
                                budget_judge_unknown=0.02)
    rates = {"unparsed": 0.12, "hit_max_new": 0.0, "no_survivor": 0.05, "judge_unknown": 0.52}
    cfg = c.propose([hist(1, rates)])
    assert math.isclose(cfg["lambda_unparsed"], 2.0 * 0.10)
    assert cfg["lambda_hit_max_new"] == 0.0                        # max(0, 0 + 2*(0-0.02))
    assert math.isclose(cfg["lambda_no_survivor"], 0.0, abs_tol=1e-12)
    assert math.isclose(cfg["lambda_judge_unknown"], 1.0)
    # below budget -> multiplier shrinks, never below 0
    cfg = c.propose([hist(1, rates), hist(2, dict(R0))])
    assert math.isclose(cfg["lambda_unparsed"], 0.2 - 0.04) and math.isclose(cfg["lambda_judge_unknown"], 1.0 - 0.04)
    # upper bound clip
    big = RC.DualAscentController(eta=100.0)
    cfg = big.propose([hist(1, {"unparsed": 1.0, "hit_max_new": 1.0, "no_survivor": 1.0, "judge_unknown": 1.0})])
    assert cfg["lambda_unparsed"] == 10.0
    # the multipliers only; lr / kl / weights untouched
    for k in ("lr", "kl_coef", "w_goal", "w_partial", "w_over", "w_early"):
        assert cfg[k] == RC.initial_cfg()[k]
    # uses only the LAST train aggregate
    c3 = RC.DualAscentController(eta=1.0)
    a = c3.propose([hist(1, {**R0, "unparsed": 0.5}), hist(2, R0)])
    assert a["lambda_unparsed"] == 0.0
    # state round-trip
    c4 = RC.DualAscentController()
    c4.load_state_dict(c.state_dict())
    assert c4.cfg == c.cfg and c4.dual == c.dual
    raises(lambda: RC.DualAscentController(eta=-1.0))
    raises(lambda: c4.load_state_dict(RC.FixedController().state_dict()))


def test_llm_controller_clip_and_log():
    seen = []

    def stub(req):
        seen.append(req)
        payload = json.loads(req["messages"][1]["content"])
        assert set(payload) == {"train_aggregates_last_rounds", "current", "bounds", "fixed_lagrange_multipliers"}
        for h in payload["train_aggregates_last_rounds"]:
            assert h["split"] == "train"
        return {"choices": [{"message": {"content": json.dumps(
            {"lr": 1.0, "kl_coef": 0.0, "w_goal": 1.2, "w_partial": 5.0, "lambda_unparsed": 9.0,
             "rationale": "x"})}}]}

    d = tempfile.mkdtemp()
    log = os.path.join(d, "llm.jsonl")
    c = RC.LLMController(log_path=log, transport=stub, window=2)
    cfg = c.propose([hist(1, R0), hist(2, R0), hist(3, R0, secret_extra=1)])
    req = seen[0]
    assert req["model"] == "gpt-5-mini" and req["reasoning_effort"] == "minimal"
    payload = json.loads(req["messages"][1]["content"])
    assert [h["update"] for h in payload["train_aggregates_last_rounds"]] == [2, 3]
    assert "secret_extra" not in req["messages"][1]["content"]            # only declared keys are sent
    assert math.isclose(cfg["lr"], 3 * RC.TRAIN_DEFAULTS["lr"])            # x3 per round cap
    assert math.isclose(cfg["kl_coef"], 0.04 / 3)                          # /3 per round cap
    assert math.isclose(cfg["w_goal"], 1.2)
    assert cfg["w_partial"] == 1.0                                         # x3 -> 1.5, then bound 1.0
    assert cfg["lambda_unparsed"] == 0.0                                   # not an LLM-controlled key
    rec = [json.loads(l) for l in open(log)]
    assert len(rec) == 1 and rec[0]["ok"] and len(rec[0]["request_sha256"]) == 64
    assert rec[0]["ignored_keys"] == ["lambda_unparsed"]
    # failure keeps the cfg and is logged
    c2 = RC.LLMController(log_path=log, transport=lambda r: {"choices": [{"message": {"content": "not json"}}]})
    before = dict(c2.cfg)
    assert c2.propose([hist(1, R0)]) == before and c2.n_failures == 1
    assert json.loads(open(log).read().splitlines()[-1])["ok"] is False
    # zero value can only grow to zero_ref * 3
    c3 = RC.LLMController(RC.initial_cfg(kl_coef=0.0), transport=lambda r: {"choices": [{"message": {"content": '{"kl_coef": 0.9}'}}]})
    assert math.isclose(c3.propose([hist(1, R0)])["kl_coef"], 3e-3)
    # state round-trip
    c4 = RC.LLMController(transport=stub)
    c4.load_state_dict(c.state_dict())
    assert c4.cfg == c.cfg
    raises(lambda: c.propose([hist(1, R0, split="validation")]))
    raises(lambda: RC.LLMController(keys=["lambda_unparsed"]))


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print("test_rl_controllers: %d tests passed" % n)
