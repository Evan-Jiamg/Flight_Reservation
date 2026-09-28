"""Pure-python tests of rl_reward.reward_v2 (hand-computable episodes). Run: python test_rl_reward.py"""
import math

import rl_reward as RR


def step(t, status=None, decision="continue", stop_rule="none", probs=None, **kw):
    gs = None if status is None else {"status": status, "unmet": []}
    if probs is not None:
        gs["status_probs"] = probs
    s = {"t": t, "decision": decision, "goal_status": gs, "ended_planner": decision == "planner_stop",
         "stop_rule": stop_rule, "planner_unparsed": False, "planner_hit_max_new": False, "no_survivor": False}
    s.update(kw)
    return s


def ep(trace, end_kind, **kw):
    e = {"conversation_id": "c", "trace": trace, "decision_steps": len(trace), "end_kind": end_kind}
    e.update(kw)
    return e


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-12)


def test_goal_last_assessed_one_hot_and_probs():
    cfg = RR.default_cfg()
    e = ep([step(1, "NOT ASSESSED"), step(2, "NOT"), step(3, "PARTIAL"), step(4, "UNKNOWN")], "t_max")
    r = RR.reward_v2(e, cfg)
    assert close(r["components"]["goal"], 0.5) and r["components"]["goal_t"] == 3      # UNKNOWN skipped
    probs = {"SATISFIED": 0.6, "PARTIAL": 0.3, "NOT": 0.1}
    e = ep([step(1, "NOT ASSESSED"), step(2, "SATISFIED", probs=probs)], "t_max")
    r = RR.reward_v2(e, RR.default_cfg(w_partial=0.25))
    assert close(r["components"]["goal"], 0.6 + 0.25 * 0.3)
    # no assessed step at all
    e = ep([step(1, "NOT ASSESSED", decision="planner_stop")], "planner_stop")
    r = RR.reward_v2(e, cfg)
    assert r["components"]["goal"] == 0.0 and r["components"]["goal_assessed"] is False
    assert r["components"]["early_stop"] == 0                     # NOT ASSESSED is not in early_stop_statuses


def test_over_continue_and_total():
    cfg = RR.default_cfg(w_goal=2.0, w_over=0.5, w_early=1.0)
    tr = [step(1, "NOT ASSESSED"), step(2, "SATISFIED"), step(3, "SATISFIED"),
          step(4, "SATISFIED", decision="planner_stop", stop_rule="satiation")]
    r = RR.reward_v2(ep(tr, "planner_stop"), cfg)
    c = r["components"]
    assert c["over_continue_steps"] == 2 and close(c["over_continue"], 0.5)
    assert c["early_stop"] == 0 and close(c["goal"], 1.0)
    assert close(r["total"], 2.0 * 1.0 - 0.5 * 0.5)


def test_early_stop_rules():
    cfg = RR.default_cfg(w_early=1.0)
    base = [step(1, "NOT ASSESSED"), step(2, "NOT")]
    r = RR.reward_v2(ep(base + [step(3, "NOT", decision="planner_stop", stop_rule="satiation")], "planner_stop"), cfg)
    assert r["components"]["early_stop"] == 1 and close(r["total"], 0.0 - 1.0)
    r = RR.reward_v2(ep(base + [step(3, "NOT", decision="planner_stop", stop_rule="disgust")], "planner_stop"), cfg)
    assert r["components"]["early_stop"] == 0                      # abandon reason is allowed
    r = RR.reward_v2(ep(base + [step(3, "PARTIAL", decision="planner_stop")], "planner_stop"), cfg)
    assert r["components"]["early_stop"] == 0
    r = RR.reward_v2(ep(base + [step(3, "NOT")], "t_max"), cfg)
    assert r["components"]["early_stop"] == 0                      # not a planner stop
    cfg2 = RR.default_cfg(abandon_reasons=[], w_early=1.0)
    r = RR.reward_v2(ep(base + [step(3, "NOT", decision="planner_stop", stop_rule="disgust")], "planner_stop"), cfg2)
    assert r["components"]["early_stop"] == 1


def test_constraint_rates_and_lambdas():
    cfg = RR.default_cfg(lambda_unparsed=1.0, lambda_hit_max_new=2.0, lambda_no_survivor=3.0, lambda_judge_unknown=4.0,
                         w_over=0.0, w_early=0.0)
    tr = [step(1, "NOT ASSESSED", planner_unparsed=True), step(2, "UNKNOWN", planner_hit_max_new=True),
          step(3, "PARTIAL", no_survivor=True), step(4, "PARTIAL")]
    r = RR.reward_v2(ep(tr, "t_max"), cfg)
    c = r["components"]
    assert (c["rate_unparsed"], c["rate_hit_max_new"], c["rate_no_survivor"], c["rate_judge_unknown"]) == (0.25,) * 4
    assert close(c["constraint_penalty"], 0.25 * 10)
    assert close(r["total"], 0.5 - 2.5)


def test_eval_only_fields_do_not_matter():
    cfg = RR.default_cfg()
    tr = [step(1, "NOT ASSESSED"), step(2, "PARTIAL")]
    a = RR.reward_v2(ep(tr, "t_max", coverage=0.0, complete=False), cfg)
    b = RR.reward_v2(ep(tr, "t_max", coverage=1.0, complete=True, ledger={"x": 1}), cfg)
    assert a == b


def test_bounds_and_validation():
    for bad in ({"w_partial": 1.5}, {"w_goal": -1}, {"lambda_unparsed": 11}):
        try:
            RR.default_cfg(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bounds not enforced for %r" % bad)
    try:
        RR.reward_v2(ep([step(1, "PARTIAL", probs={"SATISFIED": 0.5, "PARTIAL": 0.2, "NOT": 0.1})], "t_max"),
                     RR.default_cfg())
    except ValueError:
        pass
    else:
        raise AssertionError("probabilities not summing to 1 must be refused")
    try:
        RR.reward_v2({"trace": [step(1, "NOT")], "decision_steps": 2, "end_kind": "t_max"}, RR.default_cfg())
    except ValueError:
        pass
    else:
        raise AssertionError("decision_steps mismatch must be refused")


def test_aggregate():
    cfg = RR.default_cfg()
    e1 = ep([step(1, "NOT ASSESSED"), step(2, "SATISFIED")], "t_max")
    e2 = ep([step(1, "NOT ASSESSED"), step(2, "NOT")], "t_max")
    rows = [(e1, RR.reward_v2(e1, cfg)), (e2, RR.reward_v2(e2, cfg))]
    agg = RR.aggregate(rows)
    assert agg["n_episodes"] == 2 and close(agg["components_mean"]["goal"], 0.5)
    assert "coverage" not in str(agg) and "complete" not in str(agg)


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print("test_rl_reward: %d tests passed" % n)
