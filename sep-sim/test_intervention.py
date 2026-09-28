"""User-approved intervention on the LLM (v4 factor) controller: values set once, bounds at every launch,
rollback inside the bounds, provenance, and the verifier's rl.intervention check (dry-run loop, no GPU)."""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_controllers as RC  # noqa: E402
import train_planner_rl as T  # noqa: E402
import verify_pipeline as V  # noqa: E402
from test_rl_advantages import args, make_splits  # noqa: E402

IV = {"set_cfg": {"w_dist": 1.0}, "controller_bounds": {"w_dist": [1.0, 5.0]},
      "reason": "test: turns collapsing while w_dist was halved", "approved": "test"}


def v4ctl():
    return RC.LLMFactorController(RC.initial_cfg(version="v4", lambda_unparsed=1.0, lambda_hit_max_new=1.0),
                                  transport=lambda req: None)


def test_intervene_sets_value_and_drops_pending():
    c = v4ctl()
    c.cfg["w_dist"] = 0.5
    c.pending = {"prev_cfg": dict(c.cfg), "baseline": 0.1, "bad": 1}
    c.set_bounds("w_dist", 1.0, 5.0)
    cfg = c.intervene({"w_dist": 1.0}, 16, "r")
    assert cfg["w_dist"] == 1.0 and c.pending is None
    assert c.decisions[-1]["human_intervention"] == {"w_dist": 1.0}


def test_intervene_outside_bounds_rejected():
    c = v4ctl()
    c.set_bounds("w_dist", 1.0, 5.0)
    with pytest.raises(ValueError):
        c.intervene({"w_dist": 0.5}, 3, "r")
    with pytest.raises(ValueError):
        c.set_bounds("w_dist", 0.0, 100.0)
    with pytest.raises(ValueError):
        c.set_bounds("lr", 1e-5, 1e-4)


def test_rollback_stays_inside_bounds():
    c = v4ctl()
    prev = dict(c.cfg)
    prev["w_dist"] = 0.5
    c.set_bounds("w_dist", 1.0, 5.0)
    c.cfg["w_dist"] = 2.0
    c.pending = {"prev_cfg": prev, "baseline": 10.0, "bad": 1}
    hist = [{"split": "train", "update": u, "reward_mean": 0.0, "shadow_reward_mean": 0.0, "components_mean": {}} for u in range(1, 6)]
    out = c.propose(hist)
    assert out["w_dist"] == 1.0              # rolled back, clamped to the tightened lower bound


def _run(argv):
    T.main(argv)


def test_dry_run_intervention_resume_and_verify():
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    ivp = os.path.join(d, "iv.json")
    json.dump(IV, open(ivp, "w"))
    _run(args(sp, out, controller="llm", updates=2))
    _run(args(sp, out, "--resume", "--intervention", ivp, controller="llm", updates=4))
    upd = T.read_jsonl(os.path.join(out, "updates.jsonl"))
    assert [u["cfg_used"]["w_dist"] for u in upd if u["update"] >= 3][0] == 1.0
    st = json.load(open(os.path.join(out, "ckpt", "u00004", "state.json")))
    assert st["intervention"]["at_update"] == 3 and st["intervention"]["set_cfg"] == {"w_dist": 1.0}
    rep = V.Report()
    V.check_intervention(out, rep)
    assert rep.checks["rl.intervention"]["fail"] == 0 and rep.checks["rl.intervention"]["n"] > 0
    # a later launch without the file, or with a changed file, is refused
    with pytest.raises(SystemExit):
        _run(args(sp, out, "--resume", controller="llm", updates=5))
    json.dump({**IV, "reason": "changed"}, open(ivp, "w"))
    with pytest.raises(SystemExit):
        _run(args(sp, out, "--resume", "--intervention", ivp, controller="llm", updates=5))
    json.dump(IV, open(ivp, "w"))
    _run(args(sp, out, "--resume", "--intervention", ivp, controller="llm", updates=5))
    st = json.load(open(os.path.join(out, "ckpt", "u00005", "state.json")))
    assert st["intervention"]["at_update"] == 3                       # applied once
    # the verifier catches a weight outside the intervention bounds
    rows = T.read_jsonl(os.path.join(out, "updates.jsonl"))
    rows[-1]["cfg_used"]["w_dist"] = 0.5
    with open(os.path.join(out, "updates.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    rep = V.Report()
    V.check_intervention(out, rep)
    assert rep.checks["rl.intervention"]["fail"] == 1
