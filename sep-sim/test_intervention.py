"""User-approved intervention on the LLM (v4 factor) controller: values set once, bounds at every launch, rollback
inside the bounds (rl_controllers); v17 (S9): the trainer refuses --intervention (fixed controller); the verifier's
rl.intervention check is tested on the archived v16 run (test_v16). No GPU."""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_controllers as RC  # noqa: E402
import train_planner_rl as T  # noqa: E402
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


def test_v17_refuses_an_intervention(tmp_path, capsys):
    """SPEC v17 S9 (user 2026-09-30): the controller is fixed, so --intervention is refused with a reason; the verifier's
    rl.intervention check still serves archived v16 runs (test_v16.test_intervention_verified_on_archived_run)."""
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    ivp = str(tmp_path / "iv.json")
    json.dump(IV, open(ivp, "w"))
    with pytest.raises(SystemExit):
        T.main(args(sp, out, "--intervention", ivp, controller="llm", updates=2))
    assert "not available in v17" in capsys.readouterr().err
    assert not os.path.exists(os.path.join(out, "run_meta.jsonl"))
