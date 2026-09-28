"""SPEC v16 re-audit tests (N1-N5 and the remaining gaps F, H, I); dry-run loop, no GPU."""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_planner_rl as T  # noqa: E402
import verify_pipeline as V  # noqa: E402
from test_rl_advantages import make_splits  # noqa: E402
from test_v16 import _rewrite, fresh, run, v16_report  # noqa: E402


def test_n1_splits_path_from_cli_and_sha(tmp_path):
    sp, out = fresh(updates=2)
    moved = str(tmp_path / "elsewhere.json")
    os.replace(sp, moved)                                        # the run's own path no longer exists (other machine)
    rep = V.Report()
    V.check_v16(out, rep)
    assert rep.checks["rl.task1_refill"]["fail"] >= 1            # never skipped silently (gap I)
    rep = V.Report()
    V.check_v16(out, rep, splits_path=moved)                     # the verifier's --splits
    assert all(c["fail"] == 0 for c in rep.checks.values()), {k: c for k, c in rep.checks.items() if c["fail"]}
    other = str(tmp_path / "other.json")
    s_ = json.load(open(moved))
    s_["folds"][0]["test"] = s_["folds"][0]["test"][:-1]
    json.dump(s_, open(other, "w"))
    rep = V.Report()
    V.check_v16(out, rep, splits_path=other)                     # a different split file: sha differs
    assert rep.checks["rl.task1_refill"]["fail"] >= 1


def test_n2_update_without_task1_rows(monkeypatch):
    # every conversation has a single message: no decision position, no Task 1 row at all
    monkeypatch.setattr(T.FakeEnv, "human_turns", lambda self, cid: 1)
    monkeypatch.setattr(T.FakeEnv, "task1_prompts",
                        lambda self, cid: [{"t": 1, "n_real": 1, "real_final": True, "user_prompt": "f"}])
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    run(sp, out, "--val-every", "0", updates=1)
    u = T.read_jsonl(os.path.join(out, "updates.jsonl"))[0]
    th = u["train_aggregate"]["task1_train"]
    assert th is not None and th["n_all"] == 0 and th["groups"] == [] and len(th["base_convs"]) == 8
    rep = v16_report(out)
    assert all(c["fail"] == 0 for c in rep.checks.values()), {k: c for k, c in rep.checks.items() if c["fail"]}


def test_n3_initial_weight_below_floor_refused():
    base = ["--fold", "0", "--out", "o", "--splits", "s", "--dry-run", "--ablation", "x"]
    with pytest.raises(SystemExit):
        T.parse_args(base + ["--stop-sup-weight", "0.3"])
    assert T.parse_args(base + ["--stop-sup-weight", "0.3", "--stop-sup-floor", "0.2"]).stop_sup_floor == 0.2


def test_h_unscored_point_rule_and_n4_check():
    tr = T.Trainer.__new__(T.Trainer)

    class Env:
        def run_task1(self, cid, keep_prompts=False):
            return {"conversation_id": cid, "n_real": 3, "turns": [
                {"t": t, "real_final": t == 3, "ended_planner": False, "user_prompt": "p"} for t in (1, 2, 3)]}

        def task1_end_probe(self, cid, t, prompt, real_final):
            if t == 2:
                return {"valid": False, "decision_valid": False, "greedy_end": True}     # unparsed plan -> 0
            return {"valid": False, "decision_valid": True, "greedy_end": True}          # no value located -> greedy
    tr.env, tr.learner = Env(), None
    row = tr.task1_eval_row(5, "sha", "c1")
    ps = {p["t"]: p for p in row["end_probs"]}
    assert ps[2]["p_end"] == 0.0 and ps[3]["p_end"] == 1.0 and ps[3]["decision_valid"] is True
    assert all("user_prompt" not in x for x in row["task1"]["turns"])
    # N4: the verifier checks the rule on logged rows
    sp, out = fresh(updates=2)
    pv = os.path.join(out, "validation.jsonl")
    val = T.read_jsonl(pv)
    for r in val:
        if r["kind"] == "task1" and r["update"] == 2:
            r["end_probs"][0].update(valid=False, decision_valid=False, greedy_end=True, p_end=0.5)
            break
    _rewrite(pv, val)
    assert v16_report(out).checks["rl.task1_prob"]["fail"] >= 1


def test_n5_validation_without_task1_resets_the_streak():
    d = tempfile.mkdtemp()
    out = os.path.join(d, "run")
    os.makedirs(out)
    meta = {"kind": "start", "config": {"args": {"task1_G": 8, "task1_convs": 0, "t1_trigger_margin": 0.1, "splits": "x",
                                                 "fold": 0}, "algo_cfg": {"grpo_std_norm": False}}}
    _rewrite(os.path.join(out, "run_meta.jsonl"), [meta])
    summ = [{"kind": "summary", "update": 0, "task1": {"bal_p": 0.5}, "d2": None},
            {"kind": "summary", "update": 5, "task1": {"bal_p": 0.7}, "d2": {"met": True, "streak": 1, "triggered_at": None}},
            {"kind": "summary", "update": 10, "task1": None, "d2": None},
            {"kind": "summary", "update": 15, "task1": {"bal_p": 0.7}, "d2": {"met": True, "streak": 1, "triggered_at": None}}]
    _rewrite(os.path.join(out, "validation.jsonl"), summ)
    rep = V.Report()
    V.check_v16(out, rep)
    assert rep.checks["rl.d2_trigger"]["fail"] == 0


def test_f_intervention_lowering_w_dist_bound_is_honoured(tmp_path):
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    ivp = str(tmp_path / "iv.json")
    json.dump({"set_cfg": {"w_dist": 0.5}, "controller_bounds": {"w_dist": [0.5, 5.0]}, "reason": "test", "approved": "test"},
              open(ivp, "w"))
    run(sp, out, updates=2)
    run(sp, out, "--resume", "--intervention", ivp, updates=4)
    rep = v16_report(out)
    assert rep.checks["rl.w_dist_floor"]["fail"] == 0
    assert [u["cfg_used"]["w_dist"] for u in T.read_jsonl(os.path.join(out, "updates.jsonl")) if u["update"] == 3] == [0.5]
