"""SPEC v16 audit P tests: F1 (SPEC gate for earlier approved values), F2 (v16 runs select on bal_p), F3 (re-selection
summaries recomputed), F4 (refill decision and the floor value re-derived), F6 (controller w_aux after the anneal)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_planner_rl as T  # noqa: E402
import verify_pipeline as V  # noqa: E402
from test_v16 import _rewrite, fresh, run, v16_report  # noqa: E402

REAL = ["--fold", "2", "--out", "o", "--splits", "s", "--planner-path", "Qwen3-4B-Instruct-2507"]


@pytest.mark.parametrize("extra", [["--G", "8"], ["--behav-mismatch-abort", "0.5"], ["--w-sel-task1", "0"],
                                   ["--w-sel-cov", "2"], ["--w-sel-w1", "0.5"], ["--batch", "0"], ["--task1-tol", "1"],
                                   ["--stop-sup-weight", "3"], ["--scenarios-per-update", "1"], ["--val-every", "2"]])
def test_f1_earlier_approved_values_need_an_ablation(extra):
    with pytest.raises(SystemExit):
        T.parse_args(REAL + extra)
    assert T.parse_args(REAL + extra + ["--ablation", "x"])


def test_f1_formal_flags_are_spec():
    a = T.parse_args(REAL + ["--gpu", "1", "--G", "4", "--scenarios-per-update", "4", "--updates", "5", "--val-every", "5",
                             "--rollout-workers", "4", "--resume"])
    assert a.ablation is None


def _selection_fails(out):
    rep = V.Report()
    V.check_selection(os.path.join(out, "validation.jsonl"), out, rep)
    return rep.checks["rl.selection"]["fail"]


def test_f2_v16_run_must_select_on_bal_p():
    sp, out = fresh(updates=2)
    assert _selection_fails(out) == 0
    p = os.path.join(out, "validation.jsonl")
    rows = T.read_jsonl(p)
    for r in rows:
        if r["kind"] == "summary":
            r["selection_task1_metric"] = "term_f1"
            if r["selection_score"] is not None:
                r["selection_score"] += r["w_sel_task1"] * (r["task1"]["term_f1"] - r["task1"]["bal_p"])
    _rewrite(p, rows)
    assert _selection_fails(out) >= 1


def test_f3_reselect_summaries_recomputed():
    sp, out = fresh(updates=2)
    run(sp, out, "--reselect-seeds", "0", "1", "2", updates=2)
    assert all(c["fail"] == 0 for c in v16_report(out).checks.values())
    p = os.path.join(out, "reselect.jsonl")
    rows = T.read_jsonl(p)
    for r in rows:
        if r["kind"] == "summary" and r["selection_score"] is not None:
            r["selection_score"] += 0.25
            break
    _rewrite(p, rows)
    rep = v16_report(out)
    assert rep.checks["rl.selection"]["fail"] >= 1


def test_f4_refill_decision_and_floor_value():
    sp, out = fresh(updates=3)
    assert all(c["fail"] == 0 for c in v16_report(out).checks.values())
    p = os.path.join(out, "updates.jsonl")
    ups = T.read_jsonl(p)
    for u in ups:
        th = u["train_aggregate"]["task1_train"]
        th["refill_stop"] = "cap" if th["refill_stop"] != "cap" else "filled"   # a stop reason that does not fit
    ups[0]["train_aggregate"]["aux_floor"] = 0.3                              # not the run's --stop-sup-floor
    _rewrite(p, ups)
    rep = v16_report(out)
    assert rep.checks["rl.task1_refill"]["fail"] >= 1 and rep.checks["rl.aux_floor"]["fail"] >= 1


def test_f6_controller_scales_aux_after_the_anneal():
    sp, out = fresh("--t1-trigger-margin", "-1", "--stop-sup-anneal", "2", "--ablation", "test-f6", updates=8)
    ups = T.read_jsonl(os.path.join(out, "updates.jsonl"))
    late = [u for u in ups if u["update"] >= 7]                                # anneal over (trigger u4, 2 updates)
    for u in late:
        ag, w = u["train_aggregate"], u["cfg_used"]["w_aux"]
        assert abs(ag["aux_weight"] - max(0.5, w * 0.5)) < 1e-12                 # factor floor 0.5 / 1.0
    assert any(u["cfg_used"]["w_aux"] > 1.0 and u["train_aggregate"]["aux_weight"] > 0.5 for u in late), \
        "the stub controller raises w_aux: the effective weight must follow it"
    assert all(c["fail"] == 0 for c in v16_report(out).checks.values())


def test_r1_refill_conversation_without_rows_is_not_a_false_failure(monkeypatch):
    # a drawn refill conversation that leaves no row (one message / capped history) still counts in n_refill_convs
    import tempfile
    from test_v16 import make_big_splits
    orig = T.FakeEnv.task1_sample

    def never_end(self, *a, **k):
        out = orig(self, *a, **k)
        for x in out:
            x["ended_planner"] = False
            x["reward"] = float(not x["real_final"])
        return out
    monkeypatch.setattr(T.FakeEnv, "task1_sample", never_end)
    d = tempfile.mkdtemp()
    sp, out = make_big_splits(d), os.path.join(d, "run")
    run(sp, out, updates=1)
    pu, pt = os.path.join(out, "updates.jsonl"), os.path.join(out, "rollouts_task1.jsonl")
    ups, t1 = T.read_jsonl(pu), T.read_jsonl(pt)
    th = ups[0]["train_aggregate"]["task1_train"]
    assert th["refill_stop"] == "cap" and th["n_refill_convs"] == 8
    gone = th["refill_convs"][0]                                             # pretend it produced no row
    th["refill_convs"] = th["refill_convs"][1:]
    th["groups"] = [g for g in th["groups"] if g[0] != gone]
    th["n_refill_groups"] = sum(1 for g in th["groups"] if g[2])
    _rewrite(pu, ups)
    _rewrite(pt, [r for r in t1 if r["conversation_id"] != gone])
    rep = v16_report(out)
    assert rep.checks["rl.task1_refill"]["fail"] == 0, rep.checks["rl.task1_refill"]
