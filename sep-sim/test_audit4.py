# -*- coding: utf-8 -*-
"""Regression tests for the round-4 audit fixes (2026-09-26)."""
import json
import os
import random

import pytest

import implicit_profile as IP
import rl_controllers as RC
import task1_stop as T1
import train_planner_rl as T


BASE = ["--dry-run", "--fold", "0", "--out", "o"]


@pytest.mark.parametrize("extra", [["--kl", "0.2"], ["--val-seeds", "5"], ["--val-temperature", "1.0"],
                                   ["--stop-credit", "0"], ["--task1-convs", "0"], ["--stop-sup-weight", "0"],
                                   ["--stop-sup-anneal", "3"], ["--controller", "fixed"]])
def test_every_spec_setting_needs_an_ablation(extra, capsys):
    with pytest.raises(SystemExit):
        T.parse_args(BASE + extra)
    assert "--ablation" in capsys.readouterr().err
    T.parse_args(BASE + extra + ["--ablation", "test"])           # allowed when named


def test_initial_w_aux_must_lie_in_the_controller_bounds():
    with pytest.raises(SystemExit):
        T.parse_args(BASE + ["--stop-sup-weight", "8"])
    with pytest.raises(ValueError):
        RC.make_controller("llm", RC.initial_cfg(version="v4", lambda_unparsed=1.0, lambda_hit_max_new=1.0, w_cov=7.0),
                           transport=lambda r: None)


def test_torn_last_line_is_cut_before_the_next_append(tmp_path):
    p = str(tmp_path / "x.jsonl")
    T.append_jsonl(p, {"a": 1})
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"a": 2, "b"')                                   # crash mid-write
    assert [r["a"] for r in T.read_jsonl(p)] == [1]
    T.append_jsonl(p, {"a": 3})
    assert [r["a"] for r in T.read_jsonl(p)] == [1, 3]           # no merged line, nothing corrupt


HERE = os.path.dirname(os.path.abspath(__file__))
E1R = os.environ.get("E1R_TREE") or os.path.abspath(os.path.join(HERE, "..", "audit_e1r"))
REDRAW_CODE = """
import json, random
import task2_env as T2
T2.setup_environment("pend")
import planner_prompt_v3 as V3
raw = json.dumps({"move": "Complete", "act": "settle", "end_session": False, "goal_met": "no", "still_wanted": "x",
                  "act_distribution": [{"move": "Complete", "act": "settle", "p": 0.9, "length_words": 5},
                                       {"move": "Reveal", "act": "summarise_things", "p": 0.05, "length_words": 9},
                                       {"move": "Inquire", "act": "ask_more", "p": 0.05, "length_words": 7}]})
out = []
for s in range(40):
    f, d, e = V3.read_plan_pend(raw, 3, {"goal": {}}, random.Random(s), None)
    out.append(d.get("complete_redrawn_to") if f is not None else None)
print(json.dumps(out))
"""


@pytest.mark.skipif(not os.path.isdir(os.path.join(E1R, "sepsim")), reason="E1.6 tree not present")
def test_complete_redraw_drops_malformed_entries():
    """A Complete act drawn without an end is redrawn among WELL-FORMED non-Complete entries only (as E1.6
    sample_act): the malformed 'summarise_things' entry must never become Other/other."""
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if not k.startswith("SEPSIM_")}
    pre = "import sys; sys.path[:0] = [%r, %r, %r]\n" % (E1R, os.path.join(E1R, "scripts"), HERE)
    out = subprocess.run([sys.executable, "-c", pre + REDRAW_CODE], capture_output=True, text=True, env=env, timeout=120)
    if out.returncode != 0 and "No module named 'torch'" in out.stderr:
        pytest.skip("the E1.6 tree needs torch here")
    assert out.returncode == 0, out.stderr[-2000:]
    redraws = [x for x in json.loads(out.stdout.strip().splitlines()[-1]) if x]
    assert redraws and all(x == ["Inquire", "ask_more"] for x in redraws), redraws


def test_scaffold_guard_catches_the_closing_line():
    assert IP.leaks_scaffold("ok thanks - this is their last message: they close the conversation")


def test_task1_metrics_count_capped_emissions():
    conv = {"turns": [{"t": 1, "real_final": False, "ended_planner": False, "emitted_capped": True},
                      {"t": 2, "real_final": True, "ended_planner": True}]}
    assert T1.task1_stop_metrics([conv])["n_emitted_capped_turns"] == 1
