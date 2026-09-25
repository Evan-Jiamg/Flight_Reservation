# -*- coding: utf-8 -*-
"""The E1.6 port (task2_env arms e16/final, planner_prompt_v3 ACT_FULL, ditto_e16) against the real
E1.6 tree (../audit_e1r = Sep-1st-Simulator-e1r @ cf19400) and the v2fix tree (../audit_src).

Each case runs in a subprocess: the sepsim switches are read from the environment, and the two
trees must never share one interpreter.
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
E1R = os.environ.get("E1R_TREE") or os.path.abspath(os.path.join(HERE, "..", "audit_e1r"))
V2F = os.environ.get("V2FIX_TREE") or os.path.abspath(os.path.join(HERE, "..", "audit_src"))
CORPUS = os.environ.get("TEST_CORPUS", "")

sys.path.insert(0, HERE)
import task2_env as T2  # noqa: E402  (constants only)

pytestmark = pytest.mark.skipif(not (os.path.isdir(os.path.join(E1R, "sepsim")) and
                                     os.path.isdir(os.path.join(V2F, "sepsim"))),
                                reason="audit trees not present")

SCENARIO = {
    "goal": {"topic": "air quality datasets for Taipei", "context": "I am writing a thesis on PM2.5",
             "discipline": "environmental science"},
    "persona": {"individual_traits": {"interaction_style": "Concise", "patience": "Low"},
                "general_info": {"proficiency_in_english": "Fluent"}},
    "persona_goal_interaction": {},
}


def run(tree, env, code):
    e = {k: v for k, v in os.environ.items() if not k.startswith("SEPSIM_")}
    e.update(env)
    pre = "import sys, json; sys.path[:0] = [%r, %r]\n" % (tree, HERE)
    out = subprocess.run([sys.executable, "-c", pre + code], capture_output=True, text=True, env=e, timeout=120)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


SYS = "from sepsim import planner_prompt as PP\nprint(json.dumps(PP.system_prompt()))"
SYS_V3 = "import planner_prompt_v3 as V\nprint(json.dumps(V.system_prompt_v3()))"


def test_e1r_flags_off_equals_v2fix():
    """G0: with every E1.x switch off, the E1.6 Planner system prompt is v2fix's, byte for byte."""
    off = {"SEPSIM_ACT_PRIOR": "off"}
    assert run(E1R, off, SYS) == run(V2F, off, SYS)
    v3 = {"SEPSIM_ACT_PRIOR": "nostopclobber"}
    assert run(E1R, v3, SYS_V3) == run(V2F, v3, SYS_V3)


def test_e16_system_prompt_is_e16():
    s = run(E1R, T2.ARM_ENV["e16"], SYS)
    assert '"last_reply_helpful"' in s                          # SELF_JUDGE
    assert '"length_words": <words if they made THIS move>' in s   # ACT_FULL
    assert "inside that move's band" in s
    assert '"end_session"' not in s


def test_final_system_prompt():
    s = run(E1R, T2.ARM_ENV["final"], SYS_V3)
    assert '"length_words": <words if they made THIS move>' in s   # per-act length kept
    assert "inside that move's band" not in s                  # band gone
    assert "and its own length_words: how many words this turn has if they made that move." in s
    assert '"end_session"' in s and '"patience"' in s
    assert '"last_reply_helpful"' not in s                      # SELF_JUDGE replaced by the goal judge
    assert "band you were given" not in s


def test_final_user_prompt_has_no_band_rows():
    code = ("from sepsim import stopping, state, persona as P\nimport planner_prompt_v3 as V\n"
            "sc = json.loads(%r)\nled = stopping.StoppingLedger(sc)\n"
            "b0 = state.d0(P.initial_stage(sc.get('goal')))\n"
            "up = V.user_prompt_v3(sc, b0, [], [], 1, led, {'status': 'NOT ASSESSED', 'unmet': []}, prev_ann={})\n"
            "print(json.dumps(up))") % json.dumps(SCENARIO)
    up = run(E1R, T2.ARM_ENV["final"], code)
    assert "HOW LONG THEY WRITE" not in up and "- Disclose: " not in up and "- band:" not in up
    assert "GOAL STATUS" in up and "- status: NOT ASSESSED" in up
    # the same call with ACT_FULL off would leave the rows behind if band_block ignored the switch
    code_e16 = code.replace("V.user_prompt_v3(sc, b0, [], [], 1, led, {'status': 'NOT ASSESSED', 'unmet': []}, prev_ann={})",
                            "__import__('sepsim.planner_prompt', fromlist=['x']).user_prompt(sc, b0, [], [], 1, ledger=led, prev_ann={}, agenda_view='', p_end=None)")
    up16 = run(E1R, T2.ARM_ENV["e16"], code_e16)
    assert "HOW LONG THEY WRITE" in up16 and "- Disclose: " in up16


def plan(entries, top_len=40, end=False):
    return json.dumps({
        "current_stage": "exploring", "affect": "fine", "patience": "full",
        "act_distribution": entries, "next_step": "ask for hourly data", "case_for_leaving": "x",
        "case_for_staying": "y", "stop_rule": "none", "end_session": end, "length_words": top_len,
        "gain": 0.3})


def test_final_read_plan_takes_the_drawn_acts_length():
    entries = [{"move": "Inquire", "act": "ask clarifying question", "p": 1.0, "length_words": 17},
               {"move": "Complete", "act": "thank and leave", "p": 0.0, "length_words": 4}]
    code = ("import random\nfrom sepsim import stopping\nimport planner_prompt_v3 as V\n"
            "sc = json.loads(%r)\nf, d, e = V.read_plan_v3(%r, 3, sc, random.Random(0), stopping.StoppingLedger(sc))\n"
            "print(json.dumps([f and f.get('length_words'), f and f.get('move'), d.get('length_source'), d.get('length_clamped'), e]))"
            ) % (json.dumps(SCENARIO), plan(entries, top_len=200))
    lw, move, src, clamped, end = run(E1R, T2.ARM_ENV["final"], code)
    if move == "Inquire":
        assert (lw, src) == (17, "act_entry")
    assert clamped is False and end is False
    # a length far outside any band is kept as is (no clamp)
    entries[0]["length_words"] = 900
    code2 = code.replace(plan([dict(entries[0], length_words=17), entries[1]], top_len=200), plan(entries, top_len=200))
    lw2, move2, src2, _, _ = run(E1R, T2.ARM_ENV["final"], code2)
    if move2 == "Inquire":
        assert (lw2, src2) == (900, "act_entry")


def test_final_turn1_end_is_ignored_and_logged():
    entries = [{"move": "Complete", "act": "thank and leave", "p": 1.0, "length_words": 4}]
    code = ("import random\nfrom sepsim import stopping\nimport planner_prompt_v3 as V\n"
            "sc = json.loads(%r)\nout = []\n"
            "for t in (1, 3):\n"
            "    f, d, e = V.read_plan_v3(%r, t, sc, random.Random(0), stopping.StoppingLedger(sc))\n"
            "    out.append([e, bool(d.get('end_session_t1_ignored')), d.get('end_session_raw')])\n"
            "print(json.dumps(out))") % (json.dumps(SCENARIO), plan(entries, end=True))
    (e1, ign1, raw1), (e3, ign3, raw3) = run(E1R, T2.ARM_ENV["final"], code)
    assert (e1, ign1, raw1) == (False, True, True)
    assert (e3, ign3, raw3) == (True, False, True)


def test_arm_env_consistency():
    assert T2.ARM_ENV["final"]["SEPSIM_ACT_PRIOR"] == "nostopclobber"
    assert T2.ARM_ENV["e16"]["SEPSIM_ACT_PRIOR"] == "off"
    for arm in ("e16", "final"):
        env = T2.ARM_ENV[arm]
        for k in ("SEPSIM_INTENT_PROSE", "SEPSIM_ENDGATE", "SEPSIM_ENDMASK_RETRY", "SEPSIM_NEXTSTEP_INLINE"):
            assert env[k] == "0"
        assert env["SEPSIM_NO_ANN"] == env["SEPSIM_ACT_FULL"] == env["SEPSIM_T1_SAMPLE"] == "1"
    assert T2.tree_of("final") == T2.tree_of("e16") == T2.TREE_E16 != T2.tree_of("a2")


def test_no_ann_strips_at_load(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"record_id": "r", "chat_messages": [
        {"role": "user", "text": "hi"}, {"role": "assistant", "text": "yo", "annotations": {"helpful": True}},
        {"role": "user", "text": "bye", "is_final": True}]}) + "\n", encoding="utf-8")
    code = ("from sepsim import pipeline\nrecs, fin = pipeline.load_corpus(%r)\n"
            "print(json.dumps(['annotations' in m for m in recs[0]['chat_messages']]))") % str(p)
    assert not any(run(E1R, T2.ARM_ENV["final"], code))
    assert any(run(E1R, {"SEPSIM_NO_ANN": "0"}, code))


def test_verify_recomputes_e16_shas():
    import verify_pipeline as VP
    for arm in ("e16", "final"):
        sha = VP.recompute_system_sha(arm, E1R)
        assert sha and len(sha) == 64
    assert VP.recompute_system_sha("e16", E1R) != VP.recompute_system_sha("final", E1R)
