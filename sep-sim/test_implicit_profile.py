# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys

import pytest

import implicit_profile as IP

HERE = os.path.dirname(os.path.abspath(__file__))
E1R = os.environ.get("E1R_TREE") or os.path.abspath(os.path.join(HERE, "..", "audit_e1r"))


def test_task1_context_reads_only_message_t_minus_1():
    real = ["r1", "r2", "r3", "r4"]
    preds = ["p1", "p2", "p3", "p4"]
    for t in range(2, 5):
        ctx = IP.task1_context(real, preds, t)
        assert ctx == {"mode": "task1", "pred_prev": preds[t - 2], "gold_prev": real[t - 2]}
        # changing message t or later (real or predicted) never changes turn t's context
        real2 = real[: t - 1] + ["CHANGED"] * (len(real) - t + 1)
        preds2 = preds[: t - 1] + ["CHANGED"] * (len(preds) - t + 1)
        assert IP.task1_context(real2, preds2, t) == ctx
        # and the caller may pass only the past: nothing beyond index t-2 is needed
        assert IP.task1_context(real[: t - 1], preds[: t - 1], t) == ctx
    assert IP.task1_context(real, preds, 1) is None
    with pytest.raises(ValueError):
        IP.task1_context(real, [], 3)


def test_planner_sections():
    notes = ["n%d" % i for i in range(10)]
    s = IP.render_planner_sections(notes, None)
    assert s.startswith(IP.IP_HEAD) and all("- n%d" % i in s for i in range(10))           # every note, uncut
    t1 = IP.render_planner_sections([], {"mode": "task1", "pred_prev": "Hello! Could you please list datasets?",
                                         "gold_prev": "need taipei pm2.5 data"})
    assert "(no notes yet)" in t1 and IP.T1_HEAD in t1 and 'you predicted: "Hello! Could you please list datasets?"' in t1
    assert "they wrote 4 words, you wrote 6" in t1 and "they asked 0 question(s), you asked 1" in t1
    assert "did not open with a greeting or thanks, you did" in t1 and "they started in lower case, you capitalised" in t1
    t2 = IP.render_planner_sections(["x"], {"mode": "task2"})
    assert IP.T2_HEAD in t2 and IP.T1_HEAD not in t2


def test_copies_example_and_speaker_lines():
    ex = [{"text": "I am looking for a dataset of hourly air quality readings in Taipei for my thesis"}]
    assert IP.copies_example("well, i am looking for a dataset of hourly air quality readings please", ex)
    assert not IP.copies_example("I need hourly PM2.5 data for Taipei, any public source?", ex)
    lines = IP.speaker_lines(["short", "no greetings"], ex)
    assert IP.SPK_NOTES + "short | no greetings" in lines and IP.SPK_EXAMPLES in lines
    assert IP.speaker_lines([], []) == ""
    assert IP.leaks_scaffold(lines.strip().splitlines()[0]) and IP.leaks_scaffold(lines.strip().splitlines()[1])
    assert not IP.leaks_scaffold("I need hourly PM2.5 data for Taipei, any public source?")


def _rec(cid, style, prof, texts):
    msgs = []
    for t in texts:
        msgs.append({"role": "user", "text": t})
    return {"conversation_id": cid, "scenario": {"persona": {"individual_traits": {"interaction_style": style},
                                                             "general_info": {"proficiency_in_english": prof}}},
            "msgs": msgs}


def _split(rec):
    return rec["msgs"], []


def test_fewshot_pool_excludes_self_goal_persona_and_matches_style():
    recs = {
        "a": _rec("a", "Concise", "Basic", ["a1", "a2"]),
        "b": _rec("b", "Concise", "Basic", ["b1", "b2"]),       # same goal as a
        "c": _rec("c", "Concise", "Basic", ["c1", "c2"]),       # same persona as a
        "d": _rec("d", "Concise", "Basic", ["d1", "d2", "d3"]),
        "e": _rec("e", "Elaborate", "Advanced", ["e1", "e2"]),  # other style
        "f": _rec("f", "Concise", "Basic", ["f1"]),             # not allowed (e.g. validation/test)
    }
    goal_of = {"a": 1, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}
    persona_of = {"a": 10, "b": 11, "c": 10, "d": 12, "e": 13, "f": 14}
    pool = IP.FewShotPool(recs, ["a", "b", "c", "d", "e"], goal_of, persona_of, _split)
    got = pool.select("a", recs["a"]["scenario"]["persona"], 2, k=3)
    assert got and {x["cid"] for x in got} == {"d"}
    assert pool.select("a", recs["a"]["scenario"]["persona"], 2, k=3) == got          # deterministic
    first = pool.select("a", recs["a"]["scenario"]["persona"], 1, k=1)
    assert first[0]["t"] == 1 and first[0]["cid"] == "d"
    assert all(x["cid"] != "f" for x in pool.items)


def run(env, code):
    e = {k: v for k, v in os.environ.items() if not k.startswith("SEPSIM_")}
    e.update(env)
    pre = "import sys, json; sys.path[:0] = [%r, %r]\n" % (E1R, HERE)
    out = subprocess.run([sys.executable, "-c", pre + code], capture_output=True, text=True, env=e, timeout=120)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not os.path.isdir(os.path.join(E1R, "sepsim")), reason="E1.6 tree not present")
def test_pend_prompts_with_implicit_profile():
    import task2_env as T2
    env = T2.ARM_ENV["pend"]
    s_on = run(env, "import planner_prompt_v3 as V\nprint(json.dumps(V.system_prompt_pend(implicit_profile=True)))")
    s_off = run(env, "import planner_prompt_v3 as V\nprint(json.dumps(V.system_prompt_pend()))")
    assert s_on.count('"profile_note"') == 1 and '"profile_note"' not in s_off
    sc = {"goal": {"topic": "t", "context": "c"}, "persona": {"individual_traits": {"interaction_style": "Concise"}}}
    code = ("from sepsim import stopping, state, persona as P\nimport planner_prompt_v3 as V, implicit_profile as IP\n"
            "sc = json.loads(%r)\nled = stopping.StoppingLedger(sc)\n"
            "sec = IP.render_planner_sections(['short'], {'mode': 'task2'})\n"
            "print(json.dumps(V.user_prompt_pend(sc, state.d0('exploring'), ['hi'], ['hello'], 2, led, prev_ann={}, ip_sections=sec)))"
            ) % json.dumps(sc)
    up = run(env, code)
    static = up.split("\n\nTHE CONVERSATION SO FAR\n", 1)[0]
    assert IP.IP_HEAD.strip() in static and IP.T2_HEAD.strip() in static
    assert up.index(IP.IP_HEAD.strip()) < up.index("THEY HAVE SENT ")
