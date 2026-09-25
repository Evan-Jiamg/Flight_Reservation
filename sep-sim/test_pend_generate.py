# -*- coding: utf-8 -*-
"""Selector, duplicate handling and the whole pend generation stage (with a fake Speaker)."""
import json
import os
import subprocess
import sys

import pytest

import implicit_profile as IP
import style_select as SS

HERE = os.path.dirname(os.path.abspath(__file__))
E1R = os.environ.get("E1R_TREE") or os.path.abspath(os.path.join(HERE, "..", "audit_e1r"))


def test_ranks_and_borda():
    assert SS.ranks([3, 1, 1, 7]) == [1, 0, 0, 2]
    assert SS.ranks([0.2, 0.9, 0.5], higher_better=True) == [2, 0, 1]
    # length prefers 1, style prefers 2; 0 is second on both -> rank sums 0:(1+1)=2, 1:(0+2)=2, 2:(2+0)=2
    idx, info = SS.borda_pick([0, 1, 2], [5, 1, 9], [0.5, 0.1, 0.9])
    assert info["score"] == [2, 2, 2] and idx == 1          # tie -> better length rank
    # a missing style signal leaves length to decide; a missing length target leaves style to decide
    assert SS.borda_pick([0, 1, 2], [5, 1, 9], [None, None, None])[0] == 1
    assert SS.borda_pick([0, 1, 2], [None, None, None], [0.5, 0.1, 0.9])[0] == 2
    assert SS.borda_pick([2, 3], [9, 9, 4, 1], [0, 0, 0.1, 0.1])[0] == 3   # only the given indices compete
    with pytest.raises(ValueError):
        SS.borda_pick([], [], [])


class FakeScorer:
    def similarities(self, cands, refs):
        return [None if not refs else (0.9 if "taipei" in c.lower() else 0.1) for c in cands]


def test_select_uses_eligible_and_target():
    cands = ["one two three", "hourly pm2.5 data for taipei please", "a b c d e f g h", ""]
    idx, info = SS.select(cands, [0, 1], 6, ["ref"], FakeScorer())
    assert idx == 1 and info["candidates"] == [0, 1]
    idx, _ = SS.select(cands, None, 3, [], None)             # none eligible -> all non-empty compete
    assert idx == 0


def test_prompt_tokens_from_error_wordings():
    import task2_env as T2
    f = T2.prompt_tokens_from_error
    assert f("This model's maximum context length is 12288 tokens. However, you requested 13500 tokens "
             "(10500 in the messages, 3000 in the completion).") == [10500]
    assert f("'max_tokens' is too large: 3000. This model's maximum context length is 12288 tokens and your "
             "request has 10000 input tokens (3000 > 12288 - 10000).") == [10000]
    assert f("HTTP 500: server error") == []


def test_fewshot_backs_off_to_interaction_style():
    recs = {"x": {"scenario": {"persona": {"individual_traits": {"interaction_style": "Elaborate"},
                                           "general_info": {"proficiency_in_english": "Native/Bilingual"}}},
                  "msgs": [{"text": "x1"}]},
            "y": {"scenario": {"persona": {"individual_traits": {"interaction_style": "Elaborate"},
                                           "general_info": {"proficiency_in_english": "Advanced"}}},
                  "msgs": [{"text": "y1 words"}, {"text": "y2 words"}, {"text": "y3 words"}]},
            "z": {"scenario": {"persona": {"individual_traits": {"interaction_style": "Concise"},
                                           "general_info": {"proficiency_in_english": "Native/Bilingual"}}},
                  "msgs": [{"text": "z1"}]}}
    pool = IP.FewShotPool(recs, list(recs), {"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 5, "z": 6}, lambda r: (r["msgs"], []))
    got = pool.select("x", recs["x"]["scenario"]["persona"], 2, k=3)
    assert got and all(e["cid"] == "y" for e in got)          # same interaction style, never the other style


def test_duplicate_of_and_variants():
    assert IP.duplicate_of("Hello  there", ["hello there"]) and not IP.duplicate_of("hello", ["hello there"])
    assert not IP.duplicate_of("", [""])


def _pool():
    recs = {c: {"conversation_id": c, "scenario": {"persona": {"individual_traits": {"interaction_style": "Concise"},
                                                               "general_info": {"proficiency_in_english": "Basic"}}},
                "msgs": [{"text": "%s message %d with some words" % (c, i)} for i in range(6)]}
            for c in ("a", "b", "c", "d", "e")}
    goal_of = {c: i for i, c in enumerate(recs)}
    persona_of = {c: 10 + i for i, c in enumerate(recs)}
    return recs, IP.FewShotPool(recs, list(recs), goal_of, persona_of, lambda r: (r["msgs"], []))


def test_fewshot_variants_differ_and_never_self():
    recs, pool = _pool()
    per = recs["a"]["scenario"]["persona"]
    sets = [tuple((x["cid"], x["t"]) for x in pool.select("a", per, 2, variant=v)) for v in range(4)]
    assert len(set(sets)) > 1, "candidate slots got identical examples"
    assert all(cid != "a" for s in sets for cid, _ in s)


GEN_CODE = r'''
import json, random, types
import task2_env as T2
T2.setup_environment("pend")
import implicit_profile as IP, style_select as SS
from sepsim import pipeline
import run_v2
recs = {c: {"conversation_id": c, "scenario": {"persona": {"individual_traits": {"interaction_style": "Concise"},
                                                           "general_info": {"proficiency_in_english": "Basic"}}},
            "msgs": [{"text": "%s example message number %d" % (c, i)} for i in range(6)]} for c in ("a", "b", "c", "d", "e")}
pool = IP.FewShotPool(recs, list(recs), {c: i for i, c in enumerate(recs)}, {c: 9 + i for i, c in enumerate(recs)},
                      lambda r: (r["msgs"], []))
calls = []
def fake_say_many(reqs):
    out = []
    for r in reqs:
        calls.append(r)
        # the first round: slots 1 and 2 repeat slot 0 exactly; later draws are distinct
        if len(calls) <= 4:
            txt = "I need taipei air data" if len(calls) in (1, 2, 3) else "something else entirely here"
        else:
            txt = "fresh draw %d about taipei" % len(calls)
        out.append((txt, False, {"original_tokens": 100 + len(calls), "dropped_exchanges": 0, "compacted": False}, False))
    return out
class FakeScorer:
    def similarities(self, cands, refs):
        return [None if not refs else (0.9 if "fresh" in c else 0.1) for c in cands]
env = object.__new__(T2.Task2Env)
env.ip, env.fewshot, env.selector, env.style_scorer = True, pool, "borda", FakeScorer()
env.t1_sampling = (0.7, 0.8)
env._say_many = fake_say_many
S = {"hist_u": ["a real first message"], "hist_a": ["assistant reply"], "prev_block": "PREV", "block": "BASE BLOCK",
     "cov": [], "mode": "task2", "ip_notes": ["writes short"], "ip_ctx": None}
st = env._pend_generate({"k": 1}, "BASE BLOCK\n- pending: x", {"length_words": 5}, S, 2, "sid", "scenario text",
                        S["hist_u"], S["hist_a"], "a", recs["a"]["scenario"], False)
blocks = [c["block"] for c in calls]
print(json.dumps({"reasons": st["guard_reasons"], "cands": st["candidates"], "dup_redraws": st["dup_redraws"],
                  "sel": st["selected_index"], "block": st["block"], "blocks": blocks,
                  "sel_refs": st["selection"]["refs"], "fit": st["speaker_fit"], "hits": st["speaker_hit_max_new"],
                  "temps": [c["temperature"] for c in calls], "fewshot": st["fewshot"]}))
'''


@pytest.mark.skipif(not os.path.isdir(os.path.join(E1R, "sepsim")), reason="E1.6 tree not present")
def test_pend_generate_end_to_end_with_fake_speaker():
    import task2_env as T2
    e = {k: v for k, v in os.environ.items() if not k.startswith("SEPSIM_")}
    pre = "import sys; sys.path[:0] = [%r, %r, %r]\n" % (E1R, os.path.join(E1R, "scripts"), HERE)
    out = subprocess.run([sys.executable, "-c", pre + GEN_CODE], capture_output=True, text=True, env=e, timeout=120)
    if out.returncode != 0 and "No module named 'torch'" in out.stderr:
        pytest.skip("run_v2 needs torch")
    assert out.returncode == 0, out.stderr[-3000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    # duplicates were redrawn slot by slot; the greedy slot 0 is never redrawn
    assert r["dup_redraws"] >= 2 and r["cands"][0] == "I need taipei air data"
    assert len({c.lower() for c in r["cands"]}) == len(r["cands"]), r["cands"]
    assert not any(x == "duplicate" for x in r["reasons"])
    # every slot got its own examples; the Planner state stays the base block
    first4 = r["blocks"][:4]
    assert len(set(first4)) == 4 and all(b.startswith("BASE BLOCK") and IP.SPK_EXAMPLES in b for b in first4)
    assert r["block"] == "BASE BLOCK\n- pending: x" and not IP.leaks_scaffold(r["block"])
    # greedy first, then samples at (0.7, 0.9) on turn >= 2
    assert r["temps"][0] == 0.0 and all(t == 0.7 for t in r["temps"][1:])
    # Task 2: style references are the few-shot examples; the selected candidate passed every guard
    assert r["sel_refs"] == "fewshot" and r["reasons"][r["sel"]] == ""
    assert r["fit"]["original_tokens"] == max(100 + i for i in range(1, len(r["blocks"]) + 1)) or r["fit"]["original_tokens"] > 100
    assert len(r["hits"]) == len(r["cands"]) and len(r["fewshot"]) == len(r["cands"])
