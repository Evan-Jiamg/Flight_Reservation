"""v3 prompts must equal the original except for exactly the documented edits; read_plan_v3
must not force stops or clamp lengths.

Runs against the frozen sepsim sources (audit snapshot locally, the live tree on the server).
"""
import json
import os
import random
import sys

os.environ["SEPSIM_ACT_PRIOR"] = "nostopclobber"
HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE, "..", "audit_src"), "/home/mzjiang/Sep-Simulator"):
    if os.path.isdir(os.path.join(cand, "sepsim")):
        sys.path.insert(0, cand)
        break

from sepsim import agenda as AG, persona as P, planner_prompt as PP, state, stopping  # noqa: E402
import planner_prompt_v3 as V3  # noqa: E402

SCENARIOS = [
    {"persona": {"general_info": {"proficiency_in_english": "Proficient"},
                 "individual_traits": {"frustration_threshold": "Moderately quickly",
                                       "interaction_style": "Concise"},
                 "experience_with_ai": {"trust": "Somewhat trusting"}},
     "goal": {"topic": "Plant growth images with metadata", "context": "phenotyping", "stage": "focusing"},
     "persona_goal_interaction": {"domain_familiarity": "Moderately familiar", "known_datasets": []}},
    {"persona": {"individual_traits": {"frustration_threshold": "Very quickly", "interaction_style": "Elaborate"}},
     "goal": {"topic": "Letters from the 16th century", "context": "history"},
     "persona_goal_interaction": {}},
    {"persona": {}, "goal": {"topic": "x"}, "persona_goal_interaction": {}},
]
HARD_RULE_TEXT = ("[strong]", "[weak]", "[moderate]", "met FIRST governs", "does not decide on its own",
                  "word band", "inside the band", "performed worst")


def original(sc, prev_block, hu, ha, t, led, ag):
    return PP.user_prompt(sc, prev_block, hu, ha, t, ledger=led, prev_ann={}, agenda_view=ag.render(), p_end=None)


def expected_v3(sc, prev_block, hu, ha, t, led, ag, gs):
    o = original(sc, prev_block, hu, ha, t, led, ag)
    led_block = "\n\nWHAT HAS ACTUALLY HAPPENED (counted, not judged)\n" + led.brief()
    pend, held = ag.render()
    ag_block = "\n\nWHAT THEY STILL WANT\n- pending: %s" % pend + (("\n- put aside for now: %s" % held) if held else "")
    assert o.count(led_block) == 1 and o.count(ag_block) == 1 and o.count(V3.band_block(sc)) == 1
    o = o.replace(led_block + ag_block, "", 1)
    o = o.replace(V3.band_block(sc), "", 1)
    o = o.replace(V3.gives_up_suffix(sc) + "\n", "\n", 1)
    i = o.find(V3.ANCHOR)
    return o[:i] + V3.FACTS_HEAD + V3.facts_v3(led) + V3.goal_block(gs) + o[i:]


def test_user_prompt():
    rng = random.Random(0)
    n = 0
    for sc in SCENARIOS:
        led = stopping.StoppingLedger(sc)
        ag = AG.Agenda(AG.terms_of(sc["goal"]["topic"] + " " + sc["goal"].get("context", ""), 8))
        hu, ha = [], []
        prev_block = state.d0(P.initial_stage(sc.get("goal")))
        for t in range(1, 11):
            gs = {"status": "NOT ASSESSED"} if t == 1 else {
                "status": rng.choice(["SATISFIED", "PARTIAL", "NOT", "UNKNOWN"]),
                "unmet": rng.choice([[], ["license info"], ["license", "18th-century letters too"]])}
            got = V3.user_prompt_v3(sc, prev_block, hu, ha, t, led, gs, prev_ann={})
            exp = expected_v3(sc, prev_block, hu, ha, t, led, ag, gs)
            assert got == exp, (t, got, exp)
            st = got.split("THE CONVERSATION SO FAR")[0]
            for bad in ("useful replies", "best offered", "repeated the previous offer", "stopping condition",
                        "WHAT THEY STILL WANT", "unhelpful replies before", "HOW LONG THEY WRITE", "- band:"):
                assert bad not in st, bad
            assert "- turns so far: %d" % led.turns in got
            n += 1
            hu.append("user message %d about %s" % (t, sc["goal"]["topic"]))
            reply = "Here are datasets: " + " ".join(rng.sample(["alpha", "beta", "gamma", "delta", "plant",
                                                                 "letters", "images", "metadata"], 4))
            prev_reply = ha[-1] if ha else ""
            ha.append(reply)
            led.observe({}, reply, prev_reply)
            led.note_gain(rng.random())
            ag.retire_satisfied(reply, {})
            if AG.looks_like_new_offer(reply, prev_reply):
                ag.reset_on_new_offer()
    return n


def test_system_prompt():
    s0, s3 = PP.system_prompt(), V3.system_prompt_v3()
    exp = s0.replace(stopping.rules_block(), V3.rules_block_v3(), 1).replace(V3.OLD_LENGTH_PARA, V3.NEW_LENGTH_PARA, 1) \
        .replace(V3.OLD_STOP_FIELD, V3.NEW_STOP_FIELDS, 1).replace(V3.OLD_LEN_FIELD, V3.NEW_LEN_FIELD, 1) \
        .replace(V3.OLD_REAL_PERSON, V3.NEW_REAL_PERSON, 1).replace(V3.OLD_AFFECT_FIELD, V3.NEW_AFFECT_FIELDS, 1)
    assert s3 == exp
    assert "typed by the real person" not in s3 and '"patience":' in s3
    for bad in HARD_RULE_TEXT:
        assert bad not in s3, bad
    assert '"end_session":' in s3 and "Then decide end_session yourself" in s3
    for name in stopping.RULES:
        assert name in s3


def test_read_plan_v3():
    sc = SCENARIOS[0]                                         # Concise: legacy band 8..22 words
    led = stopping.StoppingLedger(sc)
    rng = random.Random(1)
    base = {"critique": "c", "gain": 0.4, "case_for_leaving": "l", "case_for_continuing": "c",
            "current_stage": "refining", "revealed": "r", "unrevealed": "u", "affect": "a", "terms": "t",
            "act_distribution": [{"move": "Inquire", "act": "ask_more", "p": 1.0}],
            "length_reason": "r", "next_step": "n"}
    # a named strong reason must NOT force a stop any more; long length is kept, not clamped
    f, d, end = V3.read_plan_v3(json.dumps(dict(base, stop_rule="satiation", end_session=False,
                                                length_words=150)), 3, sc, rng, led)
    assert end is False and f["act"] == "ask_more" and f["move"] == "Inquire" and f["stop_rule"] == "satiation"
    assert f["length_words"] == 150 and d["length_clamped"] is False and d.get("stop_override") is False
    # the Planner's own decision ends the session
    f, d, end = V3.read_plan_v3(json.dumps(dict(base, stop_rule="satiation", end_session=True,
                                                length_words=6)), 4, sc, rng, led)
    assert end is True and d["end_session_valid"]
    # invalid values are recorded, not repaired
    f, d, end = V3.read_plan_v3(json.dumps(dict(base, stop_rule="none", end_session="maybe",
                                                length_words="lots")), 5, sc, rng, led)
    assert end is False and not d["end_session_valid"] and "length_words" not in f
    f, d, end = V3.read_plan_v3(json.dumps(dict(base, end_session="TRUE", length_words=0)), 6, sc, rng, led)
    assert end is True and "length_words" not in f
    # a sampled Complete act without end_session is flagged, not changed
    f, d, end = V3.read_plan_v3(json.dumps(dict(base, end_session=False, length_words=5,
        act_distribution=[{"move": "Complete", "act": "settle", "p": 1.0}])), 7, sc, rng, led)
    assert end is False and f["act"] == "settle" and d["complete_act_without_end"] is True
    assert V3.read_plan_v3("not json", 8, sc, rng, led)[0] is None
    # patience now comes from the Planner and reaches the rendered state; invalid values fall back
    f, d, end = V3.read_plan_v3(json.dumps(dict(base, end_session=False, length_words=5, patience="thin")),
                                9, sc, rng, led)
    assert "patience thin" in state.render(f)
    f, d, end = V3.read_plan_v3(json.dumps(dict(base, end_session=False, length_words=5, patience="angry")),
                                10, sc, rng, led)
    assert "patience full" in state.render(f)     # state.from_plan's own validation (not ours)


def test_speaker_block():
    fields = {"current_stage": "refining", "revealed": "r", "unrevealed": "u", "affect": "ok", "patience": "full",
              "terms": "a, b", "move": "Complete", "act": "settle", "bench_act": "x", "next_step": "close",
              "length_words": 9, "length_reason": "short", "stop_rule": "satiation"}
    gs = {"status": "PARTIAL", "unmet": ["license", "format"]}
    b3 = V3.speaker_block_v3(fields, gs)
    b0 = PP.render_block(fields, ("orig terms", "held"))
    assert b3 == b0.replace("- pending: orig terms\n- put aside: held", "- pending: license; format")
    assert V3.speaker_block_v3(fields, {"status": "NOT ASSESSED"}).count(V3.FIRST_TURN_UNMET) == 1
    unk = V3.speaker_block_v3(fields, {"status": "UNKNOWN", "unmet": []})
    assert V3.UNKNOWN_UNMET in unk and "(nothing identified)" not in unk
    assert V3.UNKNOWN_UNMET in V3.goal_block({"status": "UNKNOWN", "unmet": []})


if __name__ == "__main__":
    n = test_user_prompt()
    test_system_prompt()
    test_read_plan_v3()
    test_speaker_block()
    print("planner_prompt_v3 ok: %d user prompts, system prompt, read_plan_v3, speaker block" % n)
