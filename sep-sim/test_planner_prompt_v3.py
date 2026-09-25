"""v3 prompt must equal the original except for exactly the documented edits.

Runs against the frozen sepsim sources (audit snapshot locally, the live tree on the server).
Method: build the ORIGINAL prompt with the real ledger and agenda, then apply the documented
edits to it by hand (remove the whole original ledger block and the agenda block, strip the
gives-up parenthetical, insert facts + goal status at the anchor) and require equality with
user_prompt_v3. Also checks the facts block contains no Task-1-only line.
"""
import os
import random
import sys

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


def original(sc, prev_block, hu, ha, t, led, ag):
    return PP.user_prompt(sc, prev_block, hu, ha, t, ledger=led, prev_ann={}, agenda_view=ag.render(), p_end=None)


def expected_v3(sc, prev_block, hu, ha, t, led, ag, gs):
    o = original(sc, prev_block, hu, ha, t, led, ag)
    led_block = "\n\nWHAT HAS ACTUALLY HAPPENED (counted, not judged)\n" + led.brief()
    pend, held = ag.render()
    ag_block = "\n\nWHAT THEY STILL WANT\n- pending: %s" % pend + (("\n- put aside for now: %s" % held) if held else "")
    assert o.count(led_block) == 1 and o.count(ag_block) == 1
    o = o.replace(led_block + ag_block, "", 1)
    o = o.replace(V3.gives_up_suffix(sc) + "\n", "\n", 1)
    i = o.find(V3.ANCHOR)
    return o[:i] + V3.FACTS_HEAD + V3.facts_v3(led) + V3.goal_block(gs) + o[i:]


def main():
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
            for bad in ("useful replies", "best offered", "repeated the previous offer",
                        "stopping condition", "WHAT THEY STILL WANT", "unhelpful replies before"):
                assert bad not in got.split("THE CONVERSATION SO FAR")[0], bad
            assert "- turns so far: %d" % led.turns in got
            n += 1
            # advance one exchange, feeding the ledger like the runner does
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
    # speaker block: only the pending slot differs from the original rendering
    fields = {"current_stage": "refining", "revealed": "r", "unrevealed": "u", "affect": "ok", "patience": "full",
              "terms": "a, b", "move": "Complete", "act": "settle", "bench_act": "x", "next_step": "close",
              "length_words": 9, "length_reason": "short", "stop_rule": "satiation"}
    gs = {"status": "PARTIAL", "unmet": ["license", "format"]}
    b3 = V3.speaker_block_v3(fields, gs)
    b0 = PP.render_block(fields, ("orig terms", "held"))
    assert b3 == b0.replace("- pending: orig terms\n- put aside: held", "- pending: license; format")
    assert V3.speaker_block_v3(fields, {"status": "NOT ASSESSED"}).count(V3.FIRST_TURN_UNMET) == 1
    print("planner_prompt_v3 ok on %d prompts; speaker block ok" % n)


if __name__ == "__main__":
    main()
