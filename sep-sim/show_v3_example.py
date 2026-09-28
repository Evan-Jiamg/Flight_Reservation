"""Print one v3 Planner prompt and Speaker block for manual inspection."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audit_src"))
from sepsim import state, stopping  # noqa: E402
import planner_prompt_v3 as V3  # noqa: E402
from test_planner_prompt_v3 import SCENARIOS  # noqa: E402

sc = SCENARIOS[0]
led = stopping.StoppingLedger(sc)
for g in (0.6, 0.4, 0.3):
    led.observe({}, "reply", "")
    led.note_gain(g)
gs = {"status": "PARTIAL", "unmet": ["environmental metadata per sample"]}
print(V3.user_prompt_v3(sc, state.d0("refining"), ["hi, I need plant image datasets"] * 3,
                        ["Here are three datasets ..."] * 3, 4, led, gs))
print("=" * 60)
print(V3.speaker_block_v3({"move": "Inquire", "act": "ask_more", "next_step": "ask about metadata",
                           "length_words": 14, "stop_rule": "none"}, gs))
