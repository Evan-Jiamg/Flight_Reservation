from stop_prompt_v2 import mask_rules, MARKER

EX = """WHO THEY ARE
- writes: Concise, English Proficient
- gives up: Moderately quickly (so about 3 unhelpful replies before frustration governs)
- starts at stage: refining

THE STATE YOU WROTE LAST TURN
- state: neutral; somewhat curious; patience full

WHAT HAS ACTUALLY HAPPENED (counted, not judged)
- turns so far: 3
- useful replies: 0, unhelpful: 0, unhelpful in a row: 0 of 3 this person tolerates
- best offered so far: nothing yet
- last reply repeated the previous offer: no
- no stopping condition is met

THEY HAVE SENT 3 MESSAGES SO FAR. You are writing the state for message 4.

""" + MARKER + "USER: turns so far: 3 (a user literally wrote this)\n"

m = mask_rules(EX)
assert "- gives up: Moderately quickly\n" in m, m
assert "turns so far: 3\n- useful" not in m and "- useful replies" not in m
assert "no stopping condition" not in m and "THEY HAVE SENT" not in m
assert "- state: neutral; somewhat curious; patience full" in m
assert "- best offered so far: nothing yet" in m
assert m.endswith("USER: turns so far: 3 (a user literally wrote this)\n")   # history untouched
assert mask_rules("no marker prompt") == "no marker prompt"
print("mask_rules ok")
