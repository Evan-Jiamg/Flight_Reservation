"""Truncation must equal the online episode produced by run_episode under shared randomness."""
from derive_gate_arms import derive
from task2_episode import run_episode


def make(script, cov, gate_p=None, thr=0.5):
    calls = []

    def speak(t):
        text, end = script[t - 1]
        return {"user": text, "ended_speaker": end, "ended_planner": False, "gate_prompt": "P%d" % t}

    state = {"cov": 0.0, "comp": False}

    def respond(t, text):
        calls.append(t)
        state["cov"], state["comp"] = cov[t - 1]
        return "r%d" % t

    ep = run_episode(10, (lambda t: gate_p[t]) if gate_p else None, speak, respond, threshold=thr)
    last = (0.0, False)
    for s in ep["trace"]:
        if s["decision"] == "continue":
            last = cov[s["t"] - 1]
        s["coverage_after"], s["complete_after"] = last
    ep.update(arm="nogate" if gate_p is None else "gate", conversation_id="c", seed=0,
              coverage=last[0], complete=last[1])
    return ep


def check(script, cov, gate_p, thr=0.5):
    base = make(script, cov)
    online = make(script, cov, gate_p, thr)
    d = derive(base, gate_p, thr, "gate")
    for k in ("emitted_user_turns", "decision_steps", "end_kind", "coverage", "complete"):
        assert d[k] == online[k], (k, d[k], online[k])
    assert [s["t"] for s in d["trace"]] == [s["t"] for s in online["trace"]]
    return d


script = [("u%d" % i, False) for i in range(1, 10)] + [("bye", True)]
cov = [(0.1 * i, i >= 6) for i in range(1, 11)]
# fires at step 4
d = check(script, cov, {t: (0.9 if t == 4 else 0.1) for t in range(1, 11)})
assert d["end_kind"] == "stop_gate" and d["emitted_user_turns"] == 3 and abs(d["coverage"] - 0.3) < 1e-9
# fires at step 1
d = check(script, cov, {t: 0.8 for t in range(1, 11)})
assert d["emitted_user_turns"] == 0 and d["coverage"] == 0.0 and d["complete"] is False
# never fires -> speaker_end at 10, identical to no-gate
d = check(script, cov, {t: 0.2 for t in range(1, 11)})
assert d["end_kind"] == "speaker_end" and d["emitted_user_turns"] == 10
# fires exactly at the step where no-gate would emit its terminal utterance
d = check(script, cov, {t: (0.7 if t == 10 else 0.0) for t in range(1, 11)})
assert d["end_kind"] == "stop_gate" and d["emitted_user_turns"] == 9 and d["complete"] is True
# early empty draw in no-gate at step 3, gate would fire at 5 -> never reached
script2 = [("a", False), ("b", False), ("", False)] + [("x", False)] * 7
d = check(script2, cov, {t: (0.9 if t == 5 else 0.0) for t in range(1, 11)})
assert d["end_kind"] == "empty" and d["emitted_user_turns"] == 2
print("derive_gate_arms == online run_episode on 5 cases: ok")

# ---- hazard rule: exact expectation == Monte Carlo of the online Bernoulli gate ----
import random  # noqa: E402
from derive_gate_arms import derive_hazard  # noqa: E402

h = {t: 0.05 * t for t in range(1, 11)}
base = make(script, cov)
hz = derive_hazard(base, h, "hz")
rng = random.Random(1)
n, acc = 20000, {"emitted_user_turns": 0.0, "coverage": 0.0, "complete": 0.0, "stop": 0.0}
for _ in range(n):
    draws = {t: (1.0 if rng.random() < h[t] else 0.0) for t in range(1, 11)}
    ep = make(script, cov, draws, 0.5)
    acc["emitted_user_turns"] += ep["emitted_user_turns"]
    acc["coverage"] += ep["coverage"]
    acc["complete"] += ep["complete"]
    acc["stop"] += ep["end_kind"] == "stop_gate"
for k in ("emitted_user_turns", "coverage", "complete"):
    assert abs(acc[k] / n - hz[k]) < 0.03 * max(1, hz[k]), (k, acc[k] / n, hz[k])
p_stop = sum(o["prob"] for o in hz["outcomes"] if o["end_kind"] == "stop_gate")
assert abs(acc["stop"] / n - p_stop) < 0.01
zero = derive_hazard(base, {t: 0.0 for t in range(1, 11)}, "z")
assert zero["emitted_user_turns"] == base["emitted_user_turns"] and zero["coverage"] == base["coverage"]
print("hazard expectation == Monte Carlo online gate: ok", round(hz["emitted_user_turns"], 3))
