"""Controlled episodes for the four end kinds plus t_max (no GPU, no API)."""
from task2_episode import run_episode


class Env:
    def __init__(self):
        self.r0_calls, self.ledger_updates = [], []

    def respond(self, t, text):
        self.r0_calls.append((t, text))
        self.ledger_updates.append(t)
        return "reply %d" % t


def speaker(script):
    def speak(t):
        text, end_s, end_p = script[t - 1]
        return {"user": text, "ended_speaker": end_s, "ended_planner": end_p}
    return speak


def gate_at(k, p=0.9):
    return lambda t: p if t == k else 0.1


def test_stop_gate_before_utterance():
    env = Env()
    ep = run_episode(10, gate_at(3), speaker([("u", False, False)] * 10), env.respond)
    assert ep["end_kind"] == "stop_gate"
    assert ep["emitted_user_turns"] == 2 and ep["decision_steps"] == 3
    assert [t for t, _ in env.r0_calls] == [1, 2]
    assert ep["trace"][-1]["user"] == "" and ep["trace"][-1]["agent"] is None


def test_empty_not_a_turn():
    env = Env()
    ep = run_episode(10, None, speaker([("a", False, False), ("  ", False, False)]), env.respond)
    assert ep["end_kind"] == "empty"
    assert ep["emitted_user_turns"] == 1 and ep["decision_steps"] == 2
    assert env.ledger_updates == [1]


def test_speaker_end_counts_turn_but_no_r0_or_ledger():
    env = Env()
    ep = run_episode(10, gate_at(99), speaker([("a", False, False), ("thanks, bye", True, False)]),
                     env.respond)
    assert ep["end_kind"] == "speaker_end"
    assert ep["emitted_user_turns"] == 2 and ep["decision_steps"] == 2
    assert env.r0_calls == [(1, "a")] and env.ledger_updates == [1]
    assert ep["trace"][-1]["user"] == "thanks, bye" and ep["trace"][-1]["agent"] is None


def test_planner_end_only_when_enabled():
    env = Env()
    script = [("a", False, True), ("b", False, False)] + [("c", False, False)] * 8
    off = run_episode(10, None, speaker(script), env.respond, planner_end=False)
    assert off["end_kind"] == "t_max" and off["emitted_user_turns"] == 10
    env2 = Env()
    on = run_episode(10, None, speaker(script), env2.respond, planner_end=True)
    assert on["end_kind"] == "planner_end" and on["emitted_user_turns"] == 1
    assert env2.r0_calls == []


def test_t_max():
    env = Env()
    ep = run_episode(10, gate_at(99), speaker([("u", False, False)] * 10), env.respond)
    assert ep["end_kind"] == "t_max" and ep["emitted_user_turns"] == 10 == ep["decision_steps"]
    assert len(env.r0_calls) == 10


def test_gate_first_step_and_threshold_boundary():
    env = Env()
    ep = run_episode(10, lambda t: 0.5, speaker([("u", False, False)] * 10), env.respond, threshold=0.5)
    assert ep["end_kind"] == "stop_gate" and ep["emitted_user_turns"] == 0 and env.r0_calls == []
    ep2 = run_episode(10, lambda t: 0.49, speaker([("u", False, False)] * 10), Env().respond, threshold=0.5)
    assert ep2["end_kind"] == "t_max"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
