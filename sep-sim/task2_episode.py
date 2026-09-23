"""Terminal semantics for one Task 2 episode, independent of any model or API.

The corrected order at decision step t (protocols/t2_rollout.md step 3, with the
2026-09-23 terminal fix):

  1. gate(t) -> p_stop.  If p_stop >= threshold the user stops BEFORE writing:
     end_kind "stop_gate", no utterance, no R0 call, NOT an emitted turn.
  2. speak(t) -> the selected user candidate and its terminal flags.
  3. An empty candidate ends the episode: end_kind "empty", NOT an emitted turn,
     no R0 call, no ledger update.
  4. A non-empty terminal candidate (Speaker end token, or the Planner end act
     when planner_end is on) IS an emitted turn and stays in the trace, but the
     episode ends there: no R0 reply, no Ledger.update, no satisfaction update.
  5. Only a non-terminal, non-empty utterance calls respond(t, text), which is
     where R0.reply, Ledger.update and the stopping ledger live.
  6. Reaching t_max after a normal exchange gives end_kind "t_max".

`decision_steps` counts every step that was entered (including a gate stop or an
empty draw), so emitted_user_turns <= decision_steps <= t_max.
"""
from __future__ import annotations

END_KINDS = ("stop_gate", "empty", "speaker_end", "planner_end", "t_max")


def run_episode(t_max, gate, speak, respond, threshold=0.5, planner_end=False):
    """gate: callable(t) -> float p_stop, or None for a no-gate arm.
    speak: callable(t) -> dict with keys user, ended_speaker, ended_planner (+extras).
    respond: callable(t, text) -> agent reply (performs R0 + ledger updates)."""
    trace, emitted, end_kind = [], 0, "t_max"
    for t in range(1, t_max + 1):
        p_stop = gate(t) if gate is not None else None
        if p_stop is not None and p_stop >= threshold:
            trace.append({"t": t, "decision": "stop_gate", "p_stop": p_stop,
                          "emitted": False, "user": "", "agent": None})
            end_kind = "stop_gate"
            break
        step = dict(speak(t))
        text = (step.get("user") or "")
        rec = {"t": t, "p_stop": p_stop, **step}
        if not text.strip():
            rec.update(decision="empty", emitted=False, user="", agent=None)
            trace.append(rec)
            end_kind = "empty"
            break
        emitted += 1
        if step.get("ended_speaker") or (planner_end and step.get("ended_planner")):
            kind = "speaker_end" if step.get("ended_speaker") else "planner_end"
            rec.update(decision=kind, emitted=True, agent=None)
            trace.append(rec)
            end_kind = kind
            break
        rec.update(decision="continue", emitted=True, agent=respond(t, text))
        trace.append(rec)
    return {"emitted_user_turns": emitted, "decision_steps": len(trace),
            "end_kind": end_kind, "trace": trace}
