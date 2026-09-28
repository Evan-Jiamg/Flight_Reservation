#!/usr/bin/env python3
"""Assertions over the v3 smoke outputs (a0 and a2 episodes, run meta, runstats)."""
import glob
import json
import sys

S = sys.argv[1] if len(sys.argv) > 1 else "/tmp2/mzjiang_usersim/grpo_planner/dev_v3/smoke"
REMOVED = ("- useful replies", "- best offered so far", "- last reply repeated the previous offer",
           "stopping condition", "WHAT THEY STILL WANT", "unhelpful replies before frustration governs",
           "- condition met first", "HOW LONG THEY WRITE", "- band:")


def static(p):
    return p.split("THE CONVERSATION SO FAR", 1)[0]


def check_a0(ep):
    for s in ep["trace"]:
        p = static(s["planner_prompt"])
        assert "- useful replies" in p and "WHAT THEY STILL WANT" in p and "HOW LONG THEY WRITE" in p, \
            "a0 must keep the original prompt"
        assert "GOAL STATUS" not in p and s.get("goal_status") is None
        assert s["planner_fit"]["compacted"] is False
        if s["decision"] == "continue":
            assert "- pending:" in s["block"] and s.get("speaker_fit") is not None
    assert ep["end_kind"] in ("t_max", "empty", "speaker_end"), ep["end_kind"]
    return "a0 ok: %d steps, end %s, ended_planner at %s" % (
        len(ep["trace"]), ep["end_kind"], [s["t"] for s in ep["trace"] if s.get("ended_planner")])


def check_a2(ep):
    tr = ep["trace"]
    assert tr[0]["goal_status"]["status"] == "NOT ASSESSED"
    for s in tr:
        p = static(s["planner_prompt"])
        for bad in REMOVED:
            assert bad not in p, (s["t"], bad)
        assert "GOAL STATUS (assessed from the conversation by a separate model)" in p
        assert "- status: %s" % s["goal_status"]["status"] in p
        if s["t"] > 1:
            assert "raw" in s["goal_status"] and s["goal_status"]["status"] in ("SATISFIED", "PARTIAL", "NOT", "UNKNOWN")
        d = s.get("planner_diag") or {}
        if not s.get("planner_unparsed"):
            # stop decision = the Planner's end_session; no override, no length clamp
            assert s["ended_planner"] == (d.get("end_session_raw") is True or
                                          str(d.get("end_session_raw")).strip().lower() == "true")
            assert d.get("stop_override") in (None, False) and d.get("length_clamped") is False
        if s["decision"] == "continue":
            unmet = s["goal_status"].get("unmet") or []
            exp = ("the whole request (nothing has been answered yet)" if s["t"] == 1 else
                   ("; ".join(unmet) if unmet else "(nothing identified)"))
            assert "- pending: %s" % exp in s["block"], (s["t"], s["block"])
    last = tr[-1]
    if ep["end_kind"] == "planner_stop":
        assert last["decision"] == "planner_stop" and last["user"] == "" and not last["emitted"]
        assert "block" not in last and ep["emitted_user_turns"] == len(tr) - 1
        assert last["ended_planner"]
    return "a2 ok: %d steps, end %s, emitted %d, statuses %s, stop_rule at end %s" % (
        len(tr), ep["end_kind"], ep["emitted_user_turns"], [s["goal_status"]["status"] for s in tr],
        last.get("stop_rule"))


def main():
    for arm, fn in (("a0", check_a0), ("a2", check_a2)):
        rows = [json.loads(l) for p in glob.glob("%s/%s/%s.jsonl" % (S, arm, arm)) for l in open(p)]
        assert rows, "no %s episode" % arm
        for ep in rows:
            print(fn(ep))
        meta = json.load(open(glob.glob("%s/%s/run_meta_%s_rep*.json" % (S, arm, arm))[0]))
        stats = json.load(open(glob.glob("%s/%s/runstats_%s_rep*.json" % (S, arm, arm))[0]))
        assert stats["planner_legacy_tok_calls"] > 0, "TokProxy was not exercised"
        print("  meta gate:", meta["gate"], "| exit:", meta["planner_exit"], "| versions:", meta["versions"])
        print("  stats:", {k: stats[k] for k in ("planner_legacy_tok_calls", "goal_judge_calls",
                                                  "goal_judge_unparsed", "episodes_written")})
    print("SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
