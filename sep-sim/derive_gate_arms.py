#!/usr/bin/env python3
"""Exact gate-arm counterfactuals by truncating no-gate episodes.

Why exact: the stop gate never feeds back into generation before it fires. The
Planner decodes greedily, Speaker seeds are fixed per (scenario, seed, turn), the
ledger judge is cached by input, and under common random numbers (one R0 draw
per history, shared by all arms of a replicate) the gated episode is identical to
the no-gate episode up to the first step t with p_stop(t) >= threshold. There the
user stops before writing: emitted turns = t-1, coverage/complete = the latched
values after step t-1. If the gate never fires on the steps the no-gate episode
entered, the gated episode IS the no-gate episode. (Verified online with
rollout_stop_sft.py --r0-crn; see the smoke check.)

Outputs one JSONL per derived arm plus the matching no-gate subset, both in
analyze_task2.py format, restricted to the scenarios of --scenarios.
"""
import argparse
import copy
import json
import os


def derive(episode, p_by_t, threshold, arm):
    row = copy.deepcopy(episode)
    for step in row["trace"]:
        step.pop("gate_prompt", None)
        step["p_stop"] = p_by_t.get(step["t"])
    fire = next((s["t"] for s in row["trace"]
                 if s["p_stop"] is not None and s["p_stop"] >= threshold), None)
    row.update(arm=arm, threshold=threshold, derived_from="nogate truncation")
    if fire is None:
        return row
    kept = [s for s in row["trace"] if s["t"] < fire]
    assert all(s["decision"] == "continue" for s in kept), "truncation before a terminal step"
    cov, comp = (kept[-1]["coverage_after"], kept[-1]["complete_after"]) if kept else (0.0, False)
    kept.append({"t": fire, "decision": "stop_gate", "p_stop": p_by_t[fire], "emitted": False,
                 "user": "", "agent": None, "coverage_after": cov, "complete_after": comp})
    row.update(trace=kept, emitted_user_turns=fire - 1, turns=fire - 1, decision_steps=fire,
               end_kind="stop_gate", stop_kind="stop_gate", ended_by_token=True,
               coverage=cov, complete=comp, ledger=None)
    return row


def derive_planner_exit(episode, arm="a1_planner_exit"):
    """A1: the no-gate episode as if the Planner's first session-ending act had ended it
    SILENTLY (task2_episode planner_stop semantics): the user leaves at step t_p without
    writing, so emitted = t_p - 1 and coverage/complete are the values after step t_p - 1.
    Exact because nothing before t_p depends on whether t_p ends the episode."""
    row = copy.deepcopy(episode)
    for step in row["trace"]:
        step.pop("gate_prompt", None)
        step.pop("planner_prompt", None)
    row.update(arm=arm, derived_from="no-gate truncation at first ended_planner (silent exit)")
    tp = next((s["t"] for s in row["trace"] if s.get("ended_planner")), None)
    if tp is None:
        return row
    kept = [s for s in row["trace"] if s["t"] < tp]
    assert all(s["decision"] == "continue" for s in kept), "truncation before a terminal step"
    cov, comp = (kept[-1]["coverage_after"], kept[-1]["complete_after"]) if kept else (0.0, False)
    at = [s for s in row["trace"] if s["t"] == tp][0]
    stop = {"t": tp, "decision": "planner_stop", "planner_stop": True, "emitted": False, "user": "",
            "agent": None, "ended_planner": True, "stop_rule": at.get("stop_rule"),
            "move": at.get("move"), "act": at.get("act"),
            "coverage_after": cov, "complete_after": comp}
    row.update(trace=kept + [stop], emitted_user_turns=tp - 1, turns=tp - 1, decision_steps=tp,
               end_kind="planner_stop", stop_kind="planner_stop", ended_by_token=True,
               coverage=cov, complete=comp, ledger=None)
    return row


def outcome(row):
    return {"prob": 1.0, "emitted_user_turns": row["emitted_user_turns"],
            "decision_steps": row["decision_steps"], "coverage": row["coverage"],
            "complete": bool(row["complete"]), "end_kind": row["end_kind"]}


def derive_hazard(episode, p_by_t, arm):
    """Session-level hazard: stop before writing at step t with probability p_stop(t).
    Returns the exact outcome distribution over the no-gate trajectory and its expectations."""
    row = copy.deepcopy(episode)
    for step in row["trace"]:
        step.pop("gate_prompt", None)
        step["p_stop"] = p_by_t.get(step["t"])
    outcomes, survive = [], 1.0
    cov, comp = 0.0, False
    for step in row["trace"]:
        h = step["p_stop"]
        if h is None:
            raise ValueError("missing hazard at step %d" % step["t"])
        if survive * h > 0:
            outcomes.append({"prob": survive * h, "emitted_user_turns": step["t"] - 1,
                             "decision_steps": step["t"], "coverage": cov, "complete": comp,
                             "end_kind": "stop_gate"})
        survive *= 1 - h
        if step["decision"] == "continue":
            cov, comp = step["coverage_after"], step["complete_after"]
    outcomes.append(dict(outcome(episode), prob=survive))
    assert abs(sum(o["prob"] for o in outcomes) - 1) < 1e-9
    exp = lambda k: sum(o["prob"] * float(o[k]) for o in outcomes)
    row.update(arm=arm, threshold=None, decision_rule="hazard", derived_from="nogate hazard expectation",
               expected=True, outcomes=outcomes, emitted_user_turns=exp("emitted_user_turns"),
               turns=exp("emitted_user_turns"), decision_steps=exp("decision_steps"),
               coverage=exp("coverage"), complete=exp("complete"),
               end_kind="expected", stop_kind="expected", ledger=None)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--scenarios", required=True, help="scenario list defining the analysis set")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--threshold", type=float, action="append", default=[])
    ap.add_argument("--hazard", action="store_true", help="also write the hazard-rule arm")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    keep = {s["conversation_id"] for s in json.load(open(args.scenarios))["scenarios"]}
    episodes = [json.loads(l) for l in open(args.episodes, encoding="utf-8") if l.strip()]
    episodes = [e for e in episodes if e["conversation_id"] in keep]
    scores = {}
    for l in open(args.scores, encoding="utf-8"):
        r = json.loads(l)
        if r["adapter"] == args.adapter:
            scores.setdefault((r["conversation_id"], r["seed"], r["replicate"]), {})[r["t"]] = r["p_stop"]
    os.makedirs(args.out_dir, exist_ok=True)
    missing = [e["conversation_id"] for e in episodes
               if (e["conversation_id"], e["seed"], e.get("replicate", 0)) not in scores]
    if missing:
        raise SystemExit("no scores for %d episodes (fold restriction?): %s" % (len(missing), missing[:3]))
    with open(os.path.join(args.out_dir, "nogate.jsonl"), "w", encoding="utf-8") as f:
        for e in episodes:
            r = copy.deepcopy(e)
            for s in r["trace"]:
                s.pop("gate_prompt", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for thr in args.threshold:
        name = "%s@%g" % (args.adapter, thr)
        with open(os.path.join(args.out_dir, name + ".jsonl"), "w", encoding="utf-8") as f:
            for e in episodes:
                p = scores[(e["conversation_id"], e["seed"], e.get("replicate", 0))]
                f.write(json.dumps(derive(e, p, thr, name), ensure_ascii=False) + "\n")
    if args.hazard:
        name = "%s@hazard" % args.adapter
        with open(os.path.join(args.out_dir, name + ".jsonl"), "w", encoding="utf-8") as f:
            for e in episodes:
                p = scores[(e["conversation_id"], e["seed"], e.get("replicate", 0))]
                f.write(json.dumps(derive_hazard(e, p, name), ensure_ascii=False) + "\n")
    print(json.dumps({"episodes": len(episodes), "scenarios": len({e["conversation_id"] for e in episodes}),
                      "adapter": args.adapter, "thresholds": args.threshold, "hazard": args.hazard,
                      "out": args.out_dir}))


if __name__ == "__main__":
    main()
