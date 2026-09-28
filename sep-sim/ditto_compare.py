#!/usr/bin/env python3
"""Ditto vs UserLM Speaker, same corrected runner, same scenarios x seeds, replicate 0.

1) Planner parity across the two library stacks: the Planner's turn-1 block is produced
   before any Speaker or R0 output, so it must be identical if the stack change did not
   alter the Planner. It is read from the step-2 gate prompt ("THE STATE YOU WROTE LAST TURN").
2) Paired Task 2 comparison (analyze_task2.py, scenario unit) + utterance-level descriptives:
   words per utterance, first-turn duplicate rate across seeds, exact repeat of an earlier turn,
   and a crude off-task marker rate (instruction-like openers seen in UserLM traces).
"""
import glob
import json
import os
import re
import subprocess
import sys

R = "/tmp2/mzjiang_usersim/grpo_planner/stageC_v1"
HERE = os.path.dirname(os.path.abspath(__file__))
OFF = re.compile(r"^(write|rewrite|what should the agent|system:|you are )", re.I)


def load(pattern):
    eps = [json.loads(l) for f in sorted(glob.glob(pattern)) for l in open(f)]
    return {(e["conversation_id"], e["seed"]): e for e in eps}


def state_block(prompt):
    m = re.search(r"THE STATE YOU WROTE LAST TURN\n(.*?)\n\n", prompt, re.S)
    return m.group(1) if m else None


def describe(eps):
    utt = [s["user"] for e in eps.values() for s in e["trace"] if (s.get("user") or "").strip()]
    words = [len(u.split()) for u in utt]
    first = {}
    for (cid, seed), e in eps.items():
        u = [s["user"] for s in e["trace"] if (s.get("user") or "").strip()]
        first.setdefault(cid, []).append(u[0].strip().lower() if u else "")
    dup = sum(1 for v in first.values() if len(v) == 2 and v[0] == v[1]) / len(first)
    rep = 0
    for e in eps.values():
        seen = set()
        for s in e["trace"]:
            u = (s.get("user") or "").strip().lower()
            if u:
                rep += u in seen
                seen.add(u)
    return {"utterances": len(utt), "words_mean": sum(words) / len(words),
            "first_turn_identical_across_seeds": dup, "exact_repeat_rate": rep / len(utt),
            "offtask_opener_rate": sum(bool(OFF.match(u.strip())) for u in utt) / len(utt)}


def main():
    u = load(R + "/rep0_shard*/nogate.jsonl")
    d = load(R + "/rep0_ditto_shard*/nogate.jsonl")
    assert len(u) == len(d) == 68 and u.keys() == d.keys(), (len(u), len(d))
    same = diff = missing = 0
    for k in u:
        pu = [s for s in u[k]["trace"] if s["t"] == 2]
        pd = [s for s in d[k]["trace"] if s["t"] == 2]
        if not pu or not pd:
            missing += 1
            continue
        a, b = state_block(pu[0]["gate_prompt"]), state_block(pd[0]["gate_prompt"])
        if a is None or b is None:
            missing += 1
        elif a == b:
            same += 1
        else:
            diff += 1
    print(json.dumps({"planner_turn1_parity": {"identical": same, "different": diff, "not_comparable": missing}}))
    for name, eps in (("userlm", u), ("ditto", d)):
        print(json.dumps({name: describe(eps)}))
    out = R + "/speaker_compare_rep0"
    os.makedirs(out, exist_ok=True)
    for name, eps in (("userlm", u), ("ditto", d)):
        with open(os.path.join(out, name + ".jsonl"), "w") as f:
            for e in eps.values():
                r = dict(e)
                r["trace"] = [{k: v for k, v in s.items() if k != "gate_prompt"} for s in e["trace"]]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    res = subprocess.run([sys.executable, os.path.join(HERE, "analyze_task2.py"), "--base",
                          os.path.join(out, "userlm.jsonl"), "--new", os.path.join(out, "ditto.jsonl"),
                          "--corpus", "/home/mzjiang/v5-latency/data.jsonl"], capture_output=True, text=True)
    open(os.path.join(out, "paired.json"), "w").write(res.stdout)
    r = json.loads(res.stdout)
    keys = ("emitted_user_turns", "abs_turn_error", "coverage", "complete", "end_t_max", "end_speaker_end")
    for k in keys:
        p = r["paired"][k]
        print("%-20s userlm %.3f ditto %.3f  diff %+.3f  SE %.3f  95%% %s" % (
            k, r["base"][k + "_mean"], r["new"][k + "_mean"], p["mean_difference_new_minus_base"],
            p["scenario_se"], [round(x, 3) for x in p["scenario_bootstrap_95"]]))
    print("scenarios", r["paired"]["n_scenarios"], "episodes", r["paired"]["n_episodes"])


if __name__ == "__main__":
    main()
