"""Synthetic smoke tests for analyze_task2.py and stop_timing.py (no GPU, no API)."""
import json
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def ep(cid, seed, arm, emitted, kind, cov, comp):
    tr = [{"t": i + 1, "user": "u", "emitted": True} for i in range(emitted)]
    if kind in ("stop_gate", "empty"):
        tr.append({"t": emitted + 1, "user": "", "emitted": False})
    return {"conversation_id": cid, "seed": seed, "arm": arm, "emitted_user_turns": emitted,
            "decision_steps": len(tr), "end_kind": kind, "coverage": cov, "complete": comp,
            "trace": tr}


def run(*a):
    r = subprocess.run([sys.executable, *a], capture_output=True, text=True, cwd=HERE)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def main():
    d = tempfile.mkdtemp()
    rng = random.Random(0)
    for arm in ("a", "b"):
        with open(os.path.join(d, arm + ".jsonl"), "w") as f:
            for c in range(6):
                for s in (0, 1):
                    k = rng.choice(["stop_gate", "t_max", "speaker_end", "empty"])
                    f.write(json.dumps(ep("c%d" % c, s, arm, rng.randint(1, 9), k, rng.random(),
                                          rng.random() < .2)) + "\n")
    o = run("analyze_task2.py", "--base", os.path.join(d, "a.jsonl"), "--new", os.path.join(d, "b.jsonl"))
    assert o["paired"]["n_scenarios"] == 6 and o["base"]["episodes"] == 12
    assert abs(sum(v for k, v in o["base"].items() if k.startswith("end_")) - 1) < 1e-9
    # identical arms => zero difference
    z = run("analyze_task2.py", "--base", os.path.join(d, "a.jsonl"), "--new", os.path.join(d, "a.jsonl"))
    assert all(v["mean_difference_new_minus_base"] == 0 for k, v in z["paired"].items() if isinstance(v, dict))
    print("analyze_task2 ok", o["paired"]["emitted_user_turns"])

    # W1 to human: corpus where every scenario's human K is known
    corpus = os.path.join(d, "corpus.jsonl")
    with open(corpus, "w") as f:
        for c in range(6):
            msgs = [{"participant_name": "user"}] * 4 + [{"participant_name": "agent"}] * 4
            f.write(json.dumps({"conversation_id": "c%d" % c, "chat_messages": msgs}) + "\n")
    with open(os.path.join(d, "k4.jsonl"), "w") as f:      # every episode emits exactly K=4 turns
        for c in range(6):
            for s in (0, 1):
                f.write(json.dumps(ep("c%d" % c, s, "k4", 4, "planner_stop", .5, False)) + "\n")
    with open(os.path.join(d, "k6.jsonl"), "w") as f:      # every episode emits 6 turns
        for c in range(6):
            for s in (0, 1):
                f.write(json.dumps(ep("c%d" % c, s, "k6", 6, "t_max", .5, False)) + "\n")
    w = run("analyze_task2.py", "--base", os.path.join(d, "k6.jsonl"), "--new", os.path.join(d, "k4.jsonl"),
            "--corpus", corpus)
    assert w["new"]["w1_to_human"] == 0.0 and abs(w["base"]["w1_to_human"] - 2.0) < 1e-9, w["base"]
    assert abs(w["paired"]["w1_to_human"]["difference_new_minus_base"] + 2.0) < 1e-9
    assert w["new"]["end_planner_stop_mean"] == 1.0
    print("w1_to_human ok")

    with open(os.path.join(d, "pred.jsonl"), "w") as f:
        # session r0 stops early at t=1; r1 exact; r2 never
        for rid, ps in (("r0", [.9, .2, .8]), ("r1", [.1, .2, .7]), ("r2", [.1, .1, .1])):
            for t, p in enumerate(ps, 1):
                f.write(json.dumps({"record_id": rid, "turn_index": t, "target_stop": t == 3,
                                    "p_stop": p}) + "\n")
    t = run("stop_timing.py", os.path.join(d, "pred.jsonl"), "--bootstrap", "200")
    p = t["by_threshold"]["0.5"]["point"]
    assert abs(p["early"] - 1 / 3) < 1e-9 and abs(p["exact"] - 1 / 3) < 1e-9
    assert abs(p["not_by_k1_censored"] - 1 / 3) < 1e-9 and p["mean_turns_early"] == 2
    assert abs(p["false_stop"] - 1 / 6) < 1e-9 and abs(p["k1_end"] - 2 / 3) < 1e-9
    print("stop_timing ok", p)


if __name__ == "__main__":
    main()
