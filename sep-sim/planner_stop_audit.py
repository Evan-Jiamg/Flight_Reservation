#!/usr/bin/env python3
"""Did the frozen JSON Planner issue a session-ending act, and when?

Per logged no-gate episode (PLANNER_END was off, so episodes continued past it):
  t_p       first step with ended_planner (state.ends_session(fields)) -- the Planner's own stop
  K         human session's user-turn count for the same conversation_id
  t_cov     last step at which requirement coverage increased (task-progress saturation)
Counterfactual "planner_end" arm by exact truncation (the rollout is causal and the stop only
truncates): the ending utterance at t_p is emitted (turn counted) but gets no R0 reply and no
ledger update, so emitted = t_p and coverage/complete = values after step t_p-1.
Also: act / move / stop_rule distributions overall and at t_p.
"""
import argparse
import copy
import glob
import json
import os
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def human_turns(corpus):
    k = {}
    for l in open(corpus):
        r = json.loads(l)
        k[r["conversation_id"]] = sum(1 for m in r["chat_messages"] if m["participant_name"].lower() == "user")
    return k


def truncate_at_planner_end(e):
    tr = e["trace"]
    tp = next((s["t"] for s in tr if s.get("ended_planner")), None)
    row = copy.deepcopy(e)
    for s in row["trace"]:
        s.pop("gate_prompt", None)
    row["arm"] = "planner_end"
    if tp is None:
        return row, None
    kept = [s for s in row["trace"] if s["t"] < tp]
    prev = kept[-1] if kept else None
    cov, comp = (prev["coverage_after"], prev["complete_after"]) if prev else (0.0, False)
    end = [s for s in row["trace"] if s["t"] == tp][0]
    if not (end.get("user") or "").strip():      # empty draw at t_p: not a turn (would end as empty anyway)
        return row, tp
    end.update(decision="planner_end", agent=None, coverage_after=cov, complete_after=comp)
    row.update(trace=kept + [end], emitted_user_turns=tp, turns=tp, decision_steps=tp,
               end_kind="planner_end", stop_kind="planner_end", ended_by_token=True,
               coverage=cov, complete=comp, ledger=None)
    return row, tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True, help="name=glob of nogate.jsonl files")
    ap.add_argument("--corpus", default="/home/mzjiang/v5-latency/data.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    K = human_turns(args.corpus)
    os.makedirs(args.out, exist_ok=True)
    for spec in args.glob:
        name, pat = spec.split("=", 1)
        eps = [json.loads(l) for p in sorted(glob.glob(pat)) for l in open(p)]
        base_rows, pe_rows = [], []
        tps, err, lead_cov, acts_at_end, rules_at_end = [], [], [], Counter(), Counter()
        acts_all, moves_all = Counter(), Counter()
        for e in eps:
            for s in e["trace"]:
                if s.get("act") is not None:
                    acts_all[s.get("act")] += 1
                    moves_all[s.get("move")] += 1
            row, tp = truncate_at_planner_end(e)
            b = copy.deepcopy(e)
            for s in b["trace"]:
                s.pop("gate_prompt", None)
            base_rows.append(b)
            pe_rows.append(row)
            k = K[e["conversation_id"]]
            cont = [s for s in e["trace"] if s["decision"] == "continue"]
            last_gain, prev = 0, 0.0
            for s in cont:
                if s["coverage_after"] > prev:
                    last_gain, prev = s["t"], s["coverage_after"]
            if tp is not None:
                tps.append(tp)
                err.append(tp - k)
                lead_cov.append(tp - last_gain)
                s = [x for x in e["trace"] if x["t"] == tp][0]
                acts_at_end[(s.get("move"), s.get("act"))] += 1
                rules_at_end[s.get("stop_rule")] += 1
        n = len(eps)
        summary = {
            "arm": name, "episodes": n,
            "episodes_with_planner_end": len(tps),
            "first_planner_end_turn_hist": dict(sorted(Counter(tps).items())),
            "human_K_hist": dict(sorted(Counter(K[e["conversation_id"]] for e in eps).items())),
            "mean_first_end_turn": sum(tps) / len(tps) if tps else None,
            "mean_signed_error_vs_K": sum(err) / len(err) if err else None,
            "mean_abs_error_vs_K": sum(abs(x) for x in err) / len(err) if err else None,
            "exact_at_K": sum(1 for x in err if x == 0), "within_1_of_K": sum(1 for x in err if abs(x) <= 1),
            "mean_turns_after_last_coverage_gain": sum(lead_cov) / len(lead_cov) if lead_cov else None,
            "move_act_at_end": {"%s/%s" % k: v for k, v in acts_at_end.most_common(8)},
            "stop_rule_at_end": dict(rules_at_end.most_common(8)),
            "act_overall_top": dict(acts_all.most_common(10)),
            "move_overall_top": dict(moves_all.most_common(10)),
        }
        print(json.dumps(summary))
        bp, pp = os.path.join(args.out, name + "_nogate.jsonl"), os.path.join(args.out, name + "_planner_end.jsonl")
        for path, rows in ((bp, base_rows), (pp, pe_rows)):
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        res = json.loads(subprocess.run([sys.executable, os.path.join(HERE, "analyze_task2.py"), "--base", bp,
                                         "--new", pp, "--corpus", args.corpus], capture_output=True, text=True).stdout)
        json.dump({"summary": summary, "paired": res}, open(os.path.join(args.out, name + "_result.json"), "w"), indent=1)
        for m in ("emitted_user_turns", "abs_turn_error", "turn_error", "coverage", "complete", "end_planner_end"):
            p = res["paired"][m]
            print("  %-20s nogate %.3f planner_end %.3f  diff %+.3f  95%% %s" % (
                m, res["base"][m + "_mean"], res["new"][m + "_mean"], p["mean_difference_new_minus_base"],
                [round(x, 3) for x in p["scenario_bootstrap_95"]]))


if __name__ == "__main__":
    main()
