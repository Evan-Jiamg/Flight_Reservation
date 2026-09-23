#!/usr/bin/env python3
"""Stage C readout: smoke equivalence, offline gate scoring, derived arms, paired analysis.

Steps (each skipped if its output already exists):
  1. smoke: score the smoke no-gate episode with the smoke adapter, derive @0.2 and
     compare with the ONLINE CRN gated episode (turns, end kind, coverage, complete,
     utterances, per-step p_stop).
  2. concatenate replicate no-gate halves; score PRISM (fold -1) and each fold's
     selected Stage B adapter (only on that fold's inner scenarios).
  3. for fold x side: derive @0.5, hazard, and the threshold grid; analyze each vs
     no-gate with the human-turn comparison; write one summary JSON + a text table.
"""
import argparse
import json
import os
import subprocess
import sys

G = "/tmp2/mzjiang_usersim/grpo_planner"
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CORPUS = "/home/mzjiang/v5-latency/data.jsonl"
SIDES = {"inner_train": "task2_train_scenarios/fold%d_inner_train_scenarios.json",
         "inner_validation": "task2_val_scenarios/fold%d_inner_validation_scenarios.json"}
GRID = [0.1, 0.2, 0.3, 0.4, 0.5]


def run(*args, out=None):
    r = subprocess.run([PY, *args], capture_output=True, text=True, cwd=HERE)
    if r.returncode:
        raise SystemExit("FAILED %s\n%s" % (args[:2], r.stderr[-3000:]))
    if out:
        open(out, "w").write(r.stdout)
    return r.stdout


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def smoke(stage):
    S = os.path.join(stage, "smoke_crn")
    ad = "%s/trec_inner_fold1_7b_v1/epoch2" % G
    sc = os.path.join(S, "scores.jsonl")
    if not os.path.exists(sc):
        run("score_logged_gates.py", "--episodes", os.path.join(S, "nogate.jsonl"),
            "--adapter", "f1ep2=%s@1" % ad, "--out", sc, "--gpu", "1")
    run("derive_gate_arms.py", "--episodes", os.path.join(S, "nogate.jsonl"), "--scores", sc,
        "--scenarios", "%s/task2_train_scenarios/fold1_inner_train_scenarios.json" % G,
        "--adapter", "f1ep2", "--threshold", "0.2", "--out-dir", os.path.join(S, "derived"))
    online = {(r["conversation_id"], r["seed"]): r for r in rows(os.path.join(S, "f1ep2_t02.jsonl"))}
    derived = {(r["conversation_id"], r["seed"]): r
               for r in rows(os.path.join(S, "derived", "f1ep2@0.2.jsonl"))}
    report = []
    for k, on in online.items():
        d = derived[k]
        same = {f: on[f] == d[f] for f in ("emitted_user_turns", "decision_steps", "end_kind",
                                             "coverage", "complete")}
        same["utterances"] = [s["user"] for s in on["trace"]] == [s["user"] for s in d["trace"]]
        dp = [abs((a.get("p_stop") or 0) - (b.get("p_stop") or 0))
              for a, b in zip(on["trace"], d["trace"]) if a.get("p_stop") is not None]
        report.append({"episode": k, "online": {f: on[f] for f in ("emitted_user_turns", "end_kind",
                                                                    "coverage", "complete")},
                       "match": same, "max_abs_p_stop_diff": max(dp) if dp else None})
    json.dump(report, open(os.path.join(S, "equivalence.json"), "w"), indent=1)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=G + "/stageC_v1")
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()
    out = {"replicate": args.replicate}
    if not args.skip_smoke:
        out["smoke_equivalence"] = smoke(args.stage)
        print("SMOKE", json.dumps(out["smoke_equivalence"]), flush=True)
    rep = os.path.join(args.stage, "rep%d" % args.replicate)
    os.makedirs(rep, exist_ok=True)
    nog = os.path.join(rep, "nogate.jsonl")
    if not os.path.exists(nog):
        eps = rows(os.path.join(args.stage, "rep%d_half_a" % args.replicate, "nogate.jsonl")) + \
              rows(os.path.join(args.stage, "rep%d_half_b" % args.replicate, "nogate.jsonl"))
        with open(nog, "w", encoding="utf-8") as f:
            for e in eps:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    adapters = {"prism": ("%s/prism_sft_7b_v1/best" % G, -1)}
    for fold in (0, 1, 2):
        adapters["f%dsel" % fold] = ("%s/trec_inner_fold%d_7b_v1/epoch2" % (G, fold), fold)
    sc = os.path.join(rep, "scores.jsonl")
    if not os.path.exists(sc):
        cmd = ["score_logged_gates.py", "--episodes", nog, "--out", sc, "--gpu", "1"]
        for name, (path, fold) in adapters.items():
            cmd += ["--adapter", "%s=%s@%d" % (name, path, fold)]
        run(*cmd)
    table = {}
    for fold in (0, 1, 2):
        for side, pattern in SIDES.items():
            scen = os.path.join(G, pattern % fold)
            if not json.load(open(scen))["scenarios"]:
                continue
            for name in ("prism", "f%dsel" % fold):
                d = os.path.join(rep, "derived", "fold%d_%s" % (fold, side), name)
                cmd = ["derive_gate_arms.py", "--episodes", nog, "--scores", sc, "--scenarios", scen,
                       "--adapter", name, "--hazard", "--out-dir", d]
                for t in GRID:
                    cmd += ["--threshold", str(t)]
                run(*cmd)
                arms = ["%s@hazard" % name] + ["%s@%g" % (name, t) for t in GRID]
                for arm in arms:
                    res = json.loads(run("analyze_task2.py", "--base", os.path.join(d, "nogate.jsonl"),
                                         "--new", os.path.join(d, arm + ".jsonl"), "--corpus", CORPUS))
                    table["fold%d|%s|%s" % (fold, side, arm)] = res
    json.dump(table, open(os.path.join(rep, "stageC_summary.json"), "w"), indent=1)
    lines = []
    fmt = "%-34s %4s %6s %6s %6s %6s %6s | d_turns %-22s d_cov %-24s d_comp %-22s"
    for key, res in table.items():
        b, n, p = res["base"], res["new"], res["paired"]
        ci = lambda m: "%+.3f[%+.3f,%+.3f]" % (p[m]["mean_difference_new_minus_base"],
                                               *p[m]["scenario_bootstrap_95"])
        lines.append(fmt % (key, p["n_scenarios"], "%.2f" % n["emitted_user_turns_mean"],
                            "%.2f" % n["human_turns_mean"], "%.2f" % n["abs_turn_error_mean"],
                            "%.3f" % n["coverage_mean"], "%.3f" % n["complete_mean"],
                            ci("emitted_user_turns"), ci("coverage"), ci("complete")))
        if key.endswith("@hazard"):
            lines.append(fmt % ("   (nogate)", p["n_scenarios"], "%.2f" % b["emitted_user_turns_mean"],
                                "%.2f" % b["human_turns_mean"], "%.2f" % b["abs_turn_error_mean"],
                                "%.3f" % b["coverage_mean"], "%.3f" % b["complete_mean"], "", "", ""))
    hdr = "%-34s %4s %6s %6s %6s %6s %6s" % ("fold|side|arm", "scen", "turns", "human", "|err|", "cov", "comp")
    text = hdr + "\n" + "\n".join(lines)
    open(os.path.join(rep, "stageC_table.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
