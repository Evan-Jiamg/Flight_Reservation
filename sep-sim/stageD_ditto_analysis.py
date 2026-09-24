#!/usr/bin/env python3
"""Stop policies on logged Ditto no-gate trajectories (exact truncation; no new rollouts).

Arms per fold x side (inner_train / inner_validation):
  prism      Stage A adapter (not TREC-trained)       hazard + thresholds
  fXsel      fold X selected Stage B adapter          hazard + thresholds
  fXturn     explicit content-free per-turn hazard from fold X inner-train human lengths
Compared with the Ditto no-gate episodes, scenario-level paired, incl. |emitted - K_human|.
"""
import argparse
import glob
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
GRID = [0.1, 0.2, 0.3, 0.5]


def run(*a):
    r = subprocess.run([PY, *a], capture_output=True, text=True, cwd=HERE)
    if r.returncode:
        raise SystemExit("FAILED %s\n%s" % (a[:2], r.stderr[-3000:]))
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    S = G + "/stageC_v1"
    out = os.path.join(S, "ditto_rep%d" % args.replicate)
    os.makedirs(out, exist_ok=True)
    nog = os.path.join(out, "nogate.jsonl")
    if not os.path.exists(nog):
        eps = [json.loads(l) for p in sorted(glob.glob(os.path.join(S, "rep%d_ditto_shard*" % args.replicate,
                                                                   "nogate.jsonl"))) for l in open(p)]
        keys = {(e["conversation_id"], e["seed"]) for e in eps}
        if len(eps) != 68 or len(keys) != 68:
            raise SystemExit("Ditto replicate %d incomplete: %d episodes" % (args.replicate, len(eps)))
        with open(nog, "w", encoding="utf-8") as f:
            for e in eps:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    sc = os.path.join(out, "scores.jsonl")
    if not os.path.exists(sc):
        cmd = ["score_logged_gates.py", "--episodes", nog, "--out", sc, "--gpu", str(args.gpu),
               "--adapter", "prism=%s/prism_sft_7b_v1/best@-1" % G]
        for f in (0, 1, 2):
            cmd += ["--adapter", "f%dsel=%s/trec_inner_fold%d_7b_v1/epoch2@%d" % (f, G, f, f)]
        run(*cmd)
    ts = os.path.join(out, "turn_scores.jsonl")
    if not os.path.exists(ts):
        run("make_turn_hazard_scores.py", "--nested-dir", G + "/nested", "--episodes", nog, "--out", ts)
    allsc = os.path.join(out, "all_scores.jsonl")
    with open(allsc, "w") as f:
        for p in (sc, ts):
            f.write(open(p).read())
    table = {}
    for fold in (0, 1, 2):
        for side, pat in SIDES.items():
            scen = os.path.join(G, pat % fold)
            if not json.load(open(scen))["scenarios"]:
                continue
            for name in ("prism", "f%dsel" % fold, "f%dturn" % fold):
                d = os.path.join(out, "derived", "fold%d_%s" % (fold, side), name)
                cmd = ["derive_gate_arms.py", "--episodes", nog, "--scores", allsc, "--scenarios", scen,
                       "--adapter", name, "--hazard", "--out-dir", d]
                for t in GRID:
                    cmd += ["--threshold", str(t)]
                run(*cmd)
                for arm in ["%s@hazard" % name] + ["%s@%g" % (name, t) for t in GRID]:
                    table["fold%d|%s|%s" % (fold, side, arm)] = json.loads(run(
                        "analyze_task2.py", "--base", os.path.join(d, "nogate.jsonl"),
                        "--new", os.path.join(d, arm + ".jsonl"), "--corpus", CORPUS))
    json.dump(table, open(os.path.join(out, "summary.json"), "w"), indent=1)
    fmt = "%-36s %4s %6s %6s %6s %6s %6s | d_turns %-22s d_cov %-24s d_comp %s"
    lines = ["%-36s %4s %6s %6s %6s %6s %6s" % ("fold|side|arm", "scen", "turns", "human", "|err|", "cov", "comp")]
    for key, res in table.items():
        n, p = res["new"], res["paired"]
        ci = lambda m: "%+.3f[%+.3f,%+.3f]" % (p[m]["mean_difference_new_minus_base"], *p[m]["scenario_bootstrap_95"])
        if key.endswith("prism@hazard"):
            b = res["base"]
            lines.append(fmt % (key.rsplit("|", 1)[0] + "|(ditto nogate)", p["n_scenarios"],
                                "%.2f" % b["emitted_user_turns_mean"], "%.2f" % b["human_turns_mean"],
                                "%.2f" % b["abs_turn_error_mean"], "%.3f" % b["coverage_mean"],
                                "%.3f" % b["complete_mean"], "", "", ""))
        lines.append(fmt % (key, p["n_scenarios"], "%.2f" % n["emitted_user_turns_mean"],
                            "%.2f" % n["human_turns_mean"], "%.2f" % n["abs_turn_error_mean"],
                            "%.3f" % n["coverage_mean"], "%.3f" % n["complete_mean"],
                            ci("emitted_user_turns"), ci("coverage"), ci("complete")))
    text = "\n".join(lines)
    open(os.path.join(out, "table.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
