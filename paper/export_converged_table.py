#!/usr/bin/env python3
"""
export_converged_table.py -- one row per run, at its converged state.

540 rows. Identifiers + the convergence summary + all 18 metrics.csv columns
read from the run's final step (which the dynamic stopping rule places at
t_conv + post_window). Nothing is recomputed; this is a join, not an analysis.

  python analysis/export_converged_table.py -o /tmp/m1_converged.csv
"""

import argparse
import csv
import glob
import json
import os

GRID = "/mnt/NewSSD/CS_project/neil/hcog_experiments/M-1_main-grid/phi4"

ID_COLS = ["topic", "network", "alpha", "seed"]
CONV_COLS = ["t_conv", "steps_run", "attractor", "period", "C_plateau",
             "hit_T_max", "n_llm", "n_numeric", "parse_failures", "llm_calls"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="/tmp/m1_converged.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(GRID, "*/*/alpha_*/seed_*/convergence.json")))
    rows, metric_cols = [], None
    skipped = []

    for cp in paths:
        rd = os.path.dirname(cp)
        d = json.load(open(cp))
        mp = os.path.join(rd, "metrics.csv")
        if not os.path.exists(mp):
            skipped.append((os.path.relpath(rd, GRID), "no metrics.csv"))
            continue
        with open(mp) as f:
            recs = list(csv.DictReader(f))
        if not recs:
            skipped.append((os.path.relpath(rd, GRID), "empty metrics.csv"))
            continue
        last = recs[-1]
        if metric_cols is None:
            metric_cols = list(last.keys())
        elif list(last.keys()) != metric_cols:
            skipped.append((os.path.relpath(rd, GRID), "column mismatch"))
            continue

        r = {
            "topic": d.get("topic"),
            "network": d.get("network"),
            "alpha": d.get("alpha"),
            "seed": int(os.path.basename(rd).split("_")[1]),
        }
        for c in CONV_COLS:
            r[c] = d.get(c)
        # the metrics columns keep their own names; `step` is the final step
        for c in metric_cols:
            r[c] = last[c]
        rows.append(r)

    header = ID_COLS + CONV_COLS + metric_cols
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {args.out}")
    print(f"  {len(rows)} rows x {len(header)} columns")
    print(f"  identifiers : {', '.join(ID_COLS)}")
    print(f"  convergence : {', '.join(CONV_COLS)}")
    print(f"  metrics ({len(metric_cols)}): {', '.join(metric_cols)}")
    if skipped:
        print(f"  SKIPPED {len(skipped)}:")
        for p, why in skipped[:10]:
            print(f"    {p}  -- {why}")


if __name__ == "__main__":
    main()
