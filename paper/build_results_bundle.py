#!/usr/bin/env python3
"""build_results_bundle.py -- the analysis-ready extract of a run grid.

The raw grid is 2.6 GB and lives outside the repository. 98% of that is LLM
prose (`opinions`, `reasonings`, `long_memory`, `short_memory` inside
agents_data.json, plus agents_interaction_data.json), which no figure or table
reads. Re-running the experiment is the reproducibility path for those; the
repository instead carries what the analysis actually consumes, at ~10 MB.

Kept per run, unchanged, so existing analysis code works by pointing GRID here:

    metrics.csv            the 18 per-step metrics
    convergence.json       t_conv, attractor, plateau height
    poa_components.csv     disagreement / conformity split
    agent_assignment.json  which agents are Type-L

Dropped:

    model_overview.json    verified byte-equivalent to metrics.csv, 25/25 runs
                           sampled -- the same table in JSONL form, 10.6 MB
    edges_per_step.json    56 MB; every graph metric computed from it is
                           already in metrics.csv, and the one figure that
                           needed the final graph is served by the derived
                           table below
    agents_data.json       2204 MB, of which beliefs is 0.12 MB per run
    agents_interaction_data.json  342 MB

Derived once, because LLM runs are not deterministic and re-running would not
reproduce these numbers:

    neighbor_gap.csv       per agent: |z_i - mean z over the neighbours it kept|
                           at the converged state, with its type. This is the
                           whole input to fig_neighbor_gap, ~2 MB instead of
                           the 2.3 GB it was read from.

  python3 build_results_bundle.py --grid /mnt/.../M-1_main-grid/phi4 \
                                  --out ../results/M-1_main-grid/phi4
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys

import numpy as np

COPY = ["metrics.csv", "convergence.json", "poa_components.csv",
        "agent_assignment.json"]


def neighbor_rows(rd, meta):
    """One row per agent: its gap to the neighbours it kept at the last step."""
    try:
        agents = json.load(open(os.path.join(rd, "agents_data.json")))
        edges = json.load(open(os.path.join(rd, "edges_per_step.json")))
        assign = json.load(open(os.path.join(rd, "agent_assignment.json")))
    except (OSError, json.JSONDecodeError):
        return []

    n = len(agents)
    z = np.array([agents[str(i)]["beliefs"][-1] for i in range(n)])
    llm = set(assign.get("llm_agents", []))
    A = np.zeros((n, n))
    for (i, j) in edges[-1]:
        A[i, j] = 1.0
    deg = A.sum(axis=1)
    nbr = np.divide(A @ z, deg, out=np.full(n, np.nan), where=deg > 0)

    out = []
    for i in range(n):
        if deg[i] == 0:                    # isolated: no gap to report
            continue
        out.append({**meta, "agent": i,
                    "type": "Type-L" if i in llm else "Type-C",
                    "z": round(float(z[i]), 6),
                    "z_neighbors": round(float(nbr[i]), 6),
                    "gap": round(float(abs(z[i] - nbr[i])), 6),
                    "degree": int(deg[i])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="raw grid root (…/phi4)")
    ap.add_argument("--out", required=True, help="bundle root to write")
    ap.add_argument("--skip-derived", action="store_true",
                    help="only copy; do not re-read the 2.3 GB agents_data")
    args = ap.parse_args()

    runs = sorted(glob.glob(os.path.join(args.grid, "*/*/alpha_*/seed_*")))
    if not runs:
        sys.exit("no runs under " + args.grid)
    print(f"{len(runs)} runs under {args.grid}")

    os.makedirs(args.out, exist_ok=True)
    copied = skipped = 0
    gaps = []

    for k, rd in enumerate(runs, 1):
        rel = os.path.relpath(rd, args.grid)
        dst = os.path.join(args.out, rel)
        os.makedirs(dst, exist_ok=True)
        for name in COPY:
            src = os.path.join(rd, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst, name))
                copied += 1
            else:
                skipped += 1

        if not args.skip_derived:
            topic, network, alpha, seed = rel.split(os.sep)
            gaps += neighbor_rows(rd, {
                "topic": topic, "network": network,
                "alpha": round(float(alpha.split("_")[1]), 3),
                "seed": int(seed.split("_")[1])})

        if k % 60 == 0 or k == len(runs):
            print(f"  {k}/{len(runs)} runs")

    if gaps:
        gp = os.path.join(args.out, os.pardir, "neighbor_gap.csv")
        gp = os.path.normpath(gp)
        with open(gp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(gaps[0]))
            w.writeheader()
            w.writerows(gaps)
        print(f"  neighbor_gap.csv: {len(gaps)} rows, "
              f"{os.path.getsize(gp) / 1048576:.1f} MB")

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(args.out) for f in fs)
    print(f"\ncopied {copied} files ({skipped} absent)")
    print(f"bundle: {total / 1048576:.1f} MB at {args.out}")


if __name__ == "__main__":
    main()
