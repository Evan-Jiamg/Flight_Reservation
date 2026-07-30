"""Probe ranges needed to fix axis limits and truncation points."""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

G = "/mnt/NewSSD/CS_project/neil/hcog_experiments/M-1_main-grid/phi4"
ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

tconv = defaultdict(list)
lens = defaultdict(list)
nonconv = defaultdict(int)
cout_lo = []

for conv in glob.glob(os.path.join(G, "*/*/alpha_*/seed_*/convergence.json")):
    d = json.load(open(conv))
    a = round(float(d["alpha"]), 3)
    if d.get("t_conv") is None:
        nonconv[a] += 1
    else:
        tconv[a].append(d["t_conv"])
    lens[a].append(d["steps_run"])

print("alpha |  n  none |  t_conv  min  med  max | steps_run min med max")
for a in ALPHAS:
    tc = np.array(tconv[a], float)
    ln = np.array(lens[a], float)
    print(f"{a:5.3f} | {len(ln):3d} {nonconv[a]:4d} | "
          f"{tc.mean():7.1f} {tc.min():4.0f} {np.median(tc):4.0f} {tc.max():4.0f} | "
          f"{ln.min():9.0f} {np.median(ln):3.0f} {ln.max():3.0f}")

# how many runs are still alive at each step, per alpha -> truncation point
print("\nsteps at which the group drops below n=30 (half) and n=15 (quarter):")
for a in ALPHAS:
    ln = np.array(lens[a], float)
    alive = lambda t: int((ln >= t).sum())
    t30 = next((t for t in range(1, 130) if alive(t) < 30), 130)
    t15 = next((t for t in range(1, 130) if alive(t) < 15), 130)
    print(f"  alpha={a:5.3f}  n>=30 until t={t30-1:3d}   n>=15 until t={t15-1:3d}")

# C_out range (for the y limit) and metric ranges for fig 3
vals = defaultdict(list)
files = sorted(glob.glob(os.path.join(G, "*/*/alpha_*/seed_*/metrics.csv")))
for p in files[::7]:
    with open(p) as f:
        for r in csv.DictReader(f):
            for c in ("C_out", "dS", "dL", "deltacon"):
                try:
                    v = float(r[c])
                    if np.isfinite(v):
                        vals[c].append(v)
                except (TypeError, ValueError):
                    pass
print("\nmetric ranges (sampled runs):")
for c, v in vals.items():
    v = np.array(v)
    print(f"  {c:9s} n={len(v):6d}  min={v.min():.4f}  p1={np.percentile(v,1):.4f}  "
          f"med={np.median(v):.4f}  max={v.max():.4f}")
d = np.array(vals["deltacon"])
print(f"  1-deltacon  min={1-d.max():.5f}  med={1-np.median(d):.5f}  max={1-d.min():.5f}")
