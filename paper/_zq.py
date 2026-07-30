"""z_Q across the whole grid: is it populated, and what does it say?"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

G = "/mnt/NewSSD/CS_project/neil/hcog_experiments/M-1_main-grid/phi4"

rows = 0
nan_z = nan_qn = 0
files_with_any_nan = []
by_alpha = defaultdict(list)          # z_Q at the converged state
by_topic = defaultdict(list)
qobs_c, qrand_c, qnorm_c, zq_c, swap = [], [], [], [], []

for p in sorted(glob.glob(os.path.join(G, "*/*/alpha_*/seed_*/metrics.csv"))):
    a = round(float(os.path.basename(os.path.dirname(os.path.dirname(p))).split("_")[1]), 3)
    topic = p.split(os.sep)[-5]
    with open(p) as f:
        recs = list(csv.DictReader(f))
    bad = 0
    for r in recs:
        rows += 1
        try:
            z = float(r["z_Q"])
        except (TypeError, ValueError):
            z = float("nan")
        try:
            qn = float(r["Q_norm"])
        except (TypeError, ValueError):
            qn = float("nan")
        if not np.isfinite(z):
            nan_z += 1; bad += 1
        if not np.isfinite(qn):
            nan_qn += 1
    if bad:
        files_with_any_nan.append((os.path.relpath(p, G), bad, len(recs)))

    last = recs[-1]
    def g(k):
        try:
            return float(last[k])
        except (TypeError, ValueError):
            return float("nan")
    by_alpha[a].append(g("z_Q"))
    by_topic[topic].append(g("z_Q"))
    qobs_c.append(g("modularity")); qrand_c.append(g("Q_rand_mean"))
    qnorm_c.append(g("Q_norm")); zq_c.append(g("z_Q")); swap.append(g("swap_fail"))

print(f"rows scanned            : {rows:,}  (540 runs)")
print(f"z_Q    non-finite       : {nan_z}")
print(f"Q_norm non-finite       : {nan_qn}")
print(f"files with any bad z_Q  : {len(files_with_any_nan)}")
for f, b, n in files_with_any_nan[:5]:
    print(f"    {f}  {b}/{n} steps")

z = np.array(zq_c, float)
print(f"\nz_Q at the converged state (n={len(z)} runs)")
print(f"  min={np.nanmin(z):.1f}  p05={np.nanpercentile(z,5):.1f}  "
      f"median={np.nanmedian(z):.1f}  p95={np.nanpercentile(z,95):.1f}  max={np.nanmax(z):.1f}")
print(f"  runs with z_Q < 2 (i.e. NOT significantly non-random): "
      f"{int(np.nansum(z < 2))} / {len(z)}")

print("\nby alpha (converged state)")
for a in sorted(by_alpha):
    v = np.array(by_alpha[a], float)
    print(f"  alpha={a:5.3f}  n={len(v):2d}  z_Q mean={np.nanmean(v):6.1f}  "
          f"min={np.nanmin(v):6.1f}  max={np.nanmax(v):6.1f}")

print("\nby topic (converged state)")
for t in sorted(by_topic):
    v = np.array(by_topic[t], float)
    print(f"  {t:12s} n={len(v):3d}  z_Q mean={np.nanmean(v):6.1f}")

print("\nnull-model summary at the converged state, pooled over 540 runs")
for name, arr in (("Q_obs", qobs_c), ("Q_rand_mean", qrand_c),
                  ("Q_norm", qnorm_c), ("z_Q", zq_c)):
    v = np.array(arr, float)
    print(f"  {name:12s} mean={np.nanmean(v):7.4f}  sd={np.nanstd(v):7.4f}")
print(f"  swap_fail    total={int(np.nansum(swap))}")
