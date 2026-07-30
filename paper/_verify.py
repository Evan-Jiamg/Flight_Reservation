"""Verify (1) the 69.2/56.7 figure in EXPERIMENT_RESULT.md 7, (2) Q_rand columns."""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

G = "/mnt/NewSSD/CS_project/neil/hcog_experiments/M-1_main-grid/phi4"

# ---------------------------------------------------------------- Q2: schema
print("=" * 70)
print("Q2  metrics.csv header (does it carry Q_rand_mean / Q_rand_sd?)")
print("=" * 70)
one = sorted(glob.glob(os.path.join(G, "*/*/alpha_*/seed_*/metrics.csv")))[0]
with open(one) as f:
    rdr = csv.DictReader(f)
    hdr = rdr.fieldnames
    row = next(rdr)
print(f"file: {one}")
print(f"n columns: {len(hdr)}")
for i, c in enumerate(hdr):
    print(f"  {i:2d}. {c:18s} = {row[c]}")
qcols = [c for c in hdr if "Q_" in c or "z_Q" in c or "swap" in c]
print(f"\nnull-model columns present: {qcols}")

# are they populated at every step, or only at the end?
with open(one) as f:
    rows = list(csv.DictReader(f))
for c in ("Q_rand_mean", "Q_rand_sd", "Q_norm", "z_Q", "swap_fail"):
    if c not in hdr:
        print(f"  {c}: ABSENT")
        continue
    vals = [r[c] for r in rows]
    nonempty = [v for v in vals if v not in ("", "nan", None)]
    print(f"  {c}: {len(nonempty)}/{len(vals)} steps populated; "
          f"first={vals[0]} last={vals[-1]}")

# ------------------------------------------------- Q1: the 69.2 / 56.7 pair
print()
print("=" * 70)
print("Q1  alpha=0 convergence by topic -- which statistic is 69.2 / 56.7?")
print("=" * 70)

by = defaultdict(list)
for conv in glob.glob(os.path.join(G, "*/*/alpha_*/seed_*/convergence.json")):
    d = json.load(open(conv))
    if abs(float(d["alpha"])) > 1e-9:
        continue
    d["_net"] = d.get("network")
    d["_seed"] = int(os.path.basename(os.path.dirname(conv)).split("_")[1])
    by[d["topic"]].append(d)

TARGET = {"gun_control": 69.2, "abortion": 56.7}

for topic, ds in sorted(by.items()):
    tc_all = np.array([d["t_conv"] for d in ds if d.get("t_conv") is not None], float)
    st_all = np.array([d["steps_run"] for d in ds], float)
    conv_only = [d for d in ds if d.get("attractor") != "none"]
    st_conv = np.array([d["steps_run"] for d in conv_only], float)
    tc_conv = np.array([d["t_conv"] for d in conv_only if d.get("t_conv") is not None], float)

    # per-network means, then averaged (unweighted across networks)
    nets = defaultdict(list)
    for d in ds:
        nets[d["_net"]].append(d["steps_run"])
    st_netmean = np.mean([np.mean(v) for v in nets.values()])
    netsc = defaultdict(list)
    for d in ds:
        if d.get("t_conv") is not None:
            netsc[d["_net"]].append(d["t_conv"])
    tc_netmean = np.mean([np.mean(v) for v in netsc.values()])

    print(f"\n--- {topic}  (n={len(ds)}, target in RESULT.md = {TARGET[topic]}) ---")
    cands = [
        ("mean t_conv (all)",            tc_all.mean()),
        ("mean t_conv + post_window 10", tc_all.mean() + 10),
        ("mean steps_run (all)",         st_all.mean()),
        ("mean steps_run (converged)",   st_conv.mean()),
        ("mean t_conv (converged)",      tc_conv.mean()),
        ("median steps_run",             float(np.median(st_all))),
        ("median t_conv",                float(np.median(tc_all))),
        ("steps_run, per-network mean",  st_netmean),
        ("t_conv, per-network mean",     tc_netmean),
    ]
    for name, v in cands:
        mark = "   <== MATCH" if abs(v - TARGET[topic]) < 0.25 else ""
        print(f"   {name:32s} {v:7.2f}{mark}")

    # seed-subset hypothesis: was RESULT.md computed before all seeds landed?
    print("   cumulative mean steps_run as seeds accumulate:")
    for smax in range(3, 11):
        sub = [d["steps_run"] for d in ds if d["_seed"] <= smax]
        if sub:
            print(f"      seeds 1-{smax:2d} (n={len(sub):2d}): {np.mean(sub):6.2f}", end="")
            if abs(np.mean(sub) - TARGET[topic]) < 0.25:
                print("   <== MATCH")
            else:
                print()

print("\npooled check (should reproduce the 7 table):")
tc = [d["t_conv"] for ds in by.values() for d in ds if d.get("t_conv") is not None]
st = [d["steps_run"] for ds in by.values() for d in ds]
print(f"  pooled mean t_conv    = {np.mean(tc):.2f}   (table says 52.2)")
print(f"  pooled mean steps_run = {np.mean(st):.2f}   (table says 62.1)")
