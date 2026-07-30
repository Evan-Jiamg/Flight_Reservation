"""What do the two attractor=none runs actually look like?"""
import csv
import glob
import json
import os

import numpy as np

G = "/mnt/NewSSD/CS_project/neil/hcog_experiments/M-1_main-grid/phi4"

for p in sorted(glob.glob(os.path.join(G, "*/*/alpha_*/seed_*/convergence.json"))):
    d = json.load(open(p))
    if d.get("attractor") != "none":
        continue
    rd = os.path.dirname(p)
    rel = os.path.relpath(rd, G)
    with open(os.path.join(rd, "metrics.csv")) as f:
        rows = list(csv.DictReader(f))
    c = []
    for r in rows:
        try:
            v = float(r["C_out"])
            c.append(v if np.isfinite(v) else np.nan)
        except (TypeError, ValueError):
            c.append(np.nan)
    c = np.asarray(c, float)
    ok = c[np.isfinite(c)]
    print(f"\n{rel}")
    print(f"  steps_run={d['steps_run']}  t_conv={d.get('t_conv')}  "
          f"hit_T_max={d.get('hit_T_max')}  C_plateau={d.get('C_plateau')}")
    print(f"  C_out: first5={np.round(ok[:5],3)}")
    print(f"         last10={np.round(ok[-10:],3)}")
    print(f"         mean(last20)={ok[-20:].mean():.4f}  sd(last20)={ok[-20:].std():.4f}")
    # how flat is the tail really? compare the two halves of the last 40 steps
    if len(ok) >= 40:
        a, b = ok[-40:-20].mean(), ok[-20:].mean()
        print(f"         mean(-40:-20)={a:.4f} vs mean(last20)={b:.4f}  "
              f"rel.diff={abs(a-b)/a:.4f}  (stopping rule eps_C = 0.01)")

# for contrast: a typical plateau run
print("\n--- contrast: three typical plateau runs ---")
n = 0
for p in sorted(glob.glob(os.path.join(G, "*/*/alpha_0.250/seed_*/convergence.json"))):
    d = json.load(open(p))
    if d.get("attractor") != "plateau":
        continue
    with open(os.path.join(os.path.dirname(p), "metrics.csv")) as f:
        rows = list(csv.DictReader(f))
    c = []
    for r in rows:
        try:
            v = float(r["C_out"])
            c.append(v if np.isfinite(v) else np.nan)
        except (TypeError, ValueError):
            c.append(np.nan)
    ok = np.asarray(c, float)
    ok = ok[np.isfinite(ok)]
    if len(ok) < 40:
        continue
    a, b = ok[-40:-20].mean(), ok[-20:].mean()
    print(f"  {os.path.relpath(os.path.dirname(p), G):46s} "
          f"steps={d['steps_run']:3d} t_conv={d['t_conv']:3d} "
          f"mean(last20)={b:.4f} rel.diff={abs(a-b)/a:.4f}")
    n += 1
    if n >= 3:
        break
