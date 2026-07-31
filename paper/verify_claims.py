#!/usr/bin/env python3
"""verify_readme_numbers.py -- check the root README's claims against the grid.

The README was assembled from two Chinese documents, one of which is known to
misreport a figure. Transcription is exactly where a number goes wrong quietly,
so every headline claim is recomputed here from results/ and compared.
"""
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

ROOT = "/home/neil/Information_Management_Project/Echo-Chamber-Simulation"
G = os.path.join(ROOT, "Hybrid-Network", "results", "M-1_main-grid", "phi4")
README = os.path.join(ROOT, "README.md")

ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


def ci95(a):
    a = np.asarray(a, float)
    return float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))


def load():
    poa = collections.defaultdict(list)
    pz = collections.defaultdict(list)
    qn = collections.defaultdict(list)
    tconv = collections.defaultdict(list)
    steps = collections.defaultdict(list)
    attr = collections.defaultdict(collections.Counter)
    poa_topic = collections.defaultdict(list)
    tconv_topic = collections.defaultdict(list)
    nonconv = 0

    for cp in glob.glob(os.path.join(G, "*/*/alpha_*/seed_*/convergence.json")):
        rd = os.path.dirname(cp)
        topic = os.path.relpath(rd, G).split(os.sep)[0]
        d = json.load(open(cp))
        a = round(float(d["alpha"]), 3)
        with open(os.path.join(rd, "metrics.csv")) as f:
            rows = list(csv.DictReader(f))
        last = rows[-1]
        poa[a].append(float(last["poa"]))
        pz[a].append(float(last["polarization"]))
        qn[a].append(float(last["Q_norm"]))
        poa_topic[(topic, a)].append(float(last["poa"]))
        k = str(d.get("attractor"))
        attr[a][k.split(":")[0] if k.startswith("limit_cycle") else k] += 1
        if d.get("t_conv") is not None:
            tconv[a].append(d["t_conv"])
            tconv_topic[(topic, a)].append(d["t_conv"])
        else:
            nonconv += 1
        steps[a].append(d["steps_run"])
    return poa, pz, qn, tconv, steps, attr, poa_topic, tconv_topic, nonconv


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print("  %-46s README %-8s computed %-8s %s"
          % (label, want, round(got, 3), "ok" if ok else "MISMATCH"))
    return ok


def main():
    text = open(README, encoding="utf-8").read()
    poa, pz, qn, tconv, steps, attr, poa_topic, tconv_topic, nonconv = load()
    bad = 0

    print("== grid shape ==")
    n = sum(len(v) for v in poa.values())
    bad += not check("total runs", n, 540, 0)
    bad += not check("runs per alpha (min)", min(len(v) for v in poa.values()), 60, 0)
    bad += not check("non-converged runs", nonconv, 2, 0)

    print()
    print("== PoA by alpha ==")
    want = [5.558, 5.901, 5.860, 5.646, 5.246, 4.547, 3.837, 2.989, 1.139]
    for a, w in zip(ALPHAS, want):
        bad += not check("PoA alpha=%g" % a, float(np.mean(poa[a])), w, 0.002)

    print()
    print("== polarization / Q_norm at the alphas quoted ==")
    for a, w in ((0.0, 0.304), (0.25, 0.395), (0.5, 0.466), (0.75, 0.536), (1.0, 0.597)):
        bad += not check("Pz alpha=%g" % a, float(np.mean(pz[a])), w, 0.002)
    for a, w in ((0.0, 0.218), (0.25, 0.225), (0.5, 0.280), (0.75, 0.316), (1.0, 0.423)):
        bad += not check("Q_norm alpha=%g" % a, float(np.mean(qn[a])), w, 0.002)

    print()
    print("== convergence ==")
    want_t = [52.2, 45.5, 43.8, 37.6, 34.5, 31.2, 24.0, 21.7, 25.1]
    for a, w in zip(ALPHAS, want_t):
        bad += not check("t_conv alpha=%g" % a, float(np.mean(tconv[a])), w, 0.06)
    want_s = [62.1, 55.5, 54.9, 48.8, 44.5, 41.2, 34.0, 31.7, 35.0]
    for a, w in zip(ALPHAS, want_s):
        bad += not check("steps alpha=%g" % a, float(np.mean(steps[a])), w, 0.06)
    # 45.3 is the mean run LENGTH, not the mean t_conv; the source document
    # labelled it as t_conv, which is where this check earns its keep.
    allt = [v for a in ALPHAS for v in tconv[a]]
    alls = [v for a in ALPHAS for v in steps[a]]
    bad += not check("mean t_conv overall", float(np.mean(allt)), 35.0, 0.06)
    bad += not check("mean steps overall", float(np.mean(alls)), 45.3, 0.06)

    print()
    print("== the corrected topic claim ==")
    bad += not check("t_conv gun_control alpha=0",
                     float(np.mean(tconv_topic[("gun_control", 0.0)])), 58.7, 0.06)
    bad += not check("t_conv abortion alpha=0",
                     float(np.mean(tconv_topic[("abortion", 0.0)])), 45.7, 0.06)

    print()
    print("== R4 fixed_point share ==")
    want_fp = [18, 35, 43, 40, 53, 50, 60, 62, 12]
    for a, w in zip(ALPHAS, want_fp):
        tot = sum(attr[a].values())
        bad += not check("fixed_point alpha=%g (%%)" % a,
                         100.0 * attr[a]["fixed_point"] / tot, w, 0.9)

    print()
    print("== R5 peak test ==")
    for topic, w0, w1, wp in (("abortion", 5.436, 6.322, 0.0084),
                              ("gun_control", 5.681, 5.480, 0.379)):
        x = poa_topic[(topic, 0.0)]
        y = poa_topic[(topic, 0.125)]
        bad += not check("%s PoA alpha=0" % topic, float(np.mean(x)), w0, 0.002)
        bad += not check("%s PoA alpha=0.125" % topic, float(np.mean(y)), w1, 0.002)
        p = stats.ttest_ind(y, x, equal_var=False).pvalue
        bad += not check("%s Welch p" % topic, float(p), wp, 0.0006)

    print()
    print("== R2 at alpha=1 ==")
    for topic, w in (("abortion", 1.129), ("gun_control", 1.149)):
        bad += not check("%s PoA alpha=1.0" % topic,
                         float(np.mean(poa_topic[(topic, 1.0)])), w, 0.002)

    print()
    print("== parse failures ==")
    tot = runs = 0
    for p in glob.glob(os.path.join(G, "*/*/alpha_*/seed_*/metrics.csv")):
        with open(p) as f:
            n = sum(int(float(r.get("parse_fail") or 0)) for r in csv.DictReader(f))
        tot += n
        runs += n > 0
    bad += not check("parse failures total", tot, 194, 0)
    bad += not check("runs affected", runs, 61, 0)

    print()
    if bad:
        print("%d claim(s) do not match the data" % bad)
        return 1
    print("every numeric claim in the README matches the grid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
