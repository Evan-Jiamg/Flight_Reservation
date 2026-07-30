#!/usr/bin/env python3
"""
make_convergence_figs.py -- the three convergence figures for Sec 4.6.1.

  figures/convergence_tconv_ecdf.png        (required)
  figures/convergence_cout_trajectory.png   (required)
  figures/convergence_metric_agreement.png  (optional / appendix)

No simulation is re-run; everything is read from the completed M-1 grid.

Conventions
  * dS / C_out / dL / deltacon are NaN at t = 1 (step-to-step differences),
    so every time axis starts at t = 2. Q_* columns start at t = 1 but are
    not used here.
  * alpha is an ORDERED variable -> one perceptually uniform sequential map
    (viridis). The three metrics in fig 3 are a CATEGORICAL set -> the first
    three validated categorical slots (blue / orange / aqua), not viridis.
  * Curves are averaged only over runs still alive at step t, and are cut
    where fewer than MIN_ALIVE of the 60 runs remain, so a late tail of slow
    runs cannot masquerade as the group mean.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy import stats

GRID = "/mnt/NewSSD/CS_project/neil/hcog_experiments/M-1_main-grid/phi4"
ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
MIN_ALIVE = 30          # half of the 60 runs per alpha
CMAP = plt.get_cmap("viridis")

# Blue and orange are RESERVED paper-wide for the Type-C / Type-L axis, so the
# three convergence metrics take a separate categorical set. dS and dL both sit
# near y ~ 0.11, so position cannot separate them -- they carry the widest gap
# in the set (green vs magenta: CVD dE 14.6, normal dE 35.4, both >= 3:1).
C_DS, C_DL, C_DC = "#008300", "#b03a8f", "#4a3aa7"

INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#d8d7d3"


def blend(color, a, onto="white"):
    """Alpha-blend `color` onto the canvas and return the equivalent opaque RGB.

    EPS carries no alpha channel, so matplotlib's PostScript backend renders a
    partially transparent artist fully opaque -- every CI band would become a
    solid block covering its own mean line. Over an opaque white canvas the
    blend is exact, so pre-computing it leaves PNG and PDF unchanged and makes
    the EPS match them. Bands are then drawn below their lines by zorder.
    """
    fg, bk = to_rgb(color), to_rgb(onto)
    return tuple(a * f + (1.0 - a) * b for f, b in zip(fg, bk))


# grid lines were drawn at alpha=0.8; fold that in once
GRIDB = blend(GRIDC, 0.8)

# single-column figure sizes (top-venue two-column layout)
W1, H1 = 3.4, 2.6
W3, H3 = 3.4, 2.4


def setup():
    plt.rcParams.update({
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Match the document body: \usepackage{times} resolves to Nimbus Roman
        # under pdfTeX, and STIX is its Times-metric math companion.
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "legend.fontsize": 6.2,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.28,
        "legend.columnspacing": 0.9,
        "lines.linewidth": 1.15,
        "text.color": INK,
        "axes.grid": False,
    })


def acolor(a):
    """Colour for a given alpha on the sequential map (dark = low alpha)."""
    return CMAP(0.08 + 0.84 * a)


def alabel(a):
    return rf"$\alpha={a:g}$"


# ---------------------------------------------------------------- loading
def load_convergence():
    out = defaultdict(list)
    for p in glob.glob(os.path.join(GRID, "*/*/alpha_*/seed_*/convergence.json")):
        d = json.load(open(p))
        out[round(float(d["alpha"]), 3)].append(d)
    return out


def load_series(cols):
    """{alpha: {col: list of per-run arrays}} for the requested columns."""
    out = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(GRID, "*/*/alpha_*/seed_*/metrics.csv")):
        a = round(float(os.path.basename(os.path.dirname(os.path.dirname(p)))
                        .split("_")[1]), 3)
        with open(p) as f:
            rows = list(csv.DictReader(f))
        for c in cols:
            v = []
            for r in rows:
                try:
                    v.append(float(r[c]))
                except (TypeError, ValueError, KeyError):
                    v.append(np.nan)
            out[a][c].append(np.asarray(v, float))
    return out


def band(series, min_alive=MIN_ALIVE):
    """Mean and 95% CI per step over runs alive at that step; cut when thin."""
    L = max(len(x) for x in series)
    M = np.full((len(series), L), np.nan)
    for i, x in enumerate(series):
        M[i, :len(x)] = x
    n = np.sum(np.isfinite(M), axis=0)
    keep = n >= min_alive
    if not keep.any():
        return None
    last = int(np.max(np.nonzero(keep)[0]))
    M = M[:, :last + 1]
    n = n[:last + 1]
    mean = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    tcrit = np.array([stats.t.ppf(0.975, max(k - 1, 1)) for k in n])
    ci = tcrit * sd / np.sqrt(np.maximum(n, 1))
    steps = np.arange(1, last + 2)
    return steps, mean, ci, n


def _eps_from_pdf(pdf, eps):
    """Convert the PDF master to EPS with Poppler.

    Not fig.savefig(".eps"): matplotlib's PS backend wraps a font as Type 42,
    which requires glyf-based TrueType outlines. Nimbus Roman ships here as an
    OTF with CFF outlines, so the direct EPS carried an invalid font and every
    Ghostscript run failed with "/invalidfont in definefont". The PDF backend
    embeds the same face correctly (CID Type 0C), so convert from that.

    Safe only because no artist is left partially transparent -- alpha is
    folded into opaque RGB by blend(). PostScript has no alpha channel, so
    surviving transparency would be silently rasterised here.
    """
    try:
        subprocess.run(["pdftops", "-eps", "-level3", pdf, eps],
                       check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"    EPS skipped ({e}); run eps_from_pdf.sh once pdftops exists")
        return False


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    # No dpi= override here. With bbox_inches="tight", passing a dpi that
    # differs from the figure dpi makes mixed-mode rendering size the
    # rasterised layer against one scale and place it against another --
    # the CI band lands in the wrong corner and is mostly clipped away.
    # The savefig.dpi rcParam (300, the print standard) applies to both.
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    ok = _eps_from_pdf(os.path.join(outdir, f"{name}.pdf"),
                       os.path.join(outdir, f"{name}.eps"))
    print(f"  wrote {name}.png / .pdf" + (" / .eps" if ok else ""))


# ---------------------------------------------------------------- figure 1
def fig_ecdf(conv, outdir):
    fig, ax = plt.subplots(figsize=(W1, H1))
    for a in ALPHAS:
        ds = conv[a]
        n_total = len(ds)                       # 60, including non-converged
        tc = np.sort([d["t_conv"] for d in ds if d.get("t_conv") is not None])
        if not len(tc):
            continue
        # steps at the observed values; a group with non-converged runs
        # deliberately tops out below 1.0
        x = np.concatenate(([0], tc))
        y = np.arange(0, len(tc) + 1) / n_total
        ax.step(x, y, where="post", color=acolor(a), label=alabel(a),
                solid_joinstyle="round")

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"attractor-entry step $t_{\mathrm{conv}}$")
    ax.set_ylabel("cumulative fraction of runs")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True, color=GRIDB, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", ncol=2, borderaxespad=0.3)
    save(fig, outdir, "convergence_tconv_ecdf")


# ---------------------------------------------------------------- figure 2
def fig_cout(series, conv, outdir, wide=False, tstart=3):
    """t = 2 is the one-off overwrite of the seeded topology by the first K-NN
    graph, so retention there is structurally low (0.26-0.67) and unrelated to
    convergence. Starting at t = 3 keeps the plateau heights legible; pass
    --cout-start 2 to include it."""
    fig, ax = plt.subplots(figsize=((6.9 if wide else W1), H1))
    ymin = 1.0
    for a in ALPHAS:
        b = band(series[a]["C_out"])
        if b is None:
            continue
        steps, mean, ci, _ = b
        m = steps >= tstart                     # C_out undefined at t = 1
        steps, mean, ci = steps[m], mean[m], ci[m]
        c = acolor(a)
        ax.fill_between(steps, mean - ci, mean + ci, color=blend(c, 0.16),
                        linewidth=0, zorder=1)
        ax.plot(steps, mean, color=c, label=alabel(a), zorder=3)
        ymin = min(ymin, np.nanmin(mean - ci))

        # No median-t_conv marker here. Its x would be meaningful but its y is
        # only "where the group mean happens to sit at the group median" --
        # not a quantity any run realises. The t_conv distribution is the whole
        # subject of convergence_tconv_ecdf, which reports it properly.

    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$C^{\mathrm{out}}(t)$")
    ax.set_xlim(tstart, None)
    # data-driven floor: no empty band below the lowest CI edge
    lo = np.floor(ymin * 50) / 50
    ax.set_ylim(lo, 1.005)
    ax.yaxis.grid(True, color=GRIDB, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", ncol=2, borderaxespad=0.3)
    save(fig, outdir, "convergence_cout_trajectory")


# ---------------------------------------------------------------- figure 3
def fig_agreement(series, conv, outdir, alpha=0.0):
    fig, ax = plt.subplots(figsize=(W3, H3))
    dc = series[alpha]["deltacon"]
    specs = [
        (series[alpha]["dS"], C_DS, r"$dS(t)$"),
        (series[alpha]["dL"], C_DL, r"$d_L(t)$"),
        ([1.0 - x for x in dc], C_DC, r"$1-\mathrm{DeltaCon}(t)$"),
    ]
    floor = None
    for arrs, c, lab in specs:
        b = band(arrs)
        if b is None:
            continue
        steps, mean, ci, _ = b
        m = steps >= 2
        steps, mean, ci = steps[m], mean[m], ci[m]
        pos = mean > 0
        if not pos.all():
            print(f"    note: {lab} mean hits 0 at {int((~pos).sum())} step(s); "
                  f"those points are dropped (log axis)")
        lo = np.clip(mean - ci, 1e-6, None)
        ax.fill_between(steps[pos], lo[pos], (mean + ci)[pos],
                        color=blend(c, 0.16), linewidth=0, zorder=1)
        ax.plot(steps[pos], mean[pos], color=c, label=lab, zorder=3)
        v = np.nanmin(mean[pos])
        floor = v if floor is None else min(floor, v)

    tc = [d["t_conv"] for d in conv[alpha] if d.get("t_conv") is not None]
    if tc:
        tmed = float(np.median(tc))
        ax.axvline(tmed, color=INK2, linestyle="--", linewidth=0.8, zorder=0)
        ax.annotate(rf"median $t_{{\mathrm{{conv}}}}={tmed:.0f}$",
                    xy=(tmed, 0.965), xycoords=("data", "axes fraction"),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=6.2, color=INK2, va="top", ha="left")

    ax.set_yscale("log")
    # the three metrics span roughly one decade, so the default decade ticks
    # give a single label -- set readable ticks explicitly
    from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
    ticks = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel("structural change per step")
    ax.set_xlim(2, None)
    ax.yaxis.grid(True, color=GRIDB, linewidth=0.5, which="major")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=1, borderaxespad=0.3)
    ax.set_title(rf"$\alpha={alpha:g}$", pad=3)
    save(fig, outdir, "convergence_metric_agreement")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--wide-cout", action="store_true",
                    help="render figure 2 at double-column width")
    ap.add_argument("--cout-start", type=int, default=3,
                    help="first step shown in figure 2 (3 skips the one-off "
                         "overwrite of the seeded topology; 2 includes it)")
    args = ap.parse_args()
    setup()

    print("loading convergence.json ...")
    conv = load_convergence()
    print(f"  {sum(len(v) for v in conv.values())} runs")
    print("loading metrics.csv series ...")
    series = load_series(["C_out", "dS", "dL", "deltacon"])
    print(f"  {sum(len(v['C_out']) for v in series.values())} series")

    print("figure 1 (ECDF) ...")
    fig_ecdf(conv, args.outdir)
    print("figure 2 (C_out) ...")
    fig_cout(series, conv, args.outdir, wide=args.wide_cout,
             tstart=args.cout_start)
    print("figure 3 (metric agreement) ...")
    fig_agreement(series, conv, args.outdir, alpha=0.0)

    # numbers worth quoting in the text
    print("\nfor the caption / text:")
    for a in (0.0, 1.0):
        tc = np.array([d["t_conv"] for d in conv[a]
                       if d.get("t_conv") is not None], float)
        print(f"  alpha={a:g}: t_conv mean={tc.mean():.1f} median={np.median(tc):.0f} "
              f"p90={np.percentile(tc,90):.0f} max={tc.max():.0f}")
    nn = {a: sum(1 for d in conv[a] if d.get("t_conv") is None) for a in ALPHAS}
    print(f"  non-converged per alpha: { {k:v for k,v in nn.items() if v} }")


if __name__ == "__main__":
    main()
