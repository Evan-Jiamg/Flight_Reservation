#!/usr/bin/env python3
"""
make_official_figs.py -- the figures that go into the paper, rebuilt.

Writes into analysis/figures/official_paper/ (PNG 300 dpi + PDF):

  fig_poa_decomposition.png   single panel: the cost split vs alpha, direct
                              labels inside the bands (no legend box over data)
  fig_neighbor_gap.png        single panel: disagreement with kept neighbours,
                              by agent type
  fig_metric_trajectories.png Pz / Q_norm / PoA over time, all nine alphas

Fixes applied relative to the earlier drafts
  * panel titles were built by stripping "$" out of the y-label, which printed
    raw LaTeX ("Q_{\\mathrm{norm}} trajectory"). Titles are now written out.
  * trajectories were truncated to the SHORTEST run in the group (17 steps),
    which hid the plateau. They now use the same alive-run rule as the
    convergence figures: average over runs alive at t, cut below MIN_ALIVE.
  * alpha is drawn on viridis, matching figures/convergence_*.
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
from matplotlib.patches import Patch
from scipy import stats

GRID = "/mnt/NewSSD/CS_project/neil/hcog_experiments/M-1_main-grid/phi4"
PROJ = "/home/neil/Information_Management_Project/Echo-Chamber-Simulation"
DATA = os.path.join(PROJ, "Hybrid-Network", "data")
ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
MIN_ALIVE = 30
K = 5

# ---------------------------------------------------------------- colour system
# Three encodings, kept strictly apart so a hue means one thing in this paper.
#
#   RESERVED  blue / orange  -> the agent-type axis ONLY. Type-C vs Type-L, and
#             its corollary in the cost split (conformity is the Type-L
#             signature), so the reader's association carries over correctly.
#   ORDERED   viridis        -> alpha. Perceptually uniform and monotone in
#             luminance, so the ordering survives greyscale.
#   OTHER     violet family  -> everything that is neither of the above, as an
#             ordinal ramp (attractor) or a single accent (stance).
#   3-WAY     violet/green/magenta -> the three convergence metrics.
#
# Every set below was run through the validator; see the module docstring.
CMAP = plt.get_cmap("viridis")
C_NUM, C_LLM = "#2a78d6", "#eb6834"      # RESERVED: Type-C / Type-L

# attractor is ordinal (fixed point = most locked -> plateau = least), so it
# gets a single-hue ramp: monotone L, light end 2.83:1, adjacent gaps >= 0.06
C_FIX, C_CYC, C_PLAT = "#2f2263", "#6351b0", "#9d8ed2"

# three convergence metrics. dS and dL overlap at y ~ 0.11, so those two carry
# the widest separation (green vs magenta: CVD dE 14.6, normal 35.4)
C_DS, C_DL, C_DC = "#008300", "#b03a8f", "#4a3aa7"

# stance topics: violet family (blue/orange are reserved for agent type).
# |dL| = 0.285 apart, plus distinct line styles -> greyscale-safe.
C_GUN, C_ABO = "#2f2263", "#9d8ed2"

INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#d8d7d3"
W1, H1 = 3.4, 2.6


def blend(color, a, onto="white"):
    """Alpha-blend `color` onto the canvas and return the equivalent opaque RGB.

    EPS carries no alpha channel. Matplotlib's PostScript backend therefore
    renders a partially transparent artist fully opaque, which would turn every
    CI band into a solid block sitting on top of its own mean line. Over an
    opaque white canvas the blend is exact, so pre-computing it here leaves the
    PNG and PDF pixel-identical to before while making the EPS match them.

    Draw order does the rest: an opaque band must be plotted before the line it
    belongs to, or use a lower zorder.
    """
    fg, bk = to_rgb(color), to_rgb(onto)
    return tuple(a * f + (1.0 - a) * b for f, b in zip(fg, bk))


# grid lines were drawn at alpha=0.8; fold that in once
GRIDB = blend(GRIDC, 0.8)

# In-figure titles are OFF for submission: top venues carry the title in the
# caption, not in the artwork (duplicated text, mismatched font, wasted column
# height). --titles re-enables them for review copies and slides. Panel names
# and condition labels such as "alpha = 0" are NOT titles and always stay.
TITLES = False


def title(ax, text):
    if TITLES:
        ax.set_title(text, pad=3)


def setup():
    plt.rcParams.update({
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        # Match the document body. \usepackage{times} resolves to Nimbus Roman
        # under pdfTeX, so this is the same typeface -- not merely a lookalike.
        # STIX is the Times-metric math companion, so an alpha in an axis label
        # renders like an alpha in the running text.
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,      # embed as TrueType, not Type-3
        "font.size": 7.5,
        "axes.titlesize": 8, "axes.labelsize": 8, "axes.labelcolor": INK,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.frameon": False, "legend.fontsize": 6.4,
        "legend.handlelength": 1.4, "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.28, "legend.columnspacing": 0.9,
        "lines.linewidth": 1.15, "text.color": INK, "axes.grid": False,
    })


def acolor(a):
    return CMAP(0.08 + 0.84 * a)


def ci95(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))


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


def band(series, min_alive=MIN_ALIVE):
    """Mean + 95% CI over runs alive at each step; cut where the group thins."""
    L = max(len(x) for x in series)
    M = np.full((len(series), L), np.nan)
    for i, x in enumerate(series):
        M[i, :len(x)] = x
    n = np.sum(np.isfinite(M), axis=0)
    keep = n >= min_alive
    if not keep.any():
        return None
    last = int(np.max(np.nonzero(keep)[0]))
    M, n = M[:, :last + 1], n[:last + 1]
    mean = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    tc = np.array([stats.t.ppf(0.975, max(k - 1, 1)) for k in n])
    return np.arange(1, last + 2), mean, tc * sd / np.sqrt(np.maximum(n, 1))


# ------------------------------------------------------------------ loading
def load_conv():
    out = defaultdict(list)
    for p in glob.glob(os.path.join(GRID, "*/*/alpha_*/seed_*/convergence.json")):
        d = json.load(open(p))
        out[round(float(d["alpha"]), 3)].append(d)
    return out


def load_series(cols):
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


def load_run_costs():
    rows = []
    for p in glob.glob(os.path.join(GRID, "*/*/alpha_*/seed_*/poa_components.csv")):
        a = round(float(os.path.basename(os.path.dirname(os.path.dirname(p)))
                        .split("_")[1]), 3)
        try:
            with open(p) as f:
                last = list(csv.DictReader(f))[-1]
            rows.append((a, float(last["poa_conflict"]), float(last["poa_conformity"])))
        except (OSError, IndexError, KeyError, ValueError):
            pass
    return rows


def load_agents(seeds):
    s_rho = {}
    for t in ("gun_control", "abortion"):
        raw = json.load(open(os.path.join(
            DATA, f"numeric_sim_opnions_and_stubbornness_num_agents_50_{t}.json")))
        s_rho[t] = np.array([float(raw["opinions"][str(i)]) for i in range(50)])
    rows = []
    for conv in glob.glob(os.path.join(GRID, "*/*/alpha_*/seed_*/convergence.json")):
        rd = os.path.dirname(conv)
        seed = int(os.path.basename(rd).split("_")[1])
        if seed not in seeds:
            continue
        d = json.load(open(conv))
        try:
            agents = json.load(open(os.path.join(rd, "agents_data.json")))
            edges = json.load(open(os.path.join(rd, "edges_per_step.json")))
            assign = json.load(open(os.path.join(rd, "agent_assignment.json")))
        except (OSError, json.JSONDecodeError):
            continue
        n = 50
        z = np.array([agents[str(i)]["beliefs"][-1] for i in range(n)])
        llm = set(assign.get("llm_agents", []))
        A = np.zeros((n, n))
        for (i, j) in edges[-1]:
            A[i, j] = 1.0
        deg = A.sum(axis=1)
        nbr = np.divide(A @ z, deg, out=np.full(n, np.nan), where=deg > 0)
        for i in range(n):
            if deg[i] == 0:
                continue
            rows.append((round(float(d["alpha"]), 3),
                         "Type-L" if i in llm else "Type-C",
                         abs(z[i] - nbr[i])))
    return rows


# ------------------------------------------------------------------ figures
def fig_poa(costs, outdir):
    """Single panel: what the efficiency loss is made of. Direct labels inside
    the bands replace the legend box, which previously sat over the data."""
    fig, ax = plt.subplots(figsize=(W1, H1))
    a = np.array(ALPHAS)
    dis = np.array([np.mean([c for x, c, _ in costs if x == v]) for v in ALPHAS])
    con = np.array([np.mean([c for x, _, c in costs if x == v]) for v in ALPHAS])

    ax.fill_between(a, 0, dis, color=blend(C_NUM, 0.9), linewidth=0)
    ax.fill_between(a, dis, dis + con, color=blend(C_LLM, 0.9), linewidth=0)
    ax.plot(a, dis + con, color=INK, linewidth=1.0)
    ax.plot(a, dis, color="white", linewidth=0.8)      # 2px surface gap

    # direct labels, placed inside each band
    i = 2
    ax.text(a[i], dis[i] * 0.52, "disagreement", color="white", fontsize=7.5,
            ha="center", va="center", weight="bold")
    ax.text(a[i], dis[i] + con[i] * 0.5, "conformity", color="white",
            fontsize=7.5, ha="center", va="center", weight="bold")

    ax.set_xlabel(r"mixing parameter $\alpha$")
    ax.set_ylabel("contribution to PoA")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, (dis + con).max() * 1.14)
    ax.yaxis.grid(True, color=GRIDB, linewidth=0.5)
    ax.set_axisbelow(True)
    # identity is never colour-alone: the in-band labels are kept AND a legend
    # is present, placed in the corner the stack vacates at high alpha
    ax.legend(handles=[Patch(facecolor=C_NUM, label="disagreement"),
                       Patch(facecolor=C_LLM, label="conformity")],
              loc="upper right", ncol=2)
    title(ax, "What the efficiency loss is made of")
    save(fig, outdir, "fig_poa_decomposition")


def fig_neighbor(rows, outdir):
    """Single panel: disagreement with the neighbours one kept, by type."""
    fig, ax = plt.subplots(figsize=(W1, H1))
    for t, c in (("Type-C", C_NUM), ("Type-L", C_LLM)):
        xs, ys, es = [], [], []
        for a in ALPHAS:
            v = [g for al, ty, g in rows if al == a and ty == t and np.isfinite(g)]
            if len(v) < 5:
                continue
            xs.append(a); ys.append(np.mean(v)); es.append(ci95(v))
        if not xs:
            continue
        xs, ys, es = np.array(xs), np.array(ys), np.array(es)
        ax.fill_between(xs, ys - es, ys + es, color=blend(c, 0.18),
                        linewidth=0, zorder=1)
        ax.plot(xs, ys, color=c, marker="o", markersize=3.6, label=t,
                zorder=3)
    ax.set_xlabel(r"mixing parameter $\alpha$")
    ax.set_ylabel(r"$|z_i - \bar{z}_{\mathcal{N}(i)}|$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color=GRIDB, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    title(ax, "Disagreement with the neighbours one kept")
    save(fig, outdir, "fig_neighbor_gap")


def fig_traj(series, conv, outdir):
    """Pz / Q_norm / PoA over time. Titles written out (no string surgery on
    the y-label), and curves cut by the alive-run rule, not by the shortest
    run in the group."""
    specs = [("polarization", r"$P_z(t)$", "Polarization"),
             ("Q_norm", r"$Q_{\mathrm{norm}}(t)$", "Null-corrected modularity"),
             ("poa", r"PoA$(t)$", "Price of Anarchy")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, (col, ylab, title) in zip(axes, specs):
        for a in ALPHAS:
            b = band(series[a][col])
            if b is None:
                continue
            steps, mean, ci = b
            c = acolor(a)
            ax.fill_between(steps, mean - ci, mean + ci, color=blend(c, 0.15),
                            linewidth=0, zorder=1)
            ax.plot(steps, mean, color=c, zorder=3,
                    label=(rf"$\alpha={a:g}$" if col == "polarization" else None))
        ax.set_xlabel(r"step $t$")
        ax.set_ylabel(ylab)
        ax.set_title(title, pad=3)
        ax.yaxis.grid(True, color=GRIDB, linewidth=0.5)
        ax.set_axisbelow(True)
    # nine series over three panels: a legend inside any panel lands on the
    # data, so it goes below the figure as a single row
    if TITLES:
        fig.suptitle("Metrics stop moving after convergence", y=1.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=9,
               bbox_to_anchor=(0.5, -0.13), columnspacing=1.1)
    save(fig, outdir, "fig_metric_trajectories")


def fig_attractor(conv, outdir):
    """Attractor composition vs alpha. Kept as a figure rather than a table
    because the finding is the SHAPE -- both pure populations low, the middle
    high -- which a nine-row table hides in its last column."""
    from collections import Counter
    # `none` (2 runs of 540, both at T_max with C_out already at 0.91-0.95) is
    # folded into `plateau`. What the two share, and what this figure is about,
    # is the absence of an exact recurrence -- so the merged bar reads "no exact
    # period". The two runs must still be disclosed in the caption, because
    # tab:convergence reports 538/540 converged and the bars would otherwise
    # imply 100%.
    kinds = ["fixed_point", "limit_cycle", "plateau"]
    cols = {"fixed_point": C_FIX, "limit_cycle": C_CYC, "plateau": C_PLAT}
    # white reads on the two dark steps, ink on the light one
    lab_ink = {"fixed_point": "white", "limit_cycle": "white", "plateau": INK}
    labels = {"fixed_point": "fixed point", "limit_cycle": "limit cycle",
              "plateau": "plateau"}
    fig, ax = plt.subplots(figsize=(W1, H1))
    a = np.array(ALPHAS)
    bottom = np.zeros(len(a))
    for k in kinds:
        v = []
        for al in ALPHAS:
            ds = conv[al]
            c = Counter(
                "limit_cycle" if str(d.get("attractor")).startswith("limit_cycle")
                else "plateau" if str(d.get("attractor")) in ("plateau", "none")
                else str(d.get("attractor"))
                for d in ds)
            v.append(100 * c.get(k, 0) / len(ds) if ds else 0)
        v = np.array(v)
        ax.bar(a, v, 0.085, bottom=bottom, color=cols[k],
               label=labels[k], edgecolor="white", linewidth=0.7)
        # aqua and yellow fall under 3:1 on a light surface -> direct labels
        for xi, vi, bi in zip(a, v, bottom):
            if vi >= 10:
                ax.text(xi, bi + vi / 2, f"{vi:.0f}", ha="center", va="center",
                        fontsize=5.8, color=lab_ink[k], weight="bold")
        bottom += v
    ax.set_xlabel(r"mixing parameter $\alpha$")
    ax.set_ylabel("share of runs (%)")
    ax.set_xticks(a)
    ax.set_xticklabels([f"{x:g}" for x in a], fontsize=6)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.32))
    ax.yaxis.grid(True, color=GRIDB, linewidth=0.5)
    ax.set_axisbelow(True)
    title(ax, "Mixed populations freeze; pure populations do not")
    save(fig, outdir, "fig_attractor")


def fig_stance(outdir, scores="stance_scores_bws", name="fig_stance_distribution"):
    """Reddit corpus stance distributions under the recalibrated scorer.

    Overlaid KDEs rather than side-by-side histograms: the two topics share one
    axis, so the reader compares them directly instead of across panels.
    Colour is the violet family -- blue and orange are reserved paper-wide for
    the agent-type axis, and topic is a different categorical variable. The two
    steps are far apart in luminance (|dL| = 0.285) and additionally separated
    by line style, so the pair survives greyscale printing.
    """
    import pandas as pd
    from scipy.stats import gaussian_kde

    GUN = {775, 708, 750, 705, 456, 725}
    ABO = {132, 131}
    df = pd.concat([pd.read_parquet(f) for f in glob.glob(
        os.path.join(PROJ, "Reddit-Dataset", scores, "*.parquet"))])
    gun = df[df.cluster_id.isin(GUN)].stance.dropna().values
    abo = df[df.cluster_id.isin(ABO)].stance.dropna().values

    x = np.linspace(-1.05, 1.05, 500)
    kg = gaussian_kde(gun, bw_method="scott")(x)
    ka = gaussian_kde(abo, bw_method="scott")(x)

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    # The two washes overlap almost everywhere. Painting them opaque would
    # let abortion simply hide gun there, so composite the overlap by hand:
    # it is exactly the region under min(kg, ka), and its colour is the
    # abortion wash laid over the gun wash. Three opaque areas reproduce the
    # translucent original exactly, and stay vector in the EPS.
    wash_gun = blend(C_GUN, 0.10)
    wash_abo = blend(C_ABO, 0.10)
    wash_both = blend(C_ABO, 0.10, onto=wash_gun)
    ax.fill_between(x, kg, color=wash_gun, linewidth=0, zorder=1)
    ax.fill_between(x, ka, color=wash_abo, linewidth=0, zorder=1)
    ax.fill_between(x, np.minimum(kg, ka), color=wash_both, linewidth=0,
                    zorder=1)
    for vals, kde, col, ls, lab in (
            (gun, kg, C_GUN, "solid", f"Gun control ($n$={len(gun):,})"),
            (abo, ka, C_ABO, (0, (6, 2)), f"Abortion ($n$={len(abo):,})")):
        ax.plot(x, kde, color=col, linestyle=ls, linewidth=1.5, label=lab,
                zorder=3)

    ymax = max(kg.max(), ka.max())

    # stance bands -- the quintile cuts used for stratified agent sampling
    for b in (-0.75, -0.25, 0.25, 0.75):
        ax.axvline(b, color=GRIDC, linewidth=0.6, linestyle=":", zorder=0)
    nl = chr(10)
    bands = [(-1.05, -0.75, "Strongly" + nl + "oppose"),
             (-0.75, -0.25, "Oppose"),
             (-0.25, 0.25, "Neutral"),
             (0.25, 0.75, "Support"),
             (0.75, 1.05, "Strongly" + nl + "support")]
    for lo, hi, band_label in bands:          # not `name` -- that is the
        ax.text((lo + hi) / 2, ymax * 1.10,   # output filename parameter
                band_label, ha="center", va="top",
                fontsize=6.4, color=INK2, linespacing=1.15)

    # stop the mean lines below the band-label row so they do not run into it
    for vals, col, ls in ((gun, C_GUN, "solid"), (abo, C_ABO, (0, (3, 2)))):
        ax.axvline(vals.mean(), color=blend(col, 0.9), linestyle=ls,
                   linewidth=1.1, zorder=4, ymin=0, ymax=0.80)
    ax.plot([], [], color=C_GUN, linestyle="solid", linewidth=1.1,
            label=f"mean gun $= {gun.mean():+.3f}$")
    ax.plot([], [], color=C_ABO, linestyle=(0, (3, 2)), linewidth=1.1,
            label=f"mean abortion $= {abo.mean():+.3f}$")

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0, ymax * 1.22)
    ax.set_xlabel("RoBERTa stance score "
                  "($-1$ = strongly oppose, $+1$ = strongly support)")
    ax.set_ylabel("density")
    ax.yaxis.grid(True, color=GRIDB, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.30),
              columnspacing=1.4)
    if TITLES:
        fig.suptitle("Reddit stance distributions under the recalibrated scorer",
                     y=1.02)
    save(fig, outdir, name)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(
        PROJ, "Hybrid-Network", "analysis", "figures", "official_paper"))
    ap.add_argument("--agent-seeds", type=int, default=4)
    ap.add_argument("--legacy-scores", action="store_true",
                    help="draw the stance figure from the pre-recalibration "
                         "scores (stance_scores) instead of stance_scores_bws; "
                         "writes chart1_distribution so it cannot overwrite the "
                         "BWS version")
    ap.add_argument("--titles", action="store_true",
                    help="draw in-figure titles (review copies / slides); "
                         "omit for submission -- the caption carries the title")
    args = ap.parse_args()
    global TITLES
    TITLES = args.titles
    setup()
    os.makedirs(args.outdir, exist_ok=True)

    print("loading ...")
    conv = load_conv()
    costs = load_run_costs()
    print(f"  {sum(len(v) for v in conv.values())} runs, {len(costs)} cost splits")

    print("PoA decomposition ...")
    fig_poa(costs, args.outdir)

    print(f"neighbour gap (seeds 1-{args.agent_seeds}) ...")
    rows = load_agents(set(range(1, args.agent_seeds + 1)))
    print(f"  {len(rows)} agent records")
    fig_neighbor(rows, args.outdir)

    print("metric trajectories ...")
    series = load_series(["polarization", "Q_norm", "poa"])
    fig_traj(series, conv, args.outdir)

    print("attractor composition ...")
    fig_attractor(conv, args.outdir)

    print("stance distribution ...")
    if args.legacy_scores:
        fig_stance(args.outdir, scores="stance_scores",
                   name="chart1_distribution")
    else:
        fig_stance(args.outdir)

    print(f"\n-> {args.outdir}")


if __name__ == "__main__":
    main()
