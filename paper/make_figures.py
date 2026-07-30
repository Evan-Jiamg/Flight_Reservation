#!/usr/bin/env python3
"""
make_figures.py -- regenerate every paper figure from the M-1 grid.

Palette follows the validated categorical/ordinal slots:
    Type-C / disagreement  blue   #2a78d6
    Type-L / conformity    orange #eb6834
    third slot             aqua   #1baf7a
    fourth slot            yellow #eda100
    alpha (ordinal, 5)     blue ramp #86b6ef -> #0d366b

Validator results (light surface, categorical):
    blue/orange           CVD dE 24.7, normal 33.6, contrast PASS
    4-way incl. aqua+yel  CVD dE  9.1, normal 22.9, contrast WARN
                          -> relief rule: those charts carry direct labels
    alpha ramp (5 steps)  monotone L, all gaps >= 0.06, light end 2.06:1

Outputs PDF (for LaTeX) and PNG (for quick viewing) into --outdir.
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------
C_NUM = "#2a78d6"   # Type-C / disagreement term
C_LLM = "#eb6834"   # Type-L / conformity term
C_AQUA = "#1baf7a"
C_YEL = "#eda100"
ALPHA_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
ALPHA_SHOWN = [0.0, 0.25, 0.5, 0.75, 1.0]

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d3"

GRID_ROOT = Path("/mnt/NewSSD/CS_project/neil/hcog_experiments/M-1_main-grid/phi4")
PROJ = Path("/home/neil/Information_Management_Project/Echo-Chamber-Simulation")
DATA = PROJ / "Hybrid-Network" / "data"
K = 5
ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
TOPICS = ["gun_control", "abortion"]
NETWORKS = ["scale_free", "random", "small_world"]

# Reddit cluster -> topic. Cluster 203 ("hero|trump|american") belongs to
# neither; excluding it reproduces the corpus sizes 3581 / 1618 exactly.
GUN_CLUSTERS = {775, 708, 750, 705, 456, 725}
ABO_CLUSTERS = {132, 131}


def setup_mpl():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "text.color": INK,
    })


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(Path(outdir) / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def ci95(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return 0.0
    from scipy import stats
    return float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_s_rho(topic, n=50):
    p = DATA / f"numeric_sim_opnions_and_stubbornness_num_agents_{n}_{topic}.json"
    raw = json.load(open(p))
    s = np.array([float(raw["opinions"][str(i)]) for i in range(n)])
    rho = np.array([float(raw["stubbornness"][str(i)]) for i in range(n)])
    return s, rho


def iter_runs(seeds=None):
    """Yield (topic, network, alpha, seed, run_dir) for every completed run."""
    for conv in sorted(GRID_ROOT.glob("*/*/alpha_*/seed_*/convergence.json")):
        d = json.load(open(conv))
        seed = int(conv.parent.name.split("_")[1])
        if seeds is not None and seed not in seeds:
            continue
        yield d["topic"], d["network"], float(d["alpha"]), seed, conv.parent, d


def load_agent_level(seeds):
    """Per-agent final-state records: intrinsic, expressed, neighbour mean, type."""
    rows = []
    for topic, net, alpha, seed, rd, conv in iter_runs(seeds):
        try:
            agents = json.load(open(rd / "agents_data.json"))
            edges = json.load(open(rd / "edges_per_step.json"))
            assign = json.load(open(rd / "agent_assignment.json"))
        except (OSError, json.JSONDecodeError):
            continue
        s, rho = load_s_rho(topic)
        n = len(s)
        z = np.array([agents[str(i)]["beliefs"][-1] for i in range(n)])
        llm = set(assign.get("llm_agents", []))
        # final graph
        A = np.zeros((n, n))
        for (i, j) in edges[-1]:
            A[i, j] = 1.0
        deg = A.sum(axis=1)
        nbr_mean = np.divide(A @ z, deg, out=np.full(n, np.nan), where=deg > 0)
        for i in range(n):
            rows.append({
                "topic": topic, "network": net, "alpha": alpha, "seed": seed,
                "agent": i, "type": "Type-L" if i in llm else "Type-C",
                "s": s[i], "z": z[i], "rho": rho[i],
                "drift": abs(z[i] - s[i]),
                "nbr_gap": abs(z[i] - nbr_mean[i]) if deg[i] > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def load_run_level():
    """Per-run converged-state metrics + convergence summary + cost split."""
    rows = []
    for topic, net, alpha, seed, rd, conv in iter_runs():
        rec = {"topic": topic, "network": net, "alpha": alpha, "seed": seed,
               "t_conv": conv.get("t_conv"), "steps": conv.get("steps_run"),
               "attractor": conv.get("attractor"),
               "C_plateau": conv.get("C_plateau"),
               "hit_T_max": conv.get("hit_T_max")}
        try:
            with open(rd / "metrics.csv") as f:
                m = list(csv.DictReader(f))
            if m:
                last = m[-1]
                for c in ("polarization", "modularity", "Q_norm", "poa", "C_out"):
                    try:
                        rec[c] = float(last[c])
                    except (KeyError, TypeError, ValueError):
                        rec[c] = np.nan
        except OSError:
            pass
        pc = rd / "poa_components.csv"
        if pc.exists():
            try:
                with open(pc) as f:
                    p = list(csv.DictReader(f))
                if p:
                    rec["conflict"] = float(p[-1]["poa_conflict"])
                    rec["conformity"] = float(p[-1]["poa_conformity"])
            except (OSError, KeyError, ValueError):
                pass
        rows.append(rec)
    return pd.DataFrame(rows)


def load_trajectories(cols, seeds):
    """Per-step series keyed by (topic, network, alpha, seed)."""
    out = {}
    for topic, net, alpha, seed, rd, conv in iter_runs(seeds):
        try:
            with open(rd / "metrics.csv") as f:
                m = list(csv.DictReader(f))
        except OSError:
            continue
        d = {}
        for c in cols:
            v = []
            for r in m:
                try:
                    v.append(float(r[c]))
                except (KeyError, TypeError, ValueError):
                    v.append(np.nan)
            d[c] = np.array(v)
        d["t_conv"] = conv.get("t_conv")
        out[(topic, net, alpha, seed)] = d
    return out


# --------------------------------------------------------------------------
# Figure 1 -- Reddit stance distribution (recalibrated)
# --------------------------------------------------------------------------
def fig_stance(outdir):
    import glob
    def load(dirname):
        return pd.concat([pd.read_parquet(f)
                          for f in glob.glob(str(PROJ / "Reddit-Dataset" / dirname / "*.parquet"))])
    new = load("stance_scores_bws")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=False)
    bins = np.linspace(-1, 1, 61)

    for ax, (tname, clusters) in zip(axes,
                                     [("Gun control", GUN_CLUSTERS),
                                      ("Abortion", ABO_CLUSTERS)]):
        x = new[new.cluster_id.isin(clusters)].stance.values
        ax.hist(x, bins=bins, color=C_NUM, alpha=0.85, edgecolor="white",
                linewidth=0.3)
        ax.axvline(0, color=INK2, linewidth=0.8, linestyle=":")
        ax.axvline(x.mean(), color=C_LLM, linewidth=1.8)
        ax.set_title(f"{tname}  ($n={len(x):,}$)")
        ax.set_xlabel("stance score")
        ax.set_xlim(-1, 1)
        sat = np.mean(np.abs(x) > 0.9) * 100
        # distributions lean left, so the upper-right corner is the clear space
        ax.text(0.97, 0.95,
                f"$\\mu={x.mean():+.3f}$\n$\\sigma={x.std():.3f}$\n"
                f"$|\\hat{{y}}|>0.9$: {sat:.1f}%",
                transform=ax.transAxes, va="top", ha="right", fontsize=8,
                color=INK2,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                          boxstyle="round,pad=0.3"))
    axes[0].set_ylabel("comments")
    save(fig, outdir, "fig_stance_distribution")


# --------------------------------------------------------------------------
# Figure 2 -- intrinsic vs expressed opinion, by agent type   [PRIORITY]
# --------------------------------------------------------------------------
def fig_drift(ag, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))

    # (a) scatter at a hybrid alpha where both types coexist
    ax = axes[0]
    sub = ag[(ag.alpha == 0.5) & (ag.topic == "gun_control")]
    for t, c in [("Type-C", C_NUM), ("Type-L", C_LLM)]:
        d = sub[sub.type == t]
        ax.scatter(d.s, d.z, s=9, c=c, alpha=0.45, linewidths=0, label=t)
    ax.plot([-1, 1], [-1, 1], color=INK2, linewidth=0.9, linestyle="--", zorder=0)
    ax.set_xlabel("intrinsic opinion $s_i$")
    ax.set_ylabel("expressed opinion $z_i$ at convergence")
    ax.set_title("(a) Where agents end up\n$\\alpha=0.5$, gun control")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")

    # (b) drift distribution by type
    ax = axes[1]
    data, colors, labels = [], [], []
    for t, c in [("Type-C", C_NUM), ("Type-L", C_LLM)]:
        data.append(ag[ag.type == t].drift.dropna().values)
        colors.append(c); labels.append(t)
    parts = ax.violinplot(data, showextrema=False, widths=0.75)
    for b, c in zip(parts["bodies"], colors):
        b.set_facecolor(c); b.set_alpha(0.55); b.set_edgecolor(c); b.set_linewidth(1.0)
    for i, (d, c) in enumerate(zip(data, colors), start=1):
        ax.hlines(np.median(d), i - 0.2, i + 0.2, color=INK, linewidth=1.8, zorder=3)
        ax.text(i, np.median(d) + 0.045, f"med {np.median(d):.2f}",
                ha="center", fontsize=7.5, color=INK)
    ax.set_xticks([1, 2]); ax.set_xticklabels(labels)
    ax.set_ylabel("$|z_i - s_i|$")
    ax.set_title("(b) Distance from one's own\nprior stance (all $\\alpha$)")
    ax.grid(axis="x", visible=False)

    # (c) drift vs alpha by type
    ax = axes[2]
    for t, c in [("Type-C", C_NUM), ("Type-L", C_LLM)]:
        xs, ys, es = [], [], []
        for a in ALPHAS:
            d = ag[(ag.type == t) & (ag.alpha == a)].drift.dropna().values
            if len(d) < 5:
                continue
            xs.append(a); ys.append(d.mean()); es.append(ci95(d))
        if not xs:
            continue
        xs, ys, es = np.array(xs), np.array(ys), np.array(es)
        ax.plot(xs, ys, color=c, marker="o", markersize=4.5, label=t)
        ax.fill_between(xs, ys - es, ys + es, color=c, alpha=0.16, linewidth=0)
    ax.set_xlabel(r"mixing parameter $\alpha$")
    ax.set_ylabel("mean $|z_i - s_i|$")
    ax.set_title("(c) Drift is a property of\nagent type, not of the mixture")
    ax.legend(loc="best")

    save(fig, outdir, "fig_opinion_drift")


# --------------------------------------------------------------------------
# Figure 3 -- own stance vs neighbours, by agent type   [PRIORITY]
# --------------------------------------------------------------------------
def fig_neighbors(ag, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))

    # (a) gap to neighbour mean vs alpha
    ax = axes[0]
    for t, c in [("Type-C", C_NUM), ("Type-L", C_LLM)]:
        xs, ys, es = [], [], []
        for a in ALPHAS:
            d = ag[(ag.type == t) & (ag.alpha == a)].nbr_gap.dropna().values
            if len(d) < 5:
                continue
            xs.append(a); ys.append(d.mean()); es.append(ci95(d))
        if not xs:
            continue
        xs, ys, es = np.array(xs), np.array(ys), np.array(es)
        ax.plot(xs, ys, color=c, marker="o", markersize=4.5, label=t)
        ax.fill_between(xs, ys - es, ys + es, color=c, alpha=0.16, linewidth=0)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$|z_i - \bar{z}_{\mathcal{N}(i)}|$")
    ax.set_title("(a) Disagreement with the\nneighbours one kept")
    ax.legend(loc="best")

    # (b) the trade-off: own stance vs neighbours
    ax = axes[2]
    for t, c in [("Type-C", C_NUM), ("Type-L", C_LLM)]:
        d = ag[ag.type == t]
        ax.scatter(d.drift, d.nbr_gap, s=5, c=c, alpha=0.20, linewidths=0)
        ax.scatter([d.drift.mean()], [d.nbr_gap.mean()], s=110, c=c,
                   edgecolors="white", linewidths=1.6, zorder=5)
        ax.annotate(t, (d.drift.mean(), d.nbr_gap.mean()),
                    textcoords="offset points", xytext=(9, 7),
                    fontsize=8.5, color=INK, weight="bold")
    ax.set_xlabel("$|z_i - s_i|$   (abandoning own stance)")
    ax.set_ylabel(r"$|z_i - \bar{z}_{\mathcal{N}(i)}|$   (arguing)")
    ax.set_title("(c) Two ways to be costly\n(large dots: type means)")

    # (b, middle) signed drift toward neighbours
    ax = axes[1]
    for t, c in [("Type-C", C_NUM), ("Type-L", C_LLM)]:
        d = ag[ag.type == t]
        sd = (d.z - d.s).values
        ax.hist(sd, bins=np.linspace(-1.2, 1.2, 61), color=c, alpha=0.55,
                label=t, density=True, edgecolor="none")
    ax.axvline(0, color=INK2, linewidth=0.9, linestyle=":")
    ax.set_xlabel("$z_i - s_i$  (signed)")
    ax.set_ylabel("density")
    ax.set_title("(b) Type-L spreads away from\nits anchor in both directions")
    ax.legend(loc="upper right")

    save(fig, outdir, "fig_neighbor_gap")


# --------------------------------------------------------------------------
# Figure 4 -- PoA cost decomposition
# --------------------------------------------------------------------------
def fig_poa_decomp(run, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    g = run.dropna(subset=["conflict", "conformity"]).groupby("alpha")
    a = np.array(sorted(g.groups))
    conf = np.array([g.get_group(x).conflict.mean() for x in a])
    confo = np.array([g.get_group(x).conformity.mean() for x in a])

    ax = axes[0]
    ax.fill_between(a, 0, conf, color=C_NUM, alpha=0.85, linewidth=0,
                    label="disagreement  $\\sum(z_i-z_j)^2$")
    ax.fill_between(a, conf, conf + confo, color=C_LLM, alpha=0.85, linewidth=0,
                    label="conformity  $\\sum\\rho_i K(z_i-s_i)^2$")
    ax.plot(a, conf + confo, color=INK, linewidth=1.2)
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("PoA (contribution)")
    ax.set_title("(a) What the efficiency loss is made of")
    ax.legend(loc="upper right")

    ax = axes[1]
    share = 100 * confo / (conf + confo)
    ax.plot(a, share, color=C_LLM, marker="o", markersize=4.5)
    for x, y in [(a[0], share[0]), (a[-1], share[-1])]:
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(6 if x == a[0] else -34, 6), fontsize=8, color=INK)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("conformity share of social cost (%)")
    ax.set_title("(b) Abandoning one's own stance is\nthe LLM population's dominant cost")
    save(fig, outdir, "fig_poa_decomposition")


# --------------------------------------------------------------------------
# Figure 5 -- F1: C_out(t) convergence curves
# --------------------------------------------------------------------------
def fig_cout(traj, run, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, net in zip(axes, NETWORKS):
        for a, c in zip(ALPHA_SHOWN, ALPHA_RAMP):
            series = [d["C_out"] for (t, n, al, s), d in traj.items()
                      if n == net and al == a]
            if not series:
                continue
            L = min(len(x) for x in series)
            L = min(L, 90)
            M = np.vstack([x[:L] for x in series])
            m = np.nanmean(M, axis=0)
            e = np.array([ci95(M[:, i]) for i in range(L)])
            t = np.arange(1, L + 1)
            ax.plot(t, m, color=c, label=f"$\\alpha={a:g}$")
            ax.fill_between(t, m - e, m + e, color=c, alpha=0.15, linewidth=0)
        tc = run[run.network == net].t_conv.dropna()
        if len(tc):
            ax.axvline(tc.median(), color=INK2, linestyle="--", linewidth=0.9)
            ax.text(tc.median() + 1.5, 0.62, f"median $t_{{conv}}$={tc.median():.0f}",
                    fontsize=7.5, color=INK2, rotation=90, va="bottom")
        ax.set_title({"scale_free": "Scale-free (BA)", "random": "Random (ER)",
                      "small_world": "Small-world (WS)"}[net])
        ax.set_xlabel("step $t$")
        ax.set_ylim(0.55, 1.005)
    axes[0].set_ylabel(r"$C^{\mathrm{out}}(t)$  (neighbours retained)")
    axes[-1].legend(loc="lower right", ncol=1)
    fig.suptitle("Structural convergence: neighbour retention rises to a plateau "
                 "in every topology", y=1.03)
    save(fig, outdir, "fig_convergence_cout")


# --------------------------------------------------------------------------
# Figure 6 -- F3: t_conv cumulative distribution
# --------------------------------------------------------------------------
def fig_tconv(run, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))

    ax = axes[0]
    for a, c in zip(ALPHA_SHOWN, ALPHA_RAMP):
        d = run[run.alpha == a].t_conv.dropna().values
        if not len(d):
            continue
        x = np.sort(d)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=c, label=f"$\\alpha={a:g}$")
    ax.axvline(35, color=C_LLM, linestyle="--", linewidth=1.2)
    ax.text(36, 0.12, "old fixed horizon $T=35$", fontsize=7.5, color=C_LLM, rotation=90)
    ax.set_xlabel(r"$t_{\mathrm{conv}}$")
    ax.set_ylabel("fraction of runs converged")
    ax.set_title("(a) Convergence time by mixture")
    ax.set_xlim(0, 120)
    ax.legend(loc="lower right")

    ax = axes[1]
    xs, ys, es = [], [], []
    for a in ALPHAS:
        d = run[run.alpha == a].t_conv.dropna().values
        xs.append(a); ys.append(d.mean()); es.append(ci95(d))
    xs, ys, es = np.array(xs), np.array(ys), np.array(es)
    ax.plot(xs, ys, color=C_NUM, marker="o", markersize=4.5)
    ax.fill_between(xs, ys - es, ys + es, color=C_NUM, alpha=0.16, linewidth=0)
    ax.axhline(35, color=C_LLM, linestyle="--", linewidth=1.2)
    ax.text(0.02, 36.5, "old fixed horizon $T=35$", fontsize=7.5, color=C_LLM)
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"mean $t_{\mathrm{conv}}$ (steps)")
    ax.set_title("(b) A fixed $T=35$ truncates most\nof the low-$\\alpha$ grid")
    save(fig, outdir, "fig_tconv")


# --------------------------------------------------------------------------
# Figure 7 -- F5: attractor composition
# --------------------------------------------------------------------------
def fig_attractor(run, outdir):
    kinds = ["fixed_point", "limit_cycle", "plateau", "none"]
    cols = {"fixed_point": C_NUM, "limit_cycle": C_LLM,
            "plateau": C_AQUA, "none": C_YEL}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    a = np.array(ALPHAS)
    counts = {k: [] for k in kinds}
    for al in ALPHAS:
        sub = run[run.alpha == al]
        n = len(sub)
        c = Counter("limit_cycle" if str(x).startswith("limit_cycle") else str(x)
                    for x in sub.attractor)
        for k in kinds:
            counts[k].append(100 * c.get(k, 0) / n if n else 0)
    bottom = np.zeros(len(a))
    w = 0.085
    for k in kinds:
        v = np.array(counts[k])
        ax.bar(a, v, w, bottom=bottom, color=cols[k],
               label=k.replace("_", " "), edgecolor="white", linewidth=0.8)
        # relief rule: aqua/yellow sit under 3:1 -> direct labels
        for xi, vi, bi in zip(a, v, bottom):
            if vi >= 9:
                ax.text(xi, bi + vi / 2, f"{vi:.0f}", ha="center", va="center",
                        fontsize=7, color="white", weight="bold")
        bottom += v
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("share of runs (%)")
    ax.set_xticks(a); ax.set_xticklabels([f"{x:g}" for x in a])
    ax.set_ylim(0, 100)
    ax.set_title("Mixed populations freeze; both pure populations do not")
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="x", visible=False)
    save(fig, outdir, "fig_attractor")


# --------------------------------------------------------------------------
# Figure 8 -- F4: metric trajectories with t_conv marked
# --------------------------------------------------------------------------
def fig_traj(traj, outdir):
    specs = [("polarization", "$P_z(t)$", None),
             ("Q_norm", "$Q_{\\mathrm{norm}}(t)$", None),
             ("poa", "PoA$(t)$", None)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (col, lab, _) in zip(axes, specs):
        for a, c in zip(ALPHA_SHOWN, ALPHA_RAMP):
            series = [d[col] for (t, n, al, s), d in traj.items() if al == a]
            series = [x for x in series if np.isfinite(x).any()]
            if not series:
                continue
            L = min(min(len(x) for x in series), 90)
            M = np.vstack([x[:L] for x in series])
            m = np.nanmean(M, axis=0)
            e = np.array([ci95(M[:, i]) for i in range(L)])
            t = np.arange(1, L + 1)
            ax.plot(t, m, color=c, label=f"$\\alpha={a:g}$")
            ax.fill_between(t, m - e, m + e, color=c, alpha=0.15, linewidth=0)
        ax.set_xlabel("step $t$"); ax.set_ylabel(lab)
        ax.set_title(lab.replace("$", "").replace("(t)", "") + " trajectory")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Metrics settle and stay settled after convergence", y=1.03)
    save(fig, outdir, "fig_metric_trajectories")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(PROJ / "Hybrid-Network" / "analysis" / "figures" / "paper"))
    ap.add_argument("--agent-seeds", type=int, default=4,
                    help="seeds used for per-agent figures (data volume)")
    ap.add_argument("--only", default=None, help="comma list of figure names")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    setup_mpl()
    only = set(args.only.split(",")) if args.only else None

    def want(n):
        return only is None or n in only

    print("loading run-level data ...")
    run = load_run_level()
    print(f"  {len(run)} runs")

    seeds = set(range(1, args.agent_seeds + 1))
    if want("drift") or want("neighbors"):
        print(f"loading agent-level data (seeds {sorted(seeds)}) ...")
        ag = load_agent_level(seeds)
        print(f"  {len(ag)} agent records")

    if want("stance"):
        print("stance distribution ..."); fig_stance(args.outdir)
    if want("drift"):
        print("opinion drift ..."); fig_drift(ag, args.outdir)
    if want("neighbors"):
        print("neighbour gap ..."); fig_neighbors(ag, args.outdir)
    if want("poa"):
        print("PoA decomposition ..."); fig_poa_decomp(run, args.outdir)
    if want("cout") or want("traj"):
        print("loading trajectories ...")
        traj = load_trajectories(["C_out", "polarization", "Q_norm", "poa"], seeds)
        print(f"  {len(traj)} trajectories")
    if want("cout"):
        print("C_out curves ..."); fig_cout(traj, run, args.outdir)
    if want("tconv"):
        print("t_conv ..."); fig_tconv(run, args.outdir)
    if want("attractor"):
        print("attractor ..."); fig_attractor(run, args.outdir)
    if want("traj"):
        print("metric trajectories ..."); fig_traj(traj, args.outdir)

    print(f"\nall figures -> {args.outdir}")


if __name__ == "__main__":
    main()
