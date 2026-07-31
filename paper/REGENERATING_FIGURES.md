# Regenerating figures

Only `analysis/figures/` is kept in the tree. Every other figure
was deleted on 2026-07-31 because all of them are derived artefacts: the data
they are drawn from is kept, so they can be rebuilt on demand.

This file is the contract that makes that safe. If you add a script that writes
a figure, add its command here.

---

## What the paper uses

`analysis/figures/` — the only figure directory under version control. Eight
generated figures plus the externally drawn pipeline diagram.

```bash
cd Hybrid-Network/analysis

# five figures: PoA split, neighbour gap, metric trajectories, attractor,
# stance distribution (BWS scorer)
python3 make_official_figs.py

# three convergence figures into the same directory
python3 make_convergence_figs.py
```

Each writes `.png` + `.pdf`, then converts the PDF to `.eps` with `pdftops`.

### Why EPS is not written by matplotlib

Nimbus Roman ships on this machine as an OTF (CFF outlines). Matplotlib's
PostScript backend can only wrap a glyf-based TrueType font as Type 42, so a
direct `savefig(".eps")` produced an invalid font and Ghostscript rejected every
file with `/invalidfont in definefont`. The PDF backend embeds the same face
correctly as CID Type 0C, so the PDF is the master and Poppler converts it.

PostScript also has no alpha channel. Translucent fills are pre-blended onto
white in `blend()`, which is exact for a single fill; the two overlapping stance
KDE washes are composited by hand into three regions. Do not reintroduce
`alpha=` on a filled artist without reading the comments in `make_official_figs.py`.

### Variants

```bash
# review copies with in-figure titles (submission copies carry the title in the
# caption instead)
python3 make_official_figs.py --titles --outdir figures_titled
python3 make_convergence_figs.py --outdir figures_titled

# stance distribution from the PRE-recalibration scores, written as
# chart1_distribution so it cannot overwrite the BWS version
python3 make_official_figs.py --legacy-scores
```

### Helpers

```bash
tools/eps_from_pdf.sh figures      # rebuild every EPS from its PDF
tools/crop_pdf.sh figures/Pipeline.pdf   # trim a PDF to its ink, re-emit EPS+PNG
python3 tools/validate_eps.py figures
```

`validate_eps.py` checks the EPSF header, a clean Ghostscript render, absence of
raster content, and that every font is embedded and subsetted.

---

## Deleted figures and how to get them back

### `analysis/figures/hybrid_<network>_K5_alpha<a>_agents50.png` (27 files)

Per-run metric charts (steps vs polarization, modularity, PoA) written by
`simulation/model.py` at the end of every run. Not a record of anything: the
filename carries no seed, so each of the 27 files only ever held whichever seed
finished last.

The data is `metrics.csv`, kept for all 540 runs under
`experiments/M-1_main-grid/phi4/<topic>/<network>/alpha_<a>/seed_<nn>/`. The
same quantities are plotted properly, with confidence bands over all seeds, by
`make_official_figs.py`'s `fig_metric_trajectories`.

They reappear on their own the next time a simulation runs.

### `analysis/figures/convergence_*.{png,pdf,eps}` (9 files)

Byte-for-byte duplicates of three figures that now live in `analysis/figures/`
itself. The nesting is gone, so this class of duplicate cannot recur.

```bash
python3 analysis/make_convergence_figs.py --outdir analysis/figures
```

### `Hybrid-Network/figures/convergence_*.{png,pdf}` (6 files)

A stale copy of the same three figures from 07-29, predating the EPS and
colour-system work. The whole directory was removed; `analysis/figures` is the
only figure root. Regenerate with the command above if you really want them.

### `analysis/figures/timeseries/` (6 files)

```bash
cd Hybrid-Network/plots && python3 plot_convergence_multiples.py
```

Reads `analysis/timeseries_per_alpha_{gun_control,abortion}.json` and
`analysis/summary_final_step_*.json`, all four of which are kept. Those JSONs
are themselves rebuilt by `plots/build_summary.py`.

### `analysis/figures/reddit_only/` (4 files)

```bash
cd Hybrid-Network/plots && python3 plot_stance_distribution.py
```

Reads the stance parquet files under `Reddit-Dataset/stance_scores/`. Note that
`plots/config.py` points `REDDIT_DIR` at `stance_scores`, the pre-recalibration
scores — the BWS scores live in `stance_scores_bws/`.

### `analysis/figures/cross_topic/` and `analysis/figures/paper/`

Already gone before this cleanup.

```bash
cd Hybrid-Network/plots && python3 plot_paper_figures.py   # cross_topic/
```

`analysis/legacy/make_figures.py`, which produced `paper/` along with
`fig_opinion_drift`, `fig_convergence_cout` and `fig_tconv`, was removed: it
duplicated `make_official_figs.py` for every figure the paper actually uses, and
shipping two scripts that both emit `fig_neighbor_gap` invites the wrong one
being cited. It remains in git history if those three are ever wanted.

---

## Where the data comes from

The figure scripts read `results/M-1_main-grid/phi4`, the ~10 MB extract of the
2.6 GB raw grid, so a fresh clone rebuilds every figure with no access to the
data disk. They fall back to the raw grid if `results/` is absent, and
`$HCOG_GRID` overrides both. `analysis/build_results_bundle.py` regenerates the
extract after a new sweep; `analysis/verify_bundle.py` checks it against the
raw grid.

---

## The rule

Figures are build output. Keep the inputs — `metrics.csv`, `convergence.json`,
`agents_data.json`, `edges_per_step.json`, `poa_components.csv`, the aggregated
JSONs in `analysis/`, and the stance parquets — and let the scripts above
produce everything else.

Scripts in `plots/` must be run from `plots/`; they do `from config import ...`.
Scripts in `analysis/` carry absolute paths and run from anywhere.
