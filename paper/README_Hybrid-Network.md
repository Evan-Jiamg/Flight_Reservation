# H-COG Hybrid Network Simulation

Simulation engine for the **Hybrid Coevolutionary Opinion Game (H-COG)**.
Agents are split into **Type-C** (Friedkin–Johnsen cost-minimising) and
**Type-L** (Phi-4 LLM-driven) by the mixing parameter `α`, placed on one of
three topologies, and evolved with K-NN (K=5) opinion-similarity rewiring until
a structural convergence criterion fires.

**Topics:** `gun_control`, `abortion`, initialised from empirical Reddit
r/politics stance distributions (see `../Reddit-Dataset/`).

**Main grid (M-1):** 2 topics × 3 topologies × 9 α values × 10 seeds =
**540 runs**.

Results and their interpretation live in the paper, not here — this README
deliberately carries no results table, because the one it used to carry
described a superseded 180-run pilot and drifted out of step with the data.

---

## Layout

```
Hybrid-Network/
├── core/                    the agents themselves
│   ├── agent.py               Type-L: Phi-4 with the 3-stage memory mechanism
│   ├── numeric_agent.py       Type-C: Friedkin–Johnsen update
│   ├── prompt.py              Type-L prompt templates
│   ├── scorer.py              stance regressor wrapper
│   ├── convergence.py         C_out, dS, d_L, DeltaCon, ARI, the stopping rule
│   └── utils.py               vLLM client
│
├── simulation/              running the model
│   ├── model.py               the model: rewiring, metrics, per-run output
│   ├── run_hybrid.py          single run
│   ├── run_all_parallel.py    the sweep that produced M-1
│   └── recompute_poa.py       recompute PoA with the dynamic denominator
│
├── data/                    inputs, grouped by kind
│   ├── networks/              93 seeded topologies (3 kinds, seeds 1-30 + a legacy seed_50)
│   │   └── _superseded_pre_rerun/   scale-free graphs as they were before the
│   │                                07-25 regeneration; the June runs used them
│   ├── agents/                backgrounds, intrinsic opinions, stubbornness
│   ├── lexicons/              belief keywords, topic questions, perspectives
│   └── gen_networks.py        regenerates data/networks/
│
├── analysis/                figures and tables for the paper
│   ├── make_official_figs.py      the five main figures
│   ├── make_convergence_figs.py   the three convergence figures
│   ├── build_results_bundle.py    raw grid -> results/ (see "Data" below)
│   ├── verify_bundle.py           hashes results/ against the raw grid
│   ├── export_converged_table.py  one row per run at its converged state
│   ├── decompose_poa.py           disagreement / conformity split
│   ├── summaries/                 pre-aggregated JSON
│   ├── tools/                     EPS validation, PDF->EPS, cropping, cleanup
│   ├── legacy/                    make_figures.py, superseded by the above
│   ├── figures/official_paper/    the figures the paper uses
│   └── REGENERATING_FIGURES.md    how to rebuild any figure
│
├── plots/                   June-era plotting, still the source of the
│                            timeseries and reddit-only figures
│
├── reddit/                  stance scoring and agent initialisation
├── scripts/                 serving Phi-4, tmux, monitoring, sweeps
│
├── results/                 committed run data (~10 MB) — see "Data"
├── experiments/             symlink to the raw grid on the data disk, ignored
└── logs/                    run logs, ignored
```

---

## Data

The raw grid is **2.6 GB** and is not in the repository. 98% of it is LLM prose
— `opinions`, `reasonings`, `long_memory`, `short_memory` inside
`agents_data.json`, plus `agents_interaction_data.json` — which no figure or
table reads.

`results/` carries the analysis-ready extract instead, at **10.3 MB**:

| Kept per run | Why |
|---|---|
| `metrics.csv` | the 18 per-step metrics; the source of most figures |
| `convergence.json` | `t_conv`, attractor class, plateau height |
| `poa_components.csv` | disagreement / conformity split |
| `agent_assignment.json` | which agents are Type-L |
| `neighbor_gap.csv` (derived, per grid) | the entire input to `fig_neighbor_gap`, 1.8 MB in place of the 2.3 GB it was computed from |

Dropped, with the reason:

| Dropped | Size | Reason |
|---|---|---|
| `model_overview.json` | 10.6 MB | byte-equivalent to `metrics.csv`, verified on 25 sampled runs — the same table in JSONL form |
| `edges_per_step.json` | 56.4 MB | every graph metric derived from it is already in `metrics.csv`, and the one figure needing the final graph is served by `neighbor_gap.csv` |
| `agents_data.json` | 2204 MB | only `beliefs` is read, and only its last entry |
| `agents_interaction_data.json` | 342 MB | not read by any analysis |

`analysis/verify_bundle.py` hashes every file in `results/` against its source;
all 2160 are byte-identical. Rebuilding the figures from `results/` instead of
the raw grid reproduces them exactly — six of eight are byte-identical PNGs and
the other two differ by 1/255 from rounding in the derived table.

The analysis scripts find the grid automatically: `results/` if present, the raw
grid otherwise, or `$HCOG_GRID` if set.

---

## Setup

```bash
pip install -r requirements.txt
bash scripts/start_vllm.sh               # serve Phi-4 locally; needed for α < 1
```

The grid was produced with `/opt/anaconda3/envs/test_env` on the lab machine;
`requirements.txt` pins what was in it. The analysis and figure scripts need
only numpy, scipy, pandas and matplotlib — no GPU, no model server.

## Running

```bash
# one run
python3 simulation/run_hybrid.py --topic gun_control --network_type random \
        --alpha 0.5 --seed 1

# the M-1 sweep
python3 simulation/run_all_parallel.py
```

Output lands in `experiments/`, which is a symlink to the data disk. After a
sweep, refresh the committed extract:

```bash
python3 analysis/build_results_bundle.py \
        --grid experiments/M-1_main-grid/phi4 \
        --out  results/M-1_main-grid/phi4
```

## Figures

```bash
python3 analysis/make_official_figs.py
python3 analysis/make_convergence_figs.py --outdir analysis/figures/official_paper
python3 analysis/tools/validate_eps.py analysis/figures/official_paper
```

`analysis/REGENERATING_FIGURES.md` covers every figure, including the ones not
kept in the tree, and explains why EPS is produced by `pdftops` rather than by
matplotlib.

---

## Metrics

| Metric | Definition | Meaning |
|---|---|---|
| **Polarization** `Pz` | (1/N) Σ(z_i − z̄)² | variance of expressed beliefs; higher = more divided |
| **Modularity** `Q` | Louvain on the K-NN graph | community structure |
| **Q_norm**, `z_Q` | Q against a degree-preserving null model | modularity above what the degree sequence alone forces |
| **PoA** | C(z_t, G_t) / C*(G_t) | actual over socially optimal cost; ≥ 1 by construction |
| **C_out**, `dS`, `d_L`, DeltaCon, ARI | successive graphs compared | structural convergence; `C_out` is the primary criterion |

`C(z_t)` and `C*(G_t)` are both evaluated on the same post-rewiring graph `G_t`,
so PoA reflects inefficiency relative to the best achievable outcome on the
current topology rather than on a stale one.

---

## Agent types

### Type-C — Friedkin–Johnsen (`core/numeric_agent.py`)

Minimises

```
C_i = Σ_{j∈N_i}(z_i − z_j)² + ρ_i · K · (z_i − s_i)²
```

with the closed-form update

```
z_i ← (Σ_{j∈N_i} z_j + ρ_i · K · s_i) / (|N_i| + ρ_i · K)
```

`s_i` is the fixed intrinsic opinion, `ρ_i` the stubbornness coefficient,
beliefs in [−1, +1].

### Type-L — Phi-4 (`core/agent.py`)

Three stages per step, only the last of which calls the model:

1. **Short-term memory** — concatenates up to 80 words of neighbour opinions
   heard this step.
2. **History compression** — sliding window over past summaries, truncated to
   60 words.
3. **Opinion update** — the prompt carries the intrinsic opinion as a fixed
   anchor plus both memories, and returns a new expressed opinion.

Decoding uses `temperature=0` with a fixed seed. That makes a re-run
*intended* to be reproducible, but it is not a substitute for keeping the
data: vLLM's continuous batching can change the numerics for the same prompt
depending on what else is in the batch, and a full re-run is roughly 40 GPU
hours. This is why `neighbor_gap.csv` is derived once and committed rather
than recomputed on demand.

---

## K-NN rewiring

Each step every agent re-links to the K=5 agents minimising

```
|s_i − z_j^(t)|
```

its own **intrinsic** opinion against the neighbour's **expressed** one. The
asymmetry is deliberate: a stable preference seeking out whoever currently
sounds compatible. It applies identically to both agent types.
