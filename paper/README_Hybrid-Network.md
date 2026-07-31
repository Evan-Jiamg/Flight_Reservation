# H-COG: Hybrid Coevolutionary Opinion Games

Simulation and analysis code for the H-COG framework. A population of `N = 50`
agents is partitioned by a mixing parameter `α` into **Type-C** agents, which
minimise a Friedkin–Johnsen cost in closed form, and **Type-L** agents, whose
opinions are produced by a local Phi-4 model. The population is embedded in one
of three topologies and evolved under K-NN (`K = 5`) opinion-similarity
rewiring until a structural convergence criterion fires.

Agent priors are drawn from empirically scored Reddit r/politics comments on two
topics, `gun_control` and `abortion` (see `../Reddit-Dataset/`).

**Main grid (M-1).** 2 topics × 3 topologies × 9 values of `α` × 10 seeds =
**540 runs**.

This README documents the code and data. Results and their interpretation
belong to the paper and are deliberately not duplicated here.

---

## Repository layout

```
Hybrid-Network/
├── core/                         agent behaviour and convergence machinery
│   ├── agent.py                    Type-L: Phi-4 with a three-stage memory
│   ├── numeric_agent.py            Type-C: Friedkin–Johnsen update
│   ├── prompt.py                   Type-L prompt templates
│   ├── scorer.py                   stance regressor wrapper
│   ├── convergence.py              C_out, dS, d_L, DeltaCon, ARI, stopping rule
│   └── utils.py                    vLLM client
│
├── simulation/                   running the model
│   ├── model.py                    rewiring, metrics, per-run output
│   ├── run_hybrid.py               a single run
│   ├── run_all_parallel.py         the sweep that produced M-1
│   └── recompute_poa.py            recompute PoA with the dynamic denominator
│
├── data/                         inputs, and the scripts that generate them
│   ├── init_agents.py              Reddit scores -> agent priors (data/agents/)
│   ├── gen_networks.py             degree-aligned topologies (data/networks/)
│   ├── networks/                   93 graphs: 3 families x seeds 1-30, plus a
│   │   │                           legacy seed_50 from the first pilot
│   │   └── _superseded_pre_rerun/  10 scale-free graphs as they stood before
│   │                               the 2026-07-25 regeneration
│   ├── agents/                     backgrounds, intrinsic opinions, stubbornness
│   └── lexicons/                   belief keywords, topic questions, perspectives
│
├── analysis/                     everything the paper is built from
│   ├── make_official_figs.py       five main figures
│   ├── make_convergence_figs.py    three convergence figures
│   ├── build_results_bundle.py     raw grid -> results/
│   ├── verify_bundle.py            hashes results/ against the raw grid
│   ├── export_converged_table.py   one row per run at its converged state
│   ├── decompose_poa.py            disagreement / conformity split
│   ├── build_summary.py            per-condition summary tables
│   ├── hcog_paths.py               resolves where the run grid is
│   ├── stats/                      the statistics the paper reports, with the
│   │                               table each one emits
│   ├── figures/                    the figures the paper uses
│   ├── summaries/                  generated aggregate tables
│   ├── tools/                      EPS validation, PDF->EPS, cropping
│   └── REGENERATING_FIGURES.md     how to rebuild any figure
│
├── scripts/                      serving Phi-4, tmux orchestration, monitoring
├── results/                      committed run data, 10.3 MB (see Data)
├── experiments/                  symlink to the raw grid on the data disk
└── logs/                         run logs
```

`experiments/`, `logs/` and the operational scratch in `ops/` are not tracked.

---

## Requirements

```bash
pip install -r requirements.txt
```

Versions are pinned to the environment that produced M-1. Only the simulation
needs a GPU and a model server; **the analysis and every figure need nothing
beyond numpy, scipy, pandas and matplotlib.**

`pdftops` (Poppler) and Ghostscript are required to produce and validate EPS.

---

## Data

The raw grid is **2.6 GB** and is not in this repository. 98% of it is LLM prose
— `opinions`, `reasonings`, `long_memory` and `short_memory` inside
`agents_data.json`, plus `agents_interaction_data.json` — which no figure or
table reads.

`results/` carries a **10.3 MB** analysis-ready extract instead.

| Retained per run | Purpose |
|---|---|
| `metrics.csv` | the 18 per-step metrics; the source of most figures |
| `convergence.json` | `t_conv`, attractor class, plateau height |
| `poa_components.csv` | disagreement / conformity decomposition |
| `agent_assignment.json` | which agents are Type-L |

| Retained per grid | Purpose |
|---|---|
| `neighbor_gap.csv` | the complete input to `fig_neighbor_gap`: 1.8 MB in place of the 2.3 GB it was computed from |
| `manifest.json` | sweep configuration |

| Excluded | Size | Reason |
|---|---|---|
| `model_overview.json` | 10.6 MB | byte-equivalent to `metrics.csv`, verified on 25 sampled runs; the same table in JSONL form |
| `edges_per_step.json` | 56.4 MB | every graph metric derived from it is already in `metrics.csv`, and the one figure requiring the final graph is served by `neighbor_gap.csv` |
| `agents_data.json` | 2204 MB | only `beliefs` is read, and only its final entry |
| `agents_interaction_data.json` | 342 MB | not read by any analysis |

`neighbor_gap.csv` is derived once rather than recomputed on demand: a full
re-run is roughly 40 GPU-hours, and although decoding is configured with
`temperature = 0` and a fixed seed, vLLM's continuous batching can perturb the
numerics for an identical prompt depending on what else is in the batch.

**Integrity.** `analysis/verify_bundle.py` hashes all 2160 retained files
against their sources; every one is byte-identical. Rebuilding the figures from
`results/` rather than the raw grid yields six byte-identical PNGs out of eight;
the remaining two differ by 1/255, from rounding in the derived table.

`results/` also carries **W-5_typeL-pilot**, six runs at α ∈ {0, 0.5} on
`gun_control`/`scale_free`. It is deliberately not a grid: it is the pilot that
established `T_max = 120`, the Type-C calibration of 60 having proved too short
for Type-L. The paper cites it in the convergence section, so it ships alongside
the main grid rather than being reconstructed from prose.

Analysis scripts locate the grid automatically: `results/` when present, the raw
grid otherwise, and `$HCOG_GRID` overrides both. `analysis/hcog_paths.py` owns
that resolution; no script hardcodes a grid location.

`data/` holds inputs and `results/` holds outputs, kept apart so that what was
measured is never confused with what was fed in. Neither is merged into
`analysis/`, which contains the code that reads them.

---

## Reproducing the paper figures

```bash
python3 analysis/make_official_figs.py
python3 analysis/make_convergence_figs.py
python3 analysis/tools/validate_eps.py analysis/figures
```

Both scripts write PNG (300 dpi), PDF and EPS into `analysis/figures/`.

| Figure | Produced by |
|---|---|
| `fig_poa_decomposition` | `make_official_figs.py` |
| `fig_neighbor_gap` | `make_official_figs.py` |
| `fig_metric_trajectories` | `make_official_figs.py` |
| `fig_attractor` | `make_official_figs.py` |
| `fig_stance_distribution` | `make_official_figs.py` |
| `convergence_tconv_ecdf` | `make_convergence_figs.py` |
| `convergence_cout_trajectory` | `make_convergence_figs.py` |
| `convergence_metric_agreement` | `make_convergence_figs.py` |
| `Pipeline` | drawn externally; the draw.io source is embedded in the PDF metadata |

`--titles` produces review copies carrying the title inside the artwork;
submission copies leave it to the caption. `--legacy-scores` redraws the stance
distribution from the pre-recalibration scores as `chart1_distribution`.

**EPS is produced by `pdftops`, not by matplotlib.** Nimbus Roman ships as an
OTF with CFF outlines, and matplotlib's PostScript backend can only wrap a
glyf-based TrueType font as Type 42, so a direct `savefig(".eps")` embeds an
invalid font that Ghostscript rejects outright. The PDF backend embeds the same
face correctly as CID Type 0C, so the PDF is the master. PostScript also has no
alpha channel: translucent fills are pre-blended onto white, and the two
overlapping stance densities are composited by hand into three regions. Read the
comments in `make_official_figs.py` before reintroducing `alpha=` on a filled
artist.

`analysis/REGENERATING_FIGURES.md` covers every figure, including those not kept
in the tree.

---

## Reproducing the reported statistics

Each script writes the table beside it, so a reviewer can compare what the code
produces against what was recorded.

```bash
python3 analysis/stats/stopping_cost.py   # -> tstats.txt
python3 analysis/stats/alpha_curves.py    # -> alpha_curves.txt
python3 analysis/stats/peak_test.py       # -> peak_test.txt
python3 analysis/stats/integrity_audit.py # -> integrity.txt (needs the raw grid)
```

| Script | Reports |
|---|---|
| `stopping_cost.py` | `t_conv` and `steps_run` per α: what dynamic stopping costs |
| `alpha_curves.py` | PoA, `Pz`, `Q_norm`, `C_out` pooled over networks and topics, with 95% CIs |
| `peak_test.py` | Welch's t-test for the PoA rise from α=0 to α=0.125, per topic |
| `integrity_audit.py` | file presence and column completeness across the campaign |

The first three run against the committed extract and reproduce their stored
tables byte for byte. `integrity_audit.py` inspects `agents_data.json` and
`edges_per_step.json`, which the extract omits, so it requires the raw grid.

`peak_test.py` tests the two topics separately rather than pooling them: they
share agent personas and stubbornness, and their stance scores correlate at
+0.881, so pooling would treat dependent samples as twice the evidence.

---

## End-to-end pipeline

```bash
# 1. agent priors from the scored Reddit corpus
python3 data/init_agents.py --topic gun_control
python3 data/init_agents.py --topic abortion

# 2. topologies, matched on mean degree across families
python3 data/gen_networks.py --out data/networks

# 3. serve Phi-4 (needed only for α < 1)
bash scripts/start_vllm.sh

# 4. a single run
python3 simulation/run_hybrid.py --topic gun_control \
        --network_type random --alpha 0.5 --seed 1

# 5. the full grid
python3 simulation/run_all_parallel.py

# 6. refresh the committed extract
python3 analysis/build_results_bundle.py \
        --grid experiments/M-1_main-grid/phi4 \
        --out  results/M-1_main-grid/phi4
python3 analysis/verify_bundle.py
```

`PYTHON=/path/to/python` overrides the interpreter used by the shell entry
points and by `run_all_parallel.py`.

---

## Metrics

| Metric | Definition | Interpretation |
|---|---|---|
| Polarization `Pz` | `(1/N) Σ(z_i − z̄)²` | variance of expressed opinions |
| Modularity `Q` | Louvain on the K-NN graph | community structure |
| `Q_norm`, `z_Q` | `Q` against a degree-preserving null model | structure beyond what the degree sequence alone forces |
| PoA | `C(z_t, G_t) / C*(G_t)` | equilibrium inefficiency; `≥ 1` by construction |
| `C_out` | correlation between consecutive out-neighbourhoods | primary convergence criterion |
| `dS`, `d_L`, DeltaCon, ARI | successive graphs compared | corroborating structural measures |

`C(z_t)` and `C*(G_t)` are evaluated on the same post-rewiring graph `G_t`, so
PoA measures inefficiency against the best achievable outcome on the current
topology rather than on a stale one.

---

## Agent types

### Type-C — Friedkin–Johnsen (`core/numeric_agent.py`)

Minimises

```
C_i = Σ_{j∈N_i}(z_i − z_j)² + ρ_i · K · (z_i − s_i)²
```

giving the closed-form update

```
z_i ← (Σ_{j∈N_i} z_j + ρ_i · K · s_i) / (|N_i| + ρ_i · K)
```

where `s_i` is the fixed intrinsic opinion and `ρ_i` the stubbornness
coefficient. Opinions lie in `[−1, +1]`.

### Type-L — Phi-4 (`core/agent.py`)

Three stages per step, only the last of which calls the model:

1. **Short-term memory** — up to 80 words of neighbour opinions heard this step.
2. **History compression** — a sliding window over past summaries, truncated to
   60 words.
3. **Opinion update** — the prompt carries the intrinsic opinion as a fixed
   anchor alongside both memories, and returns a new expressed opinion.

---

## K-NN rewiring

At each step every agent re-links to the `K = 5` agents minimising

```
|s_i − z_j^(t)|
```

that is, its own **intrinsic** opinion against each neighbour's **expressed**
opinion. The asymmetry is deliberate: a stable underlying preference seeking out
whoever currently sounds compatible. The rule applies identically to both agent
types.
