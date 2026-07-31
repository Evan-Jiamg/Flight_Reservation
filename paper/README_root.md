# Hybrid Coevolutionary Opinion Games

Companion code, data and results for the H-COG framework: how mixed populations
of cost-minimising and language-model agents shape opinion dynamics, echo
chamber formation and equilibrium efficiency in dynamically rewired social
networks.

A population of `N = 50` agents is split by a mixing parameter `α` into
**Type-C** agents, which minimise a Friedkin–Johnsen cost in closed form, and
**Type-L** agents, whose opinions are produced by a local Phi-4 model. Agent
priors are drawn from stance-scored Reddit r/politics comments. The population
is embedded in one of three topologies and evolved under K-NN (`K = 5`)
opinion-similarity rewiring until a structural convergence criterion fires.

```
Echo-Chamber-Simulation/
├── Reddit-Dataset/     Stage 1: stance corpus and agent priors
└── Hybrid-Network/     Stage 2: simulation, analysis and figures
```

Each has its own README. This file records the experimental design and the
results of the main grid.

---

## Experimental design

### The grid

| Dimension | Values | Count |
|---|---|---|
| `α` | 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 | 9 |
| topology | Barabási–Albert, Erdős–Rényi, Watts–Strogatz, **matched on mean degree** | 3 |
| seed | 1–10 | 10 |
| topic | gun control, abortion | 2 |

**540 runs, exactly n = 60 per α, with no gaps.** `N = 50`, `K = 5`, horizon set
by the stopping rule below rather than fixed.

Seeds were reduced from 30 to 10 over two revisions. Both were time trade-offs,
not methodological judgements: the first bought all three topologies, the second
followed the measured cost of low-α runs. The consequence should be disclosed —
confidence intervals at n = 10 are wider than originally designed, and they
capture variation in topology and type assignment only, not the sampling
variation in *which* 50 Reddit users were drawn.

### Convergence

Opinions never stop moving exactly, so convergence is defined on the **graph**,
not on the opinion vector. The primary criterion is the temporal correlation
coefficient of out-neighbourhoods (Tang et al., *Phys. Rev. E* 81, 055101(R),
2010; directed form: Büttner et al., *SpringerPlus* 5:1198, 2016):

```
C_i^out(t) = |N_i^out(t) ∩ N_i^out(t+1)| / sqrt(d_i^out(t) · d_i^out(t+1))
C^out(t)   = (1 / A^out) · Σ_i C_i^out(t)
```

K-NN fixes every out-degree at `K`, so `C^out` is exactly the fraction of
neighbours each agent keeps per step. Three further measures corroborate it and
are recorded per step: normalised Hamming distance `dS` over the edge set, the
spectral distance `d_L` between normalised Laplacian spectra, and DeltaCon
(Koutra et al., SDM 2013). `dS` cannot distinguish real structural change from
an edge swap between equivalent positions; `d_L` can. Partition stability is
tracked with the Adjusted Rand Index (Hubert & Arabie, 1985).

A run stops at `t_conv + post_window`, where `t_conv` is the first step at which
either an exact recurrence `E(t) = E(t−p)` holds persistently for some
`p ≤ P_MAX`, or `C^out` has been flat for `patience` consecutive steps:

```
| mean(C[t−W:t]) − mean(C[t−2W:t−W]) | / mean(C[t−2W:t−W]) < eps_C
```

| Parameter | Value | Basis |
|---|---|---|
| `W` | 10 | calibration |
| `eps_C` | 0.01 | fired in 90/90 calibration runs across both topics |
| `patience` | 5 | |
| `post_window` | 10 | run on after convergence to show the plateau |
| `T_max` | 120 | raised from 60 after a Type-L pilot; see below |
| `P_MAX` | 20 | periods above 20 are classified as aperiodic |

Calibrating on `C^out` rather than `dS` is deliberate: at `N = 50` the edge set
is small enough that per-step churn is a small integer, so `dS` is noisy and
needs `eps = 0.20` for the same trigger rate, while `C^out` is smooth enough at
0.01.

Each run is classified as `fixed_point` (the edge set stops changing),
`limit_cycle:p`, `plateau` (`C^out` flat with no exact period found), or `none`.

### Statistics

Welch's t-test throughout, because variance is visibly unequal across α, with
Benjamini–Hochberg FDR control across the family of adjacent-α comparisons.
Claims of *no difference* use two one-sided tests (TOST) with δ = 0.2 fixed in
advance, since a non-significant t-test is not evidence of equivalence.

The two topics share agent personas and stubbornness coefficients and their
stance scores correlate at +0.881, so they are **not independent samples**; they
are reported separately rather than pooled.

---

## Results

Readouts are taken at `t_conv + post_window`; intervals are 95% CI.

| | Finding | Status |
|---|---|---|
| **R1** | No topology shows a significant difference at any α, but equivalence is established only at α = 1.0 | partial |
| **R2** | Type-C agents land at PoA 1.139, against the 9/8 theoretical bound; Type-L agents land at 5.56 | established |
| **R3** | 43.6% of Type-L inefficiency comes from departing from one's own stance, against 16.1% for Type-C | established |
| **R4** | Mixed populations freeze more readily than pure ones: 18% → 62% → 12% | established |
| **R5** | The PoA peak is topic-specific: abortion peaks at α = 0.125 (p = 0.008, d = 0.71); gun control has no peak | established |

### R1 — Initial topology

Across nine α values and both PoA and polarization, the confidence intervals of
all three topologies overlap; nothing reaches significance. But TOST establishes
equivalence **only at α = 1.0** (largest pair difference −0.006, p < 0.0001). At
lower α the PoA standard deviation is 1.0–1.5, and at δ = 0.2 the data cannot
rule out a true difference of roughly 0.5 between topologies.

Reporting this as "topology does not matter" would overstate what the data show.

### R2 — The magnitude of the efficiency loss

| | PoA |
|---|---|
| abortion, α = 1.0 (n = 30) | 1.129 ± 0.001 |
| gun control, α = 1.0 (n = 30) | 1.149 ± 0.008 |
| Bindel–Kleinberg–Oren bound | 9/8 = 1.125 |

The optimum solved by `_compute_optimal_cost()`,
`z_opt = (L_sym + W)⁻¹ W s`, is exactly the fixed point of the FJ update, so
Type-C agents converge to a neighbourhood of the social optimum by construction;
the gap is the double-counting of each edge between individual and social cost.
The measured value sits essentially on the theoretical one. The model uses a
directed K-NN graph counting each directed edge once, so 9/8 does not apply
strictly, but the agreement does not look accidental.

PoA against α, pooled over topologies and topics:

| α | 0.0 | 0.125 | 0.25 | 0.375 | 0.5 | 0.625 | 0.75 | 0.875 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| PoA | 5.558 | 5.901 | 5.860 | 5.646 | 5.246 | 4.547 | 3.837 | 2.989 | **1.139** |
| ±CI | 0.309 | 0.265 | 0.268 | 0.249 | 0.242 | 0.222 | 0.193 | 0.193 | 0.005 |

Adjacent-α comparisons are non-significant up to α = 0.375 and significant from
there on (q = 0.036 at 0.375→0.5, falling to 3.9e−26 at 0.875→1.0). **α ≤ 0.375
is a plateau; beyond it the decline is monotone and significant.**

### R3 — What the inefficiency is made of

PoA is a total and cannot say whether a population is inefficient because it
argues or because it capitulates. The social cost separates:

```
C(z) = Σ_i Σ_{j∈N(i)} (z_i − z_j)²     disagreement with the neighbours one kept
     + Σ_i ρ_i · K · (z_i − s_i)²       conformity: departure from one's own stance
```

| α | PoA | disagreement | conformity | conformity share |
|---|---|---|---|---|
| 0.0 | 5.558 | 3.136 | 2.422 | **43.6%** |
| 0.5 | 5.246 | 3.610 | 1.636 | 31.2% |
| 1.0 | 1.139 | 0.956 | 0.183 | **16.1%** |

From α = 1 to α = 0 the disagreement term grows 3.3×, the conformity term
**13.2×**. The conformity share falls monotonically across all nine α values
without exception. Language-model agents are markedly easier to move off their
anchor, and that — not mutual disagreement — is the main source of their
inefficiency.

`analysis/decompose_poa.py` recomputes this after the fact and cross-validates
against the recorded `poa` column; all 540 runs agree to within 2% median
relative error.

### R4 — Mixed populations freeze

| α | 0.0 | 0.125 | 0.25 | 0.375 | 0.5 | 0.625 | 0.75 | 0.875 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| `fixed_point` share | 18% | 35% | 43% | 40% | 53% | 50% | 60% | **62%** | 12% |

Neither pure population freezes; mixtures do. At α = 0.875 just six agents are
Type-L, yet the edge set locks completely five times more often than with no
Type-L agents at all. The `C^out` plateau agrees: only α = 0 stands apart
(0.9129, about 8.7% of neighbours replaced per step) against 0.971–0.990
elsewhere. So Type-L rewiring churn is higher, but it does **not** rise
monotonically with the Type-L fraction — which corrects a reading taken from a
two-seed pilot.

A mechanism remains to be established. One testable conjecture: a few Type-L
agents supply an anchor that stops Type-C agents from adjusting indefinitely,
letting the system land on an exact fixed point instead of approaching one.

### R5 — The peak is topic-specific

| Topic | α = 0 | α = 0.125 | Welch t | p | Cohen's d |
|---|---|---|---|---|---|
| abortion | 5.436 | **6.322** | +2.74 | **0.0084** | **0.71** |
| gun control | 5.681 | 5.480 | −0.89 | 0.379 | −0.23 |
| pooled | 5.558 | 5.901 | +1.68 | 0.095 | 0.31 |

Pooling hides the effect. Abortion shows a significant rise at α = 0.125 with a
moderate-to-large effect; gun control has no peak at all, its maximum being at
α = 0.

Because the two topics share the same agent personas and stubbornness
coefficients, the difference can be attributed to the stance distribution rather
than to population composition: abortion is 68/32, gun control 54/46. A testable
conjecture is that on a topic with a clear majority, a few Type-C agents disrupt
an otherwise coherent Type-L coordination and drive social cost up, while on a
balanced topic they do not.

### Two monotone curves

| α | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| polarization | 0.304 ± 0.015 | 0.395 ± 0.013 | 0.466 ± 0.010 | 0.536 ± 0.006 | 0.597 ± 0.006 |
| `Q_norm` | 0.218 ± 0.008 | 0.225 ± 0.011 | 0.280 ± 0.006 | 0.316 ± 0.005 | 0.423 ± 0.002 |

Both rise monotonically in α with narrow intervals: language-model agents
produce **weaker** polarization and **weaker** community structure than
Friedkin–Johnsen agents, consistent with R3 — a population that is easily
persuaded does not hold camps, and so does not form strong communities.

Modularity is reported as `Q_norm = Q_obs − ⟨Q_rand⟩` against a
degree-preserving null model throughout. Roughly 44% of raw `Q` is reproduced by
a random graph with the same degree sequence, so raw `Q` overstates echo-chamber
strength.

### Convergence

| α | 0.0 | 0.125 | 0.25 | 0.375 | 0.5 | 0.625 | 0.75 | 0.875 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| `t_conv` | 52.2 | 45.5 | 43.8 | 37.6 | 34.5 | 31.2 | 24.0 | 21.7 | 25.1 |
| steps run | 62.1 | 55.5 | 54.9 | 48.8 | 44.5 | 41.2 | 34.0 | 31.7 | 35.0 |

Mean `t_conv` is 35.0 steps and mean total length 45.3, over the 538 runs
that converged. **Only 2 of 540 (0.37%) failed to converge within
`T_max = 120`**, both classified `none`.

Two consequences. `T_max = 120` is both necessary and sufficient: at 60, more
than half the low-α runs would have been truncated and no convergence claim
would hold. And **a fixed horizon of T = 35 is inadequate**, sufficient only for
α ≥ 0.75.

α = 1.0 breaks the monotone trend — 31.7 steps at α = 0.875 rising to 35.0 at
α = 1.0. Pure Type-C is slower than almost-pure Type-C, the same phenomenon as
the `fixed_point` collapse in R4 seen from the other side.

Gun control converges consistently more slowly than abortion: at α = 0,
`t_conv` 58.7 against 45.7 (68.5 against 55.7 steps run). This matches the
stance distributions — gun control is near-balanced with many boundary agents,
abortion has a clear majority and settles sooner.

---

## Data integrity and limitations

All 540 runs pass the audit in `Hybrid-Network/analysis/stats/integrity_audit.py`:
every expected output present and non-empty, `metrics.csv` carrying the same 18
columns in all 540, row counts matching `steps_run` with no truncation,
chain-of-thought text retained at 591–848 characters per step, no agent's memory
frozen (0/50), zero failures in the modularity null-model randomisation, and no
gaps in the grid.

Two things must be disclosed rather than glossed:

**Parse failures are not zero.** Guided decoding is implemented but was
**disabled** for this grid (`run_all_parallel.py`, `guided_decoding: False`).
194 parse failures occurred across 61 of the 540 runs. On a parse failure an
agent reuses its previous opinion, which is indistinguishable from deliberately
holding position — so the affected steps inflate apparent stubbornness, the
quantity R3 rests on.

**Confidence intervals understate uncertainty.** As noted above, they reflect
topology and assignment variation only, not which Reddit users were sampled.

---

## Reproduction

Every figure and statistic in the paper rebuilds from the ~10 MB extract
committed in `Hybrid-Network/results/`, with no access to the 2.6 GB raw grid:

```bash
cd Hybrid-Network
python3 analysis/make_official_figs.py        # five main figures
python3 analysis/make_convergence_figs.py     # three convergence figures
python3 analysis/tools/validate_eps.py analysis/figures

python3 analysis/stats/stopping_cost.py       # convergence table
python3 analysis/stats/alpha_curves.py        # the α curves above
python3 analysis/stats/peak_test.py           # R5
```

Each statistics script writes the table it reproduces beside it, so what the
code produces can be compared against what was recorded.

Every number quoted in this file is checked against the grid by

```bash
python3 Hybrid-Network/analysis/stats/verify_claims.py
```

which recomputes each one from `results/` and reports any that disagree. It was
written because this file was assembled from two working documents, and it
caught two figures in them that had been mislabelled.

Re-running the grid itself needs a GPU and a served Phi-4; see
`Hybrid-Network/README.md`. Rebuilding the stance corpus needs the 15 GB
Pushshift dump; see `Reddit-Dataset/README.md`.

---

## Requirements

Each stage pins its own environment: `Hybrid-Network/requirements.txt` and
`Reddit-Dataset/requirements.txt`. Neither analysis path needs a GPU — figures
and statistics need only numpy, scipy, pandas and matplotlib.

---

## Citation

```bibtex
@misc{hcog,
  title  = {Agent-based Modeling, Equilibrium, Echo Chambers, and Efficiency
            in Hybrid Coevolutionary Opinion Games},
  author = {Neil},
  year   = {2026}
}
```
