# Pre-registration: Stage B selection and Stage C Task 2 (written 2026-09-24, before Stage B epoch 1/2 metrics)

Seen at time of writing: Stage A metrics (all epochs); fold0 Stage B epoch-0 inner-validation line
(NLL 0.6016, false-stop 0/16, K+1 0/4), which is the untrained PRISM adapter. Nothing else from Stage B.

## Stage B (per fold, independent runs, init = PRISM best = epoch2 SHA 2215132f…)
1. Checkpoint: minimum inner-validation NLL among epoch 0/1/2 (unchanged from plan).
2. Threshold: inner validation has 4 sessions / 4 K+1 positions per fold, which cannot estimate a
   threshold. Decision threshold stays **0.5** for Stage C. The full grid {0.1..0.9} curve is reported
   per fold and pooled over the three folds' inner validation (12 sessions) — descriptive only, no tuning.
3. Report per fold and pooled: NLL, AUC, false-stop, K+1 recall, session first-stop timing
   (early / exact / not-by-K+1 right-censored) with session bootstrap intervals (stop_timing.py).
4. "Useful ranking" flag (needed before any GRPO, not before Stage C diagnostics):
   pooled 12-session inner-validation AUC of the selected checkpoints > 0.60 AND pooled early-stop
   session rate at 0.5 ≤ 0.50. Failing it => GRPO is not started; next step is balanced SFT / hazard v2.

## Stage C (train-only Task 2, corrected runner rollout_stop_sft.py)
- Scenarios: fold inner-train ∩ req shards (task2_train_scenarios/foldX_inner_train_scenarios.json,
  14/12/14 scenarios × seeds 0,1). Runner asserts each ID against nested_manifest before loading models.
- Arms in one process, order randomised per (conversation_id, seed, replicate):
  `nogate` (frozen v2fix pipeline, Speaker end token only) and `sft` (selected Stage B checkpoint @0.5);
  `prism` (Stage B epoch 0 = PRISM best @0.5) is added when the selected checkpoint is not epoch 0.
- R0 gpt-5-mini minimal, cache off; judge gpt-5-mini minimal (cached by input); T_max 10; PLANNER_END off.
- Primary outputs: emitted user turns, decision steps, end_kind rates, coverage, complete; scenario-level
  paired difference, SE and cluster bootstrap (analyze_task2.py). One environment realisation =>
  intervals are conditional on that R0 draw; a second replicate is run before any claim.
- Quality guard (gate arm vs nogate, train scenarios): coverage paired mean difference ≥ −0.02 AND
  completed episodes not fewer than nogate by more than 1. Violations are reported as a trade-off/failure,
  never as "improvement" because turns fell.
- Inner-validation scenarios are used only to choose among arms/checkpoints; nothing from them or from
  outer test is shown to any reward editor. Outer test stays unread.

## Addendum A (2026-09-24, after Stage B gold-prefix metrics, BEFORE any Stage C Task 2 episode exists)
Observed: all folds select epoch 2; pooled inner-val AUC 0.681 (flag passes) but max p_stop 0.36, so the
0.5 gate never fires (K+1 0/12). This is the plan's "low recall" branch. Chosen v2 = **session-level hazard
decision rule**, no retraining: at each step the simulator stops with probability p_stop (Bernoulli).
Rationale: min-NLL selection optimises calibration, which is exactly what a sampled hazard needs; class-
balanced SFT would break calibration and is deferred unless the hazard rule fails.
- Gold prefix: per session, P(early)=1-Π_{s≤K}(1-h_s), P(exact)=h_{K+1}Π_{s≤K}(1-h_s),
  P(not by K+1, censored)=Π_{s≤K+1}(1-h_s); averaged over sessions, session bootstrap.
- Task 2: exact expectations over the logged no-gate trajectory (emitted turns, coverage, complete,
  end-kind probabilities); paired with no-gate at scenario level.
- New primary timing metric in Task 2 (all arms): signed and absolute error of emitted user turns vs the
  human session's user-turn count K for the same conversation_id (inner train/val only).
- Arms reported for Stage C: nogate; sft@0.5 (primary per original prereg); sft-hazard (v2);
  prism@0.5 and prism-hazard (Stage B epoch 0) for attribution; threshold grid descriptive only.
- Same quality guard as above applies to every arm. GRPO still not started before this readout.

## Readout C0 (2026-09-24 ~11:10, UserLM replicate 0, one environment realisation)
nogate 68 episodes: emitted 8.71, t_max 57 / speaker_end 10 / empty 1, coverage .192, complete 5/68.
Coverage gained after the human stop turn K: mean .012 (2/68 episodes gain). Hazard arms of the
selected Stage B adapters cut turns by 1.5–3.0 (inner train/val) with coverage −.006 or smaller, complete
−.003 or smaller: quality guard passes. Online/offline smoke equivalence exact (max |Δp| 3.7e-7).
GRPO entry conditions judged met; Stage D proceeds with the formulation below.

## Addendum B — Stage D design (written before any Stage D training)
Formulation: exact expected-return policy gradient for the hazard gate over LOGGED no-gate trajectories
(truncation property; no new rollouts, no sampling variance). Continue the fold's selected Stage B LoRA.
Loss per fold (inner train scenarios / inner train gold prefixes only):
  L = w_gold * NLL_goldprefix  +  w_len * (E[emitted] - mean K_human)^2 / Kvar   [batch-level, distributional]
      - lambda_cov * E[coverage]  - lambda_comp * E[complete]  +  beta * KL(h || h_stageB)
Coverage/complete act as constraints: E-values must stay >= nogate value - delta (delta_cov .02, delta_comp .02);
lambdas by dual ascent. gamma fixed = 1 (no discount); no exploration knob (expectation is exact).
Per-scenario K_human is NOT used as a per-episode target (plan §5); only the pooled length distribution.
Controllers (same budget of N=8 configuration updates, same epochs, same data, same seed):
  (1) Stage B only; (2) fixed config; (3) random search in bounds; (4) non-LLM dual ascent + fixed LR;
  (5) LLM controller (local/API LLM sees only inner-train aggregates; proposes log-LR within ±0.5 decade of
      5e-6, beta in [0.01,1], w_len in [0.1,10]; clipped, versioned, hashed).
Selection: inner-validation (gold-prefix NLL + Task 2 hazard metrics on inner-val scenarios); no editor access.
Outer test untouched until method freeze.
