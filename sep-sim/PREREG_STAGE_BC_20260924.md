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
