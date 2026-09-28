# pend + GRPO — the approved design (audit reference, 2026-09-25; updated to v16 2026-09-28)

Everything below was decided or approved by the user. An audit checks the code in
`sep-sim/` against THIS list: anything missing, wired differently, or still following an
older design is a finding. Anything the code does that is NOT on this list and changes
behaviour is also a finding ("unauthorised design change"). The full v16 details (logging fields,
exact refill rules, tests) are in `ops/SPEC_v16_grpo_opt.md`; where they differ, that file wins.

## v16 changes (2026-09-28, user approved items 1, 2, 3, 4 (floor 0.5), 6, 7, 8; item 5 declined)
1. Task 1 dynamic refill: when fewer groups than the base group count (`n_base_groups`) carry a gradient (reward std ≤
   min_group_std), more unused train_all conversations are drawn from the same rng, in rounds of
   ceil(deficit / 2), capped at --task1-convs refill conversations in total; rows marked `"refill"`; the aux
   examples and `acc` / `end_at_*` come from the base groups only (`acc_all` covers all).
2. Task 1 size: new --task1-G (Task 1 samples per group) SPEC 8 (min 2); --task1-convs SPEC 4 → 8; Task 2 --G stays 4.
3. Dr. GRPO: `grpo_std_norm = False` → group advantage A_i = R_i − mean(R) (no division by the group std);
   stop credit A_stop = S − mean(S), A_seq = (R − S) − mean(R − S); groups with std ≤ min_group_std are still
   skipped. RLOO / PPO unchanged.
4. Aux floor --stop-sup-floor F, SPEC 0.5 (user decision): effective aux weight = max(F, w_aux × anneal);
   `--stop-sup-weight 0` requires F = 0 (the SPEC floor is then 0); range 0 ≤ F ≤ 5; hist logs aux_floor,
   aux_floor_active; the LLM controller prompt says the weight is annealed towards the fixed floor F.
6. LLM controller w_dist bounds [1.0, 5.0] (was a lower bound of 0.1): w_dist can only be raised from 1.0.
7. Continuous Task 1 metric `bal_p` (teacher-forced P(end_session = true) at every validation decision point,
   greedy Planner output; invalid points count as P_end = 0), plus auc / logloss / n_points / n_final /
   n_invalid. bal_p replaces term_f1 in checkpoint selection and in the D2 trigger (validation bal_p ≥
   untrained bal_p + --t1-trigger-margin, SPEC 0.10, at two consecutive validations); term_f1 is still
   computed and reported, not selected on.
8. `task1_pooled.py` (evaluation tool, not a training file): pools Task 1 over folds (duplicate conversation
   ids within an arm raise), M2 term_f1 / premature rates, paired per-conversation bootstrap (10000, seed 0)
   between two arms with identical conversation sets; records input sha256.
- Item 5 (per-person length reward) is NOT done: the visible number of requirements does not predict the real
  turn count (r = −0.36, almost always 7 or 8 requirements) and per-person turn counts vary little (sd 1.1);
  rewarding an unpredictable target is optimised by the middle value for everyone, i.e. the status quo plus noise.
- New SPEC gate values (else --ablation): task1_G 8, task1_convs 8, t1_trigger_margin 0.10, stop_sup_floor 0.5;
  none of them may change on resume.
- New verify checks: rl.adv_norm (run_meta algo_cfg.grpo_std_norm is False; learner_stats.adv_abs_mean logged),
  rl.task1_G, rl.task1_refill, rl.aux_floor, rl.w_dist_floor, rl.d2_trigger, rl.task1_prob; rl.selection
  recomputes the score with summary `selection_task1_metric` (term_f1 for older runs).

## Architecture (arm `pend`, built on the E1.6 tree `trees/e1r_cf19400`)
- Planner = Qwen3-4B-Instruct-2507 (the RL policy, LoRA r16). Speaker = Ditto-8B (frozen) + Selector.
- NO annotations (NO_ANN), NO goal judge. ACT_FULL, T1_SAMPLE, ROLESTOP on. Stop override OFF,
  no length band / clamp. Only Task-2-true facts in the Planner prompt; the full goal.
- The Planner judges goal completion itself (`goal_met`, `still_wanted`) and decides `end_session`.
- `end_session: true` at turn t ⇒ message t is the LAST message: its act = the Planner's own
  highest-p Complete entry (with that entry's length); the Speaker block gets an explicit line
  "this is their last message…"; Ditto writes a closing message (need not say thanks); the
  message is emitted, then the episode ends (no R0 reply). Approved.
- A Complete act drawn while NOT ending is redrawn from the Planner's non-Complete entries
  (p > 0, renormalised). Approved.
- Turn 1 cannot end (the raw answer is ignored and recorded). Approved.
- A blank Speaker message = END (benchmark convention). Approved.
- Generation caps: Planner max_new 1536, Speaker max_new 512. A Planner output that hits the cap
  is treated as unparsed. A Speaker candidate that hits the cap is ineligible and never emitted
  when any whole candidate exists.
- Prompts are FITTED, never truncated; for pend any compaction (dropping history) is a FAIL.
- R0 (task agent) and ledger judge = our own gpt-oss-120b on vLLM, GPU0, port 8029
  (`R0_BASE_URL`/`JUDGE_BASE_URL=http://127.0.0.1:8029/v1`, model `gpt-oss-120b`). R0 replies cut
  by length are re-requested; still-cut replies are counted. Ledger judge floor 4000 tokens, one
  re-request at 8000 when the answer is not a verdict object; empty/unparseable answers counted.

- The E1.6 "- stopping: <rule>" line appears ONLY in the Speaker block of the last message; the
  Planner's own state never carries it (approved 2026-09-26).
- Also approved (2026-09-26, previously implicit): an episode with no emitted message is unclean; with
  no guard survivor a whole non-blank candidate is preferred over a whole blank one; end_session with no
  Complete entry keeps the drawn act, and a Complete act with no non-Complete alternative is kept (both
  counted, verify WARN); evaluation episodes must be clean (verify FAIL); T_MAX = 10.

## Implicit Profile (both tasks, from the first message)
- Every turn t ≥ 2 the Planner writes `profile_note` (how THIS person writes; never advice to the
  assistant). Task 1: compare the simulator's PREDICTION of message t-1 with the REAL message t-1
  (+ a deterministic measured difference); nothing from message t or later. Task 2: self-critique
  of the simulator's own message t-1.
- Notes accumulate (never cut, never dropped; an exact repeat of the last note is skipped) and are
  fed to the Planner every turn and ALL of them to the Speaker block.
- Speaker few-shot: k = 3 real messages of the same writing style (interaction style + English
  proficiency; back-off to the same interaction style when < k), excluding the current
  conversation and every conversation sharing its goal or persona. Pool: fold runs =
  splits[fold].train_all; whole-corpus run = leave-one-out. Each candidate slot gets its own
  examples. Copying ≥ 8 consecutive words of an example = guard failure. Approved constants:
  k = 3, COPY_NGRAM = 8, duplicate redraw rounds = 4. Also approved (2026-09-26): at turn 1 the
  examples prefer other people's FIRST messages (later turns: non-first messages); a candidate
  that writes our block headers / "profile_note" into the message is rejected ("template" guard).
  The back-off is decided on the total count before the first-message preference; a repeated note is
  compared ignoring case/whitespace; the copy guard checks against the examples of all slots (approved).

## Candidates + Selector
- 4 candidates per turn (greedy + 3 samples at T 0.7 / top-p 0.9; turn 1 all sampled at the Ditto
  card values). E1.6 guards; exact-duplicate candidates are redrawn per slot (new seed, new
  examples); when every candidate fails, extra draws WITHOUT examples.
- Selector = length + style Borda (SimCSE on CPU, token-chunked). Task 1 style references = the
  person's own real earlier messages (turn ≥ 2), otherwise the few-shot examples.
- F6 metrics: official all-candidate definition + a greedy-only diagnostic.

## Task 1 (teacher-forced) — M2 mapping (decision D7)
- Row t (t = 1..n, real history): `greedy_ended` = Speaker blank at t OR Planner end decided at
  t-1 (row 1: blank only). K+1 probe: `ended` = Planner end at n OR Speaker blank at K+1.
  Raw decisions are kept. E1.6 is NOT recomputed.
- termination_f1: TP = K+1 ends, FP = END flags on real rows, FN = conversations without K+1 end.

## GRPO (Stage B; Stage A = the same evaluation without GRPO)
- D1 = (b): reward v4 = w_cov·coverage + w_dist·[log p_h(T) − log q(T)] − λ·format penalties.
  p_h = smoothed distribution of real people's message counts over splits[fold].train_all ONLY;
  q = smoothed distribution of the current update's (clean) rollouts. Coverage stays (D3,
  disclosed). Parameter sizes are to be explored.
- Group advantages (v16, Dr. GRPO): A_i = R_i − mean(R) of the group, NOT divided by the group std
  (`grpo_std_norm = False`); groups whose reward std ≤ min_group_std are skipped (no gradient) and counted.
- Stop credit: the length term's group advantage goes only to the end_session value tokens; the
  rest keeps the sequence-level advantage. stop_mask only on real decisions (t ≥ 2, parsed,
  valid end_session, not capped).
- D4: profile_note tokens carry no sequence advantage (KL only).
- Task 1 stop groups on train_all conversations (last message + one earlier, t ≥ 2): --task1-convs
  (SPEC 8) conversations per update, --task1-G (SPEC 8) samples per group (v16; Task 2 keeps --G 4), plus
  the v16 dynamic refill (unused train_all conversations, at most --task1-convs more, until the number of
  groups with a gradient reaches the base group count); their advantage acts on the end_session tokens only (approved),
  reward 1 if end_session agrees with the real person; non-decisions reward 0, no stop mask. Aux examples come
  from the base (non-refill) groups only.
- D2: auxiliary stop-token supervision on the Task 1 positions, annealed linearly (over 10 updates)
  towards the floor --stop-sup-floor (v16 SPEC 0.5; never below it) once validation Task 1 bal_p is at least
  the untrained policy's + --t1-trigger-margin (SPEC 0.10) at two consecutive validations (v16; before v16:
  annealed to 0 once validation term_f1 beat the untrained policy's). Its weight w_aux is NOT
  fixed (user, option B): initial value --stop-sup-weight, then tuned by the v4 LLM controller
  (same factors / bounds [0.01, 5] / rollback) from TRAIN statistics (aux loss, aux vs RL gradient
  norm, Task 1 train accuracy); effective weight = max(floor, w_aux x anneal) (v16); --stop-sup-weight 0
  (pure GRPO) requires floor 0, so w_aux = 0 stays off.
  The aux loss is normalised like the GRPO loss: by the number of GENERATED tokens of the
  generations its values belong to (approved 2026-09-26; the v8 smoke measured the aux gradient at
  ~1400x the RL gradient when it was divided by the 1-2 target tokens).
- KL 0.04, LoRA r16. Episodes with a cut R0 reply / lost ledger verdict / emitted capped message
  never enter reward groups.
- Controller: reuse the existing LLM controller adapted to v4 (discrete factors {0.5, 0.8, 1,
  1.25, 2} with bounds — v16: w_dist bounds [1.0, 5.0], so w_dist never drops below its initial 1.0 —, every 5 updates, summarised TRAIN stats only, fixed-weight shadow reward,
  rollback after 2 worse points, local gpt-oss). No novelty needed; NO baselines/control groups now.
- D5: validation Task 2 with the SAMPLED Planner (T 0.7, seeds 0 and 1); Task 1 greedy.
  Checkpoint selection (approved 2026-09-26) = w_sel_cov·validation coverage − w_sel_w1·W1(validation
  simulated turn counts, validation people's turn counts) + w_sel_task1·validation Task 1 bal_p
  (v16; before v16 the Task 1 term was term_f1 (M2), which is still computed and reported but not selected on),
  weights default 1, all logged (summary `selection_task1_metric: "bal_p"`); the v4 validation reward is logged
  but not selected on. Re-selection uses the same validation, so it also selects on bal_p.
  Task 1 validation during training = splits[fold].validation (the 4 sessions; not validation_all).
- D6: fold 2 first.

## Planner generation backend (approved 2026-09-26)
- Planner generation runs on a vLLM server (Qwen3-4B-Instruct-2507, runtime LoRA loading, prefix caching, bound to
  127.0.0.1); Ditto stays on HF. The HF model remains the learner and re-scores every generated token.
- Prompts are built by the one builder (PlannerLM.build_prompt: fit, chat template, ids, budget assert) and sent as
  token ids; the server returns the generated ids (ending with the end token when stopped), the sampled tokens'
  log-probs and the finish reason ("length" = cap hit). Any missing field / length mismatch / stop without an end
  token / prompt + max_new beyond the server context RAISES.
- The served adapter is the checkpoint of the current policy version, loaded under "p<version>-<sha12>"; every
  generation records it; samples from another adapter abort the update. Training rollouts use policy u-1,
  validation after update u uses u. Evaluation CLIs use the same backend (tokenizer only locally).
- Truncated importance sampling: each token's surrogate is weighted by min(pi_old / pi_vllm, 2); the mean
  |log pi_learner - log pi_vllm| is logged per update and the run stops (before the checkpoint) above 0.1
  (phase 0 measured 0.017; HF batch padding 0.007).
- lr = 2e-5 (phase 0: one step at 1e-5 gave per-token KL ~1.1e-3, the low end of the usual 1e-3..1e-2).
- GPU layout: GPU0 = gpt-oss-120b (memory share 0.78) + Planner vLLM (0.15); GPU1 = learner + Ditto.

## Comparison protocol (approved 2026-09-26)
- The 3 outer folds' test_all cover 26 of the 56 finished sessions (not a partition of the corpus). Comparisons
  with E1.6 use E1.6's EXISTING generations re-scored on the same test sessions (no regeneration), paired per
  session with Stage A and Stage B, with bootstrap confidence intervals. Fold 2 is the pilot.

## Data
- splits_v1.json fold 2: train (with requirement shards) 14; train_all 17; validation 4;
  test 5 (Task 2) / test_all 9 (Task 1); forbidden = validation ∪ test. Zero leakage: training,
  p_h, few-shot pool, controller never see validation/test; test only with --final.
- Nothing is launched formally before the user confirms the flow; smoke tests are allowed.
