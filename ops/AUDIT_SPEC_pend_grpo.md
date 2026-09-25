# pend + GRPO — the approved design (audit reference, 2026-09-25)

Everything below was decided or approved by the user. An audit checks the code in
`sep-sim/` against THIS list: anything missing, wired differently, or still following an
older design is a finding. Anything the code does that is NOT on this list and changes
behaviour is also a finding ("unauthorised design change").

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
  k = 3, COPY_NGRAM = 8, duplicate redraw rounds = 4.

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
- Stop credit: the length term's group advantage goes only to the end_session value tokens; the
  rest keeps the sequence-level advantage. stop_mask only on real decisions (t ≥ 2, parsed,
  valid end_session, not capped).
- D4: profile_note tokens carry no sequence advantage (KL only).
- Task 1 stop groups on train_all conversations (last message + one earlier, t ≥ 2), G samples,
  reward 1 if end_session agrees with the real person; non-decisions reward 0, no stop mask.
- D2: auxiliary stop-token supervision on the Task 1 positions, annealed linearly to 0 (over 10
  updates) once validation Task 1 term_f1 beats the untrained policy's. Its weight w_aux is NOT
  fixed (user, option B): initial value --stop-sup-weight, then tuned by the v4 LLM controller
  (same factors / bounds [0.01, 5] / rollback) from TRAIN statistics (aux loss, aux vs RL gradient
  norm, Task 1 train accuracy); effective weight = w_aux x anneal; w_aux = 0 stays off.
- KL 0.04, LoRA r16. Episodes with a cut R0 reply / lost ledger verdict / emitted capped message
  never enter reward groups.
- Controller: reuse the existing LLM controller adapted to v4 (discrete factors {0.5, 0.8, 1,
  1.25, 2} with bounds, every 5 updates, summarised TRAIN stats only, fixed-weight shadow reward,
  rollback after 2 worse points, local gpt-oss). No novelty needed; NO baselines/control groups now.
- D5: validation Task 2 with the SAMPLED Planner (T 0.7, seeds 0 and 1); Task 1 greedy.
  Selection = validation reward (fixed cfg0) + w·validation Task 1 term_f1. Turn W1 reported.
- D6: fold 2 first.

## Data
- splits_v1.json fold 2: train (with requirement shards) 14; train_all 17; validation 4;
  test 5 (Task 2) / test_all 9 (Task 1); forbidden = validation ∪ test. Zero leakage: training,
  p_h, few-shot pool, controller never see validation/test; test only with --final.
- Nothing is launched formally before the user confirms the flow; smoke tests are allowed.
