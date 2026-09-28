# Claims made to the user (2026-09-25/26; v16 updates 2026-09-28 per ops/SPEC_v16_grpo_opt.md) — each must be ACTUALLY implemented (say = do)

Verify every item in the code of `sep-sim/` (HEAD). For each: file:line that implements it, whether a
test covers it (test name) and whether it is really exercised (not dead code / not bypassed), or
NOT IMPLEMENTED / PARTIAL / DIFFERENT (explain).

1. Task 1 M2: row t greedy_ended = Speaker blank at t OR Planner end at t-1 (row 1 blank only);
   samples_ended likewise; K+1 ended = Planner end at n OR Speaker blank at K+1; raw decisions kept
   (planner_ends_session, ended_planner_k1, ended_by_prev_decision, ended_by_decision_at_n).
2. stop_mask only on real decisions (t >= 2, parsed, valid end_session, not capped, not turn-1-ignored)
   and dropped when the masked tokens do not spell the parsed value (stop_mask_mismatch).
3. note_mask: profile_note value tokens get no sequence advantage (KL only) in TorchLearner.update (D4).
4. A Planner output that hits max_new is treated as unparsed and is NOT read at all (no stopping-ledger
   gain, no act-RNG draw) in Task 2, Task 1 and task1_sample.
5. Implicit Profile entries = "(message t-1) measured: ... | note"; the measured diff is Task 1 only;
   exact repeat of the last note skipped; notes never cut/dropped; unparsed turns still pass notes and
   examples to the Speaker.
6. Complete redraw when not ending: from non-Complete entries with p > 0 (Other allowed); the Complete
   length is dropped when the new entry has none; no alternative -> complete_kept_no_alternative,
   verify WARN.
7. LAST_MESSAGE_LINE only in the Speaker block, never in the Planner's state (S["block"]/prev_block);
   verify pend.state_clean catches it.
8. No-survivor fallback never emits a capped or blank candidate when a whole non-blank one exists;
   emitted_capped recorded; such an episode is unclean.
9. Per-episode thread-local counters (r0_len_retries/truncated/ctx_fit/empty, judge_empty/unparseable/
   retries/retry_failed); orphan incidents (counted outside an episode thread) make run_episode raise;
   ledger judge re-requests an unparseable answer once at 8000 tokens and a failing retry never
   crashes; a final empty R0 reply is counted; `clean` flag; unclean episodes excluded from reward
   groups and from q; groups left with 1 episode counted (n_dropped_singleton_episodes).
10. Compaction: Task 2 step with a compacted Planner/Speaker prompt -> unclean; Task 1 -> raises
    (task1_generate and task1_sample); verify FAILs compaction for pend.
11. Trainer: --arm pend only; spec defaults implicit_profile 1 / fewshot fold / selector borda /
    controller llm / reward v4 / Planner path containing Qwen3-4B-Instruct-2507; any deviation needs
    --ablation.
12. Reward v4 default for pend; p_h from splits[fold].train_all only (asserted not forbidden); q per
    update from clean episodes; a capped plan pays lambda_hit_max_new only (not also lambda_unparsed).
13. Stop credit: one centring per group (split_group_advantages); A_seq + A_stop = plain GRPO A. v16 (Dr. GRPO,
    grpo_std_norm False): A = R - mean R, A_stop = S - mean S, A_seq = (R - S) - mean(R - S), no division by the
    group std, in every call site (Task 2 with/without stop credit, Task 1); groups with std <= min_group_std are
    still skipped; verify rl.adv_norm (algo_cfg.grpo_std_norm is False, learner_stats.adv_abs_mean logged).
14. Shadow reward (fixed cfg0), turn_hist and p_h in the train history; v4 LLMFactorController for
    --controller llm with v4; rollback after 2 worse decision points survives a change in between;
    a reply cut by max_tokens is a failure; finish_reason logged; controller_failures/rollbacks in
    updates.jsonl.
15. w_aux (stop-supervision weight) is a controller knob: bounds [0.01, 5], same factors, 0 stays 0;
    initial --stop-sup-weight; effective weight = max(--stop-sup-floor, w_aux x D2 anneal) (v16, floor SPEC 0.5;
    --stop-sup-weight 0 requires floor 0; hist logs aux_floor / aux_floor_active; verify rl.aux_floor); the
    controller request shows aux stats (aux loss, aux_grad_norm vs grad_norm, Task 1 train accuracy) and its
    system prompt names the floor. v16: w_dist controller bounds [1.0, 5.0] (verify rl.w_dist_floor).
16. Aux loss normalised by the generated tokens of its generations (gen_len), like the GRPO loss.
17. D2 (v16): aux_anneal_start set at the second of two consecutive validations whose Task 1 bal_p >=
    task1_base.bal_p + --t1-trigger-margin (SPEC 0.10) (before v16: when term_f1 first beat task1_base); the anneal
    then goes towards the floor, not to 0; summary logs d2 {met, streak, margin, base_bal_p, triggered_at};
    persisted in the checkpoint state; task1_base persisted immediately when first computed; verify rl.d2_trigger.
18. D5: validation Task 2 with the sampled Planner (--val-temperature 0.7, seeds 0 and 1, replicate 0);
    Task 1 greedy; turn W1; unclean validation episodes excluded; validation ids disjoint from
    train/train_all.
19. Checkpoint selection = w_sel_cov*coverage_mean - w_sel_w1*turn_w1 + w_sel_task1*bal_p (v16; term_f1 still
    reported, not selected on; summary selection_task1_metric "bal_p"); weights are CLI args and logged;
    best.json written only when a score exists; verify rl.selection recomputes with selection_task1_metric
    (term_f1 for pre-v16 runs); bal_p recomputable from the validation rows' end_probs (verify rl.task1_prob).
20. rl_manifest.json in every checkpoint (fold, splits sha, train_scenarios, train_conversations,
    fewshot_pool, p_h source, settings, init_adapter); task1_v4 / rollout_v4 find it next to the
    adapter, refuse an init adapter, refuse different settings, a different split sha, forbidden ids.
21. Resume: provenance over all args except a whitelist; read_jsonl tolerates only a torn LAST line;
    Task 1 groups resumed per (conversation, t).
22. task1_v4: spec defaults (else --ablation); fold-test scores test_all; meta has settings,
    session_ids, limit; resume refuses changed settings/code. verify_task1: M2 checks, compaction,
    capped emission, missing sessions, scored ids = the split (limit-aware), prompt tokens required,
    few-shot examples without goal/persona ids FAIL.
23. rollout_v4: pend spec defaults incl. Planner temperature 0.7 (else --ablation); done keyed with
    replicate; torn last line tolerated; manifest split sha and forbidden checks.
24. pend requires R0/JUDGE base URLs set (not api.openai.com) and models gpt-oss-120b unless
    PEND_ALLOW_OTHER_ENDPOINTS=1.
25. FewShotPool raises when a pool conversation has no goal/persona id.
26. verify_pipeline: zero-turn episode FAIL; clean flag consistent with counters; stop-mask gating and
    mask lengths; validation ids; best.json; manifests; validation episodes through structure/
    truncation/pend checks; Task 2 few-shot leakage; evaluation episodes must be clean.
27. load_split asserts the declared split sizes.
28. Timing: per step planner_s, speaker_s, r0_s, ledger_s; per update timing (task2_rollouts_s,
    task1_groups_s, learner_s); validation_s.
29. Task 1 stop groups only at t >= 2; non-decision samples reward 0, no stop mask, never the aux example.
    v16: --task1-G (SPEC 8) samples per group, --task1-convs (SPEC 8) base conversations, plus dynamic refill from
    unused train_all conversations (<= --task1-convs, rows marked "refill", aux examples from base groups only);
    verify rl.task1_G and rl.task1_refill.
30. v16 task1_pooled.py: Task 1 pooled over folds (duplicate ids within an arm raise), paired per-conversation
    bootstrap between two arms with identical conversation sets (10000, seed 0), input sha256 recorded.
