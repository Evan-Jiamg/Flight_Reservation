# v17 implementation checklist (from the pre-implementation gap check, 2026-09-30)

Paths without a prefix are under `sep-sim/`. Line numbers are those of the v16 code (commit ebb30b7). The decisions for B1–B10, S1–S15 and N1–N5 are in `ops/SPEC_v17_sft_pend.md` §13. Every item is mandatory.

## rl_algos.py
1. `ALGO_DEFAULTS` (45): set epochs 2 and minibatches 4.
2. `setup_policy` (221-241):
   - seed before `get_peft_model` (B1);
   - refuse `init_adapter`.
3. New `load_ref_adapter(model, path)`:
   - call `load_adapter(..., adapter_name="ref", is_trainable=False)`;
   - assert the trainable set and the optimizer param set are unchanged (B4).
4. `TorchLearner._logp` (322-329), ref branch:
   - `set_adapter("ref")`, then forward, then `set_adapter("default")`, inside `try/finally`;
   - assert the trainables are unchanged and the ref params have `requires_grad` False.
5. `TorchLearner.save` (518-524): `save_pretrained(..., selected_adapters=["default"])`.
6. New module function `value_nll_loss(model, examples, norm_tokens)`, extracted from `aux_backward` (385-409). SFT and aux both use it.
7. `TorchLearner.update` (353-515):
   - per-token advantage = `adv·(1−note)` + `adv_prefix·prefix_mask·(1−note)` + `adv_stop·stop_mask` (S2);
   - split aux across the minibatches of each epoch, normalised by each minibatch's aux gen_len (S4);
   - measure `aux_p_correct_before` before step 1 (no-grad);
   - log per-step mean and max of `rl_grad_norm`, `aux_grad_norm`, kl, `clip_frac`;
   - record `optimizer_steps`;
   - handle n < 4 samples and aux-only updates (S3).
8. `episode_samples` (162-178): carry `prefix_mask`.
9. `group_advantages` / `advantages_for_groups` (82-91, 181-200): skip groups with fewer than 2 samples (B7). Callers count them.
10. New `TorchLearner.p_end_batch(items)`: reuses `end_prob` (526-536); no-grad, eval mode.

## train_planner_rl.py
11. `CODE_FILES` (56-58): add any new runtime helper modules.
12. `RESUME_MAY_CHANGE` (62-64): remove `updates`, `reselect_seeds` and `reselect_updates`.
13. `load_split` (139-154):
    - return `validation_all`;
    - assert it is inside forbidden, disjoint from train and train_all, and contains validation.
14. `FakeLearner` (182-285): add multi-step, the aux split, `prefix_mask` advantage, a ref stub, `p_end_batch` and an SFT step.
15. `FakeEnv` and the `_fake_task1_*` helpers (287-387): add per-sample `decision_valid`, `target_true`/`target_false`/`prefix_ids`, a mask-mismatch case, an invalid case, and `human_turns > 10` cases.
16. `Trainer.__init__` (403-440):
    - `cfg0` gets lr 1e-5, kl 0.01, and `w_aux` from `--aux-weight`;
    - fixed controller; remove the `aux_floor` ctl_opts;
    - remove `aux_anneal_start` / D2 and `best`; keep `task1_base` (set at u0);
    - add `config_record["spec_version"] = "v17"`.
17. `build` (443-475):
    - remove the `--init-adapter` path;
    - load ref if u0 exists (after the learner/optimizer);
    - `describe()` with the new speaker fields.
18. New `sft_stage()`, implementing §1 in full:
    - examples cache and meta;
    - `base_pend_train.jsonl`;
    - a separate AdamW;
    - epoch candidates `ckpt/sft_e<k>` (k=0 is the start policy);
    - validation_all probes into `sft.jsonl`;
    - pick (nll, then n_invalid, then the earlier epoch), then `load_policy` and a sha check, then save u0, then load ref.
19. `sync_generation_policy` (654-662): accept a path and a tag (B3).
20. `aux_weight` / `aux_annealed` (495-512): replace with the constant `a.aux_weight`.
21. `meta` / `manifest` / `check_provenance` (550-575, 607-617):
    - add SFT args, the sft_examples sha, the u0 sha, the chosen epoch and spec_version;
    - `init_adapter` is asserted None.
22. `save_checkpoint` / `load_checkpoint` (580-643):
    - remove `best` and `aux_anneal_start`;
    - keep `task1_base`;
    - add `stop_reason`.
23. `task1_rollouts` (712-812):
    - positions become `range(2, n+1)` (742);
    - remove the refill (779-811) and the per-conversation x draw (735);
    - compute P_end and Brier under `gpu_lock`, recording `p_end`, `reward`, `dropped` and the reason per sample;
    - no `t1_refill` record.
24. `task1_samples` (814-834):
    - group only the non-dropped samples, skipping groups with fewer than 2 (B7) and all-invalid groups;
    - valid samples get `adv_prefix` / `prefix_mask`; invalid samples get `adv` (all tokens × (1−note));
    - bypass `stop_credit`.
25. `aux_examples` (836-848): one example per decision point, no refill filter, weight `a.aux_weight`.
26. `one_update` (876-1012):
    - `t1_hist` fields (S15) and `adv_abs` by source;
    - reward std per source, `kl_q_ph`, `drift_stat`;
    - remove the `aux_floor` / `aux_annealed` keys.
27. `validate` (1015-1151):
    - add the `task2` flag;
    - Task 1 on validation_all;
    - remove selection, best, D2 and reselect;
    - set `task1_base` at u0;
    - add `nll` per B5;
    - the summary key includes `task2` (S6).
28. Delete `reselect` (1154-1199); delete the reselect branch in `main` (1404-1405).
29. `run` (1202-1225):
    - SFT, then `validate(0, task2=True)`;
    - loop over updates with the drift check;
    - write `final.json`, then `validate(final, task2=True)`;
    - resume guards (B2, S6, S7).
30. `parse_args` (1244-1399):
    - add `--sft-lr`, `--sft-epochs-max`, `--sft-samples-per-point`, `--task1-reward`, `--task1-positions`, `--aux-weight`, `--length-drift-margin`;
    - remove `--stop-sup-*`, `--t1-trigger-margin`, `--reselect-*`, `--w-sel-*` and `--init-adapter`; `--task1-tol` is report-only;
    - gate values: task1_G 4, task1_convs 8, val_seeds 0..7, val_every 1, updates 5, controller fixed, lr 1e-5, kl 0.01, stop_credit 1, epochs 2, minibatches 4, sft_lr 5e-5, sft_epochs_max 3, sft_samples_per_point 2, aux_weight 0.5, length_drift_margin 1.0.

## task2_env.py
31. `task1_sample` (1282-1331): each valid-with-mask sample emits `target_true`, `target_false`, `prefix_ids` and `mask_ok`. The env's 0/1 reward is renamed `correct`.
32. `describe` (742-771): add `speaker_path`, `speaker_class` and `speaker_endconv_is_none` (S12).

## Other code
33. `rl_controllers.py`: leave `TRAIN_DEFAULTS` unchanged (S10). Fix the prompt text at 316-321 only if the llm controller remains usable.
34. `verify_pipeline.py`:
    - v17 branch keyed on `spec_version`; `check_v16` stays for old runs (B8);
    - `check_rl_selection` (1239, 1250) handles validation_all;
    - no best/reselect requirement for v17 (1092, 1103-1119, 1142-1170, 1199-1254);
    - Brier recompute (1052-1060);
    - `adapter_name()` for `sft_e<k>` (658-665);
    - all the §9 checks: SFT data and choice, no `ref/`, `prefix_mask`, aux count, steps, drift, validation schedule, test set, Ditto fields.
35. `eval_test_rl.py` (37, 148-156, 171-173) and `eval_test_boot.py` (47-49):
    - use `final.json`; updates are {0, final};
    - add `--include-base` (serves `ckpt/sft_e0`) (B6).
36. New `threshold_control.py`: pure CPU, as specified in B6.
37. New `smoke_v17.py` (S14).

## Snapshot, ops, tests
38. Snapshot: new `ops/local_sha_v17.txt` covering every runtime file, plus `eval_test_*`, `threshold_control.py`, `verify_pipeline.py` and the tests.
39. ops: new `ops/v11ops/run_v17_fold.sh` and `run_v17_test.sh`.
    - Model them on `run_v16_fold.sh` (keep the launch at 40-41; drop the selection/reselect at 53-103) and `run_v16_test.sh` (at 43, read the final update from `final.json`).
    - The v16-only scripts (`run_v16_reselect.sh`, `reselect_boot.py`, `watch_reselect16.sh`, `monitor_reselect16.sh`) are not used.
40. Tests to rewrite or pin:
    - `test_rl_advantages.py`: `args()` 124-127, 175-187, 208, 238-252;
    - `test_v16.py`, `test_v16b.py`, `test_v16c.py`, `test_reselect.py`: the whole files (delete, or convert to verify fixtures);
    - `test_eval_test_rl.py`: 18, 26-31, 47, 140, 147-148, 158;
    - `test_pend.py`: 95-110, 157, 167, 168;
    - `test_audit4.py`: 15-30;
    - `test_verify_pipeline.py`: 374-376, 443-446;
    - `test_intervention.py`: 67-83;
    - `test_rl_algos_server.py`: 185-198 (add a `prefix_mask` test and a ref-adapter test);
    - `test_rl_controllers.py`: 102, only if S10 goes the other way.
41. New tests, one per §10 item, plus:
    - groups with fewer than 2 samples (B7);
    - `updates` cannot change on resume, and a drift-stopped run is not resumable (B2);
    - a non-resume launch is refused over SFT leftovers (S7);
    - `human_turns > 10` in the drift rule (S5);
    - the saved adapter has no `ref/` and its sha is unchanged (B4);
    - a Task-1-only summary does not satisfy the final validation (S6).
