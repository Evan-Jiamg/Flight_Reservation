### 3.3 Training the Planner with GRPO

Each update runs two kinds of rollouts with the current policy, both sampled from the vLLM-served adapter.

**Task 2 groups.** Four training scenarios are drawn per update, and G = 4 episodes are rolled out for each (Planner temperature 1.0); the episodes of a group share the scenario seed (Speaker sampling, act draws, few-shot examples), so they differ only through the Planner's samples and the task agent. An episode's reward is

  R = w_cov · coverage + w_dist · [log p_h(T) − log q(T)] − Σ_k λ_k · rate_k ,  (1)

where T is the number of user messages (capped at T_max), p_h is the add-α smoothed distribution (α = 1) of real users' message counts over the fold's `train_all`, q is the same smoothed distribution over the current update's clean rollouts, coverage is the fraction of the user's requirements the task agent has addressed by the end of the episode (judged by the ledger LLM), and rate_k are the per-step rates of unparsable plans and plans that hit the generation cap. Because E_q[log p_h − log q] = −KL(q ‖ p_h), the length term is maximised when the *distribution* of session lengths matches the human one, rather than when every session has one length. Episodes with a truncated task-agent reply, a lost ledger verdict or a capped emitted message are excluded from the groups.

**Task 1 stop groups.** Eight `train_all` conversations are drawn per update; in each, the real final message and, when the conversation has n ≥ 3 messages, one earlier message (t ≥ 2) are decision points. At each point the Planner is given the prompt it reaches in a greedy teacher-forced pass of the current policy (the real history plus its own earlier state and notes) and samples G₁ = 8 plans at temperature 1; a plan receives reward 1 if its `end_session` agrees with what the real user did at that point, and 0 otherwise (also for invalid plans). If fewer groups than drawn have non-zero reward variance, further unused `train_all` conversations are drawn (at most eight more) until the number of informative groups is restored.

**Advantages and credit.** Within a group we use the group-mean baseline without standard-deviation normalisation (Dr. GRPO): A_i = R_i − mean_j R_j; groups with zero reward variance are skipped. For Task 2 the advantage is split into the part caused by the length term, S_i = w_dist · [log p_h(T_i) − log q(T_i)], and the rest:

  A_i^stop = S_i − mean_j S_j ,  A_i^seq = (R_i − S_i) − mean_j (R_j − S_j) .  (2)

A_i^seq is applied to every generated token of every plan in the episode except the `profile_note` tokens, and A_i^stop is added only on the tokens of the `end_session` value at real decision points (t ≥ 2, parsed, not capped). For Task 1 groups the advantage acts only on the `end_session` value tokens.

**Objective.** With importance ratio ρ = π_θ / π_old per token, the loss for a minibatch of plans is

  L = (1/N) Σ_tokens [ −w · min(ρ·A, clip(ρ, 1±0.2)·A) + β · (e^{d} − d − 1) ] + L_aux ,  d = log π_ref − log π_θ ,  (3)

where N is the number of generated tokens, β = 0.04 is the weight of the k3 KL estimator to the starting policy π_ref, and w = min(π_old / π_vLLM, 2) is a truncated importance weight that corrects for sampling with the vLLM server; the update is aborted if the mean |log π_old − log π_vLLM| exceeds 0.1. The auxiliary term

  L_aux = −(w_eff / N_aux) Σ log p_θ(y* | prompt, own prefix)  (4)

supervises the `end_session` value y* of the real user at the Task 1 decision points of the base groups (N_aux is the number of generated tokens of those plans). Its weight is w_eff = max(0.5, w_aux · a_u), where a_u = 1 until the balanced end probability (§3.4) is at least the untrained policy's + 0.10 at two consecutive validations, and then a_u = max(0.5, 1 − (u − u_trig)/10); w_aux starts at 1 and is tuned by the controller, so the supervision never drops below 0.5.

**Weight controller.** Every five updates an LLM controller (gpt-oss-120b) reads summary statistics of the *training* rollouts and may multiply each of w_cov, w_dist, λ_unparsed, λ_hit_max_new and w_aux by a factor in {0.5, 0.8, 1, 1.25, 2} within fixed bounds (w_dist may only be raised from 1); a change is rolled back if a fixed-weight shadow reward gets worse over two windows.

**Setup.** Initial weights w_cov = w_dist = λ_unparsed = λ_hit_max_new = 1. LoRA rank 16 (α = 32) on the attention projections; π_ref is the base model without the adapter (equal to the starting policy); AdamW (no weight decay), learning rate 2e-5, one optimizer step per update with gradient-norm clipping at 1.0; the HF model is the learner and re-scores every token the vLLM server generated.

### 3.4 Validation and checkpoint selection

The untrained policy (update 0) and every fifth update are validated on the fold's validation conversations: Task 2 with the sampled Planner (temperature 0.7, seeds 0 and 1) and Task 1 greedily. For Task 1 we compute, at every real decision point t, the teacher-forced probability that the Planner ends the session, P_end = p(true) / (p(true) + p(false)) for the `end_session` value given the greedy plan's prefix, and the balanced score

  bal_p = ½ · mean_{t = n} P_end + ½ · mean_{2 ≤ t < n} (1 − P_end) ,  (5)

which rewards ending at the real last message and continuing before it (invalid points count as P_end = 0; a valid decision whose value tokens cannot be located counts as its greedy decision). The checkpoint score is

  score = coverage − W1(simulated turns, validation users' turns) + bal_p ,  (6)

and training stops after two validations without improvement (at most 30 updates). A validation with an episode that stays unclean after two re-runs gets no score. Because the validation splits are tiny (four conversations in fold 2, one in fold 0), the validated checkpoints are re-validated with eight seeds (0–7) and the best-scoring one is chosen (ties go to the earlier update); the chosen checkpoint and the untrained policy are then evaluated once on the test split.

> 註：
> - 式 (1)–(6) 對應：`rl_reward.reward_v4`、`turn_distribution`；`rl_algos.split_group_advantages`（std_norm False）；`TorchLearner.update`（clipped surrogate × TIS、k3 KL、除以生成 token 數、aux_backward）；`train_planner_rl.aux_weight / aux_annealed`；`task1_stop.task1_prob_metrics`；`validate()` 的 selection。
> - clip：每次 update 只做 1 epoch × 1 minibatch，所以第一步的 ratio 恆為 1（程式有斷言），clip 在實際上不作用；式 (3) 仍照程式寫出。
> - 「連續兩次驗證沒有進步就停、最多 30 次 update」與「8 seeds 重新驗證」是實驗腳本（`ops/v11ops/run_v16_formal.sh`、`run_v16_reselect.sh`）的規則，不是 trainer 內建；「8 seeds」是使用者 2026-09-27 核准的。fold 2 已完成重選（u0/u5/u10 → u5）；fold 0/1 的腳本（`run_v16_fold.sh`）同樣對所有驗證過的 checkpoint 做 8-seed 重選。fold 0 的 validation 只有 1 段對話，如何挑 checkpoint 使用者尚未決定。
> - coverage 是 reward 的一項（D3，已揭露）：因此訓練後的 coverage 不能當成獨立的評估指標，報告時要說明。
> - Dr. GRPO 需要引文（TODO）。

---

