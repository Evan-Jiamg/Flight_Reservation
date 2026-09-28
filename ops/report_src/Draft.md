# Draft — Introduction, Related Work & Methodology (rev. 4)

> 草稿 rev.4（2026-09-28）：Intro／Related Work 經 reviewer 第 1–4 輪 ACCEPT；新增 §3 Methodology。英文正文供之後貼進 `scai_sigconf/body.tex`；`> 註` 是給作者的說明，定稿時刪除。
> **規則**：所有數字與文獻都來自文末「Sources」，文獻只照來源抄，缺的欄位標 `TODO`，定稿前請逐條對原文確認。
> 版面：模板是 2 頁 extended abstract（Intro 約 250 words、沒有 Related Work 節）。本稿 Intro 約 690 words（含小標與 TODO）、Related Work 約 560 words，
> 放進模板時：Intro 刪到 Context 2 句 / Gap 3 句 / Approach 2 句；Related Work 併成 Intro 裡約 80 words 的一段（見 §2 末的濃縮版）。

---

## 1. Introduction

**Context.** User simulators are increasingly used to evaluate and train conversational information-access systems [Balog & Zhai 2024]. The TREC 2026 User Simulation Track targets *conversational data search*, where a researcher looks for a relevant dataset through a search interface [Kreutz et al. 2025]. A simulator either predicts the next user utterance given a partial conversation history and the user's information need (Task 1), or generates a whole conversation and must itself decide when the goal is satisfied or when to give up (Task 2). The track compares the distributions of query length, turn count and clarification requests with real logs, alongside a human Turing test.

**The gap.** We call a simulator *session-level faithful* when its per-turn decisions to stop match those of real users — *when* to stop, not only *whether* — and, as a consequence, its distribution of session lengths matches theirs. Recent LLM simulators have become strong at turn-level fidelity (intent adherence, persona consistency, style) [Naous et al. 2026; Abdulhai et al. 2025; Wang et al. 2025], but on the track's data this does not carry over to the session level. In our lab's internal benchmark (not the official evaluation), a re-implementation of the Turing-RL recipe [Wang et al. 2026a] whose act transitions are within the human noise floor and whose act distribution is, with one other system, the closest to human in the benchmark (act TVD 0.185) never ends a replayed human session when offered one more turn (0/26 sessions) and runs 9.9 user turns against a human mean of 4.4; the public UserLM-8b [Naous et al. 2026], prompted with its native end token, instead emits END at 72% of the real mid-session turns where the human continued, and its sessions last 1.04 turns; zero-shot Ditto-8B comes close to the human mean (5.0 turns). Act-level fidelity thus does not predict session behaviour: one of the two simulators closest to human act distributions never stops, while Ditto-8B, further from them (act TVD 0.305), is close to the human session length. Stopping is also sensitive to the probe setup: a newer probe that adds one shared termination instruction (together with an updated system agent) moved Ditto-8B's K+1 end rate from 0.04 to 0.48. Existing RL rewards for simulators are per-utterance judgements or, in USP, one dialogue-level profile-similarity score; none compares a simulator's session lengths or stop positions with those of real users.

**Our approach.** We use a Planner–Speaker simulator in which ending the session is a separate, binding decision of the Planner (Qwen3-4B-Instruct-2507); a frozen Speaker (Ditto-8B, which has no end-of-conversation token; a blank Speaker message also ends the episode, by the benchmark convention) writes candidates that a length-and-style selector ranks. Two training-free versions of this design show why stopping must be learned. In the first, the Speaker ignored the Planner's decision to stop (asked to close, it closed 8% of the time) and 51 of 64 sessions hit the 10-turn cap. A second version, which among other changes makes the Planner's decision binding, does close (K+1 end rate 0.05 → 0.48 against the first version without annotations) but ends too early (premature end rate 0.03 → 0.14). We therefore train the Planner with a GRPO variant (group-mean baseline without standard-deviation normalisation) on session-level rewards: a log-ratio term that moves the simulated turn-count distribution towards real users' turn counts in the training fold, LLM-judged requirement coverage, and agreement with the real user's stop decision at real decision points of training conversations. The advantage of the length term is credited only to the Planner's end-session tokens, and an auxiliary supervised loss on the same stop decisions (weight floor 0.5) stabilises training.

> 註：下一段等 fold 2 test（u5 vs u0）與其他 fold 跑完再填；目前沒有任何可引用的 v16 結果。比較協定（AUDIT_SPEC）是對 E1.6 在同一批 test sessions 上重新計分，**目前沒有核准其他 baseline**；若要與 benchmark 其他方法同表，須先證明我們 Task 2 環境（自架 R0／ledger judge）與 benchmark 的 system agent 設定（prompt v4、length_retry_v1）一致。輪數要寫清楚是 W1 還是 benchmark 的平均輪數（錨 4.446），或兩個都報。

**Findings.** `TODO` (from the fold-2 test and the three-fold pooled evaluation; paired bootstrap intervals, n stated.)

**Contributions.**
- We show, on the track's data, that act-level fidelity does not predict session-level fidelity: session behaviour ranges from never ending to ending almost immediately, independently of act fidelity, and is sensitive to the prompt and probe setup.
- We make ending the session a separate, binding Planner decision and train it with GRPO against real users' session lengths and stop positions, with credit assigned to the end-session decision, instead of per-utterance rewards.
- `TODO` (result, on goal- and persona-disjoint test sessions, compared with our training-free version under the same protocol.)

> 註：
> 1. benchmark 是實驗室內部量測工具，NOTICE 說它「不是可引用的出版物；請引用 Track 與原始論文」。致謝 Lucas H.-C. Hsu（Sep-1st README §7），**不要引用 repo**。
> 2. Turing-RL 的數字：probe v2、三個 fold test side 共 26 個 session（K+1 0/26、rollout 自己結束 1/26、平均 9.885 輪），`instruments/termination_probe_v2/README.md` 與 `leaderboard.md`。真人雜訊地板：act TVD（F1）0.155、transition JSD（F2）0.19——Turing-RL 的 transition JSD 0.121 在地板內，act TVD 0.185 **在地板之上**（但與 A1-s1 的 0.182 並列最接近真人）。Ditto 的提示敏感度：加一條共用 TERMINATION_INSTRUCTION 讓全語料 K+1 從 0.0357 變 0.4821（probe README 49-52；v1 的 system agent 也是舊設定，所以不能全歸功於那條指示）。它的「結束」在自己的格式裡是空訊息，所以「不結束」可能部分來自重現方式——正文已寫 re-implementation，必要時再加一句 hedge。
> 3. UserLM-8b：**未入榜**，用它原生的 end token、沒有共用的 TERMINATION_INSTRUCTION（probe README 74-76）；72% = `teacher_forced_turn_with_end_decision_rate`（分母是所有真人繼續的中間輪）。Ditto-8B：K+1 0.577、4.962 輪，是 F10 family 最佳，而且就是我們的凍結 Speaker——reviewer 會問「為何不直接用 Ditto」，答案要在 Results 用「停在哪一輪」（premature、stop AUC）而不只是平均輪數來回答。
> 4. 兩個免訓練版本的數字是**我們自己在 benchmark 較早協定下**的量測（v2fix：Qwen2.5-32B planner + UserLM-8b speaker，64 episodes；E1.6：K+1 0.0536→0.4821、premature 0.0321→0.1446，出自 v2fix_to_E1.6 投影片 p.4-5），**不可和上面 probe v2 的數字並列成同一張表**。8% 與 51/64 屬於 v2fix；0.0536 屬於 v2fix 去掉 annotations 的版本（原 v2fix 是 0.0179）；v2fix → E1.6 改了五件事（annotations、stop 權限、speaker 輸入、role header、開場取樣），所以正文寫「among other changes」。兩版用的模型也不同（v2fix：Qwen2.5-32B planner＋UserLM-8b speaker；E1.6 的 speaker 來源未寫明，請確認）。
> 5. 方法描述對應 AUDIT_SPEC / SPEC_v16：Task 1 stop groups 用 train_all 對話（不是 held-out）；validation 只用來挑 checkpoint（bal_p）；coverage 由 gpt-oss-120b ledger judge 判斷；Dr. GRPO 需要引文（`TODO cite`，來源裡沒有書目）。
> 6. 實驗室組員的 Unified Framework 投影片提出過 user 端的 termination reward（λ4·r_term），A1-s1 是他在同一 benchmark 的 RL 模擬器。**我們不宣稱「第一個獎勵 user 端結束」**，新穎性放在「對真人長度分佈與真人結束位置做最佳化、並把 credit 給結束決策」。若要正面比較，請確認 A1-s1 是否用了 r_term。
> 7. 官方 Task 1 是 next-utterance prediction；我們自己的 Task 1 結束決策指標（M2 mapping, decision D7）不是官方指標；benchmark 的 `termination_f1` 已於 9/26 退役。

---

## 2. Related Work

**LLM user simulators.** User simulation evolved from agenda-based [Schatzmann et al. 2007] and neural sequence models [El Asri et al. 2016; Kreyssig et al. 2018] to LLMs trained to play the user [Balog & Zhai 2024]. USP [Wang et al. 2025] conditions generation on an implicit profile extracted from each dialogue (our Planner keeps a similar implicit profile); UserLM-8b [Naous et al. 2026] flips assistant dialogues to train a user model with an end-of-conversation token, and documents prompted assistants' reluctance to end a session; HumanLM [Wu et al. 2026] aligns latent user states; MUSE [Liu et al. 2026] optimises a profile by iterative self-critique against real dialogues; ProUtt [Wang et al. 2026b] predicts the user's next intent path. Most are evaluated at the turn level; UserLM also scores termination, learning to stop by imitating real ends and needing a guardrail against ending too early.

**RL for user simulators.** ConsistentPersona [Abdulhai et al. 2025] applies multi-turn PPO with a judge-scored persona-consistency reward; USP's RLCC stage [Wang et al. 2025] rewards one dialogue-level profile-similarity (cycle-consistency) score, repeated over the user turns, plus human-likeness; UserLM-R1 [Zhang et al. 2026] combines rule and rubric rewards with GRPO and judges "timing of hanging up" with an LLM; Turing-RL [Wang et al. 2026a] uses a pairwise Turing-style judge as the GRPO reward; MUSE averages turn-level rubric rewards over the session under GRPO; DITTO [Sun et al. 2026] adds verbal feedback to GRPO. Session length is either fixed (ConsistentPersona's 10, 20, 40 or 60 turns), capped (USP, up to 10 turns) or judged by an LLM; none of these rewards compares session lengths or stop positions with real users. Steering a frozen generator with a small trained planner has precedents — Dialogue Action Tokens [Li et al. 2024], PPDPP [Deng et al. 2024] and EPO (`TODO` authors, ACL 2025) — but none of them trains a *user's* decision to stop: PPDPP plans the system's moves, drops CraigslistBargain's terminal acts and caps dialogues at 8 turns, and DAT names leaving the chat as an open direction. We therefore do not claim the planner–generator split as new, only its use for a user whose stop decision is trained.

**Evaluation and stopping.** Sim4IA-Bench [Kruff et al. 2026a] scores next-query and next-utterance prediction on real search sessions, but every task is single-step; Bernard & Balog [2024] formalise simulation objectives in conversational information access; Kruff et al. [2026b] propose a taxonomy of measures for validating query simulations; clem:todd [Chalamalasetti et al. 2025] benchmarks simulator × dialogue-system combinations, and SimEval-IR [Zerhoudi 2026] separates behavioural realism from tester reliability. Zhou et al. [2026] find that simulated users are more cooperative and disclose task information earlier than real ones, and recommend reporting behaviour, task outcome and subjective ratings separately — a session-level realism gap. In interactive IR, when users stop has long been modelled with stopping rules [Cooper 1973; Kraft & Lee 1979; Maxwell et al. 2015] and foraging theory [Charnov 1976; Pirolli & Card 1999]; LLM simulators learn to stop only by imitation (UserLM's end token) or through LLM-judged rubrics, and none is optimised against real users' session-length distribution. Finally, conversational search systems are increasingly optimised against learned or interactive feedback — reward-model reranking for query reformulation [Lai et al. 2025] and GRPO-trained agentic search [Mo et al. 2026] — so realistic session endings matter for training and evaluating them.

> 註：
> - 作者先前在 Sep-1st 試過具名 stopping rules，**沒贏過單純的輪數計數器**（F1 0.442 vs 0.524/0.559，Sep-1st README）；benchmark 的 stop judgement 也顯示 `rule_turn_count` AUC 0.809。Results 要把「停在哪一輪」和這個 turn-count / hazard 基線比（不只和 E1.6 比），reviewer 會要求。
> - UserLM 的 termination F1（論文 63.54）只出自組員轉述，要引請回原論文確認。
> - UserRL [Qian 2025]、UserSimCRS v2 [Bernard & Balog 2026]、Chopra [2026]《Beyond Cooperative Simulators》在來源裡**只有標題與 venue**，沒有內容描述，所以**已從正文移除**；讀過原文、能寫出一句正確描述後再放回（Chopra 可能與 Zhou et al. 並列，UserRL 可能放 RL 段）。
> - 空白的 Speaker 輸出也會結束 episode（AUDIT_SPEC），Method 節要寫清楚，避免讀者以為只有 Planner 能結束。
> - UnifiedFramework 說「termination reward 在 task-oriented dialogue RL 是標準做法（訓練 system 端）」但沒給出處——**不寫進正文**。

**Condensed version for the 2-page template (~80 words).** User simulators learn turn-level fidelity via SFT [Naous et al. 2026] or RL with per-utterance, rubric or dialogue-level similarity rewards [Abdulhai et al. 2025; Wang et al. 2025; Zhang et al. 2026; Wang et al. 2026a]; they stop by imitation, LLM-judged rubrics or caps. Benchmarks score single-step prediction [Kruff et al. 2026a]; simulated users are over-cooperative [Zhou et al. 2026]. IR models stopping [Maxwell et al. 2015; Pirolli & Card 1999], but no simulator's reward targets real users' session-length distribution.

---

## 3. Methodology

> 註：本節只寫**已實作且在正式 run 路徑上**的設計（v16，`sep-sim/` 程式與 `ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`）；每個數值都對應一個程式常數或 SPEC 值（見本節末的註）。放進 2 頁模板時保留 3.2 前兩句、式 (1)(2)(3) 與 3.4 的選擇分數，其餘移到附錄或全文版。

### 3.1 Task and data

We follow the track's two tasks on the conversational data-search corpus. In **Task 2** the simulator receives a persona and a goal (topic, context, datasets the user already knows) and converses with a task agent until it ends the session or reaches the cap of T_max = 10 user messages. In **Task 1** it is conditioned on the real conversation up to message t−1 and produces message t; the same run also yields its decision to end at each real turn, which we score as a stop decision (our own mapping, not an official Task 1 measure). During training the task agent and the requirement ledger that scores coverage are a local gpt-oss-120b. We use the benchmark's goal- and persona-disjoint three-fold split: for each fold, training uses `train` (conversations with requirement annotations, used for Task 2 rollouts) and `train_all` (all training conversations, used for Task 1 stop groups, the few-shot pool and the human length distribution); `validation` selects checkpoints and, through one trigger, starts the annealing of the auxiliary stop loss (§3.3); `test` is read once, after training, for the final evaluation. No validation or test conversation is ever rolled out for a gradient, or enters the few-shot pool, the length distribution or the controller.

### 3.2 Simulator

**Planner.** At every turn a Planner (Qwen3-4B-Instruct-2507 with a LoRA adapter, the only trained component) reads the goal, the conversation so far and its own earlier notes, and writes a JSON plan: a one-sentence critique of its previous state, a verbalised distribution over dialogue moves with a target length for each, whether the goal has been met (`goal_met`: yes / partly / no), what the user still wants, and `end_session`, true if the message being planned is the user's last. The message's act is drawn from this distribution; a closing (*Complete*) act drawn while `end_session` is false is redrawn from the other entries, so the message and the decision agree. The first message cannot end the session. If `end_session` is true, the message's act is the Planner's own highest-probability *Complete* entry and the Speaker is told that this is the user's last message; the episode ends after it is emitted.

**Implicit profile.** From turn 2 the Planner also writes a `profile_note`: one or two sentences on how *this* person writes (length, tone, casing, phrasing) that its previous output got wrong. In Task 1 it compares its prediction of message t−1 with the real message t−1; in Task 2 it critiques its own previous message. Notes accumulate over the conversation and are passed to the Planner and the Speaker at every turn.

**Speaker and selector.** A frozen Speaker (Ditto-8B) writes four candidate messages per turn (one greedy and three sampled at temperature 0.7, top-p 0.9; at turn 1 all four are sampled at the checkpoint's own generation settings) from the Planner's plan, the notes and, for each candidate, its own k = 3 real messages of users with the same writing style, drawn from `train_all` conversations of other goals and personas. Candidates that break guards (e.g. copying eight or more consecutive words of an example) are rejected (duplicates are redrawn, and if every candidate fails, extra candidates are drawn without examples), and a Borda selector ranks the survivors, with equal weight, by closeness to the Planner's target length and by SimCSE similarity to reference texts (the user's own earlier real messages in Task 1 from turn 2; otherwise the few-shot examples). Ditto-8B has no end-of-conversation token, so a session ends through the Planner's decision or, by the benchmark convention, when the selected Speaker message is blank.

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

## References (as written in the sources; verify before use)

`TODO` = not in the sources.

| Key | Entry (as in source) | Source |
|---|---|---|
| Abdulhai et al. 2025 | Abdulhai, Cheng, Clay, Althoff, Levine, Jaques. *Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning*. NeurIPS 2025. arXiv:2511.00222 | RWS p.19; ConsistentPersona p.1 |
| Balog & Zhai 2024 | Balog, Zhai. *User Simulation for Evaluating Information Access Systems*. Foundations and Trends in IR, 2024 | RWS p.19 |
| Bernard & Balog 2024 | Bernard, Balog. *Towards a Formal Characterization of User Simulation Objectives in Conversational Information Access*. ICTIR 2024. arXiv:2406.19007 | bench metric_specs [BB24] |
| Bernard & Balog 2026 | Bernard, Balog. *UserSimCRS v2*. ECIR 2026 (full title TODO；目前未在正文引用) | RWS Appendix |
| Chalamalasetti et al. 2025 | Chalamalasetti, Hakimov, Schlangen. *clem:todd*. SIGDIAL 2025 (full title TODO) | RWS p.19 |
| Charnov 1976 | Charnov. Theoretical Population Biology 9(2):129–136, 1976 (title TODO) | Sep-1st README |
| Chopra 2026 | Chopra. *Beyond Cooperative Simulators*. arXiv 2026 (id/authors TODO；目前未在正文引用) | RWS Appendix |
| Cooper 1973 | Cooper. JASIS 24(6):413–424, 1973 (title TODO) | Sep-1st README |
| Deng et al. 2024 (PPDPP) | Deng, Zhang, Lam, Ng, Chua. ICLR 2024. arXiv:2311.00262 (title TODO) | Sep-1st README |
| EPO | ACL 2025. arXiv:2502.12486 (authors/title TODO) | Sep-1st README |
| El Asri et al. 2016 | El Asri, He, Suleman. *A Sequence-to-Sequence Model for User Simulation in Spoken Dialogue Systems*. Interspeech 2016 | RWS p.19 |
| Kraft & Lee 1979 | Kraft, Lee. IPM 15(1):47–58, 1979 (title TODO) | Sep-1st README |
| Kreutz et al. 2025 | Kreutz, Perry, Friedrich. *Data Discovery Using LLMs — A Study of Data User Behaviour*. TPDL 2025. arXiv:2507.04444 | bench metric_specs [K25] |
| Kreyssig et al. 2018 | Kreyssig et al. *Neural User Simulation for Corpus-based Policy Optimisation*. SIGDIAL 2018 | RWS p.19 |
| Kruff et al. 2026a (Sim4IA-Bench) | Kruff, Kreutz, Breuer, Schaer, Balog. *Sim4IA-Bench: A User Simulation Benchmark Suite for Next Query and Utterance Prediction*. ECIR 2026. arXiv:2511.09329 | RWS p.19; bench [S4IA26] |
| Kruff et al. 2026b | Kruff, Bernard, Schaer. *Validating Search Query Simulations: A Taxonomy of Measures*. 2026. arXiv:2601.11412 (venue TODO) | bench [KBS26] |
| Lai et al. 2025 | Lai, Wu, Wang, Zhou. *AdaRewriter: … Prompting-based Conversational Query Reformulation via Test-Time Adaptation*. EMNLP 2025. arXiv:2506.01381 (exact title TODO) | AdaRewriter p.1-3 |
| Li et al. 2024 (DAT) | Li, Wang, Viégas, Wattenberg. *Dialogue Action Tokens*. arXiv:2406.11978, 2024 (full title TODO) | RWS p.19 |
| Liu et al. 2026 (MUSE) | Liu et al. *MUSE*. arXiv:2604.13828, 2026 (full title TODO) | RWS p.19 |
| Maxwell et al. 2015 | Maxwell et al. CIKM 2015 (title TODO) | Sep-1st README |
| Mo et al. 2026 | Mo et al. *Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning*. ACL 2026. arXiv:2601.13115 | Agentic-Conv-Search slides p.1-3; RWS p.19 |
| Naous et al. 2026 (UserLM) | Naous, Laban, Xu, Neville. *Flipping the Dialogue: Training and Evaluating User Language Models*. ICLR 2026. arXiv:2510.06552 | RWS p.19 |
| Pirolli & Card 1999 | Pirolli, Card. Psychological Review 106:643–675, 1999 (title TODO) | Sep-1st README |
| Qian 2025 (UserRL) | Qian. *UserRL*. arXiv 2025 (id/title/authors TODO；目前未在正文引用) | RWS Appendix |
| Schatzmann et al. 2007 | Schatzmann et al. *Agenda-Based User Simulation*. NAACL-HLT 2007 Companion, pp. 149–152 | RWS p.19; Sep-1st README |
| Sun et al. 2026 (DITTO) | Sun, Zhou, Liu et al. *Reinforcing Human Behavior Simulation via Verbal Feedback*. arXiv:2605.20506, 2026 (venue TODO) | RWS p.19 |
| Wang et al. 2025 (USP) | Wang, Li, Yang, Zhou, Jiang, Li. *Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles*. ACL 2025 | RWS p.19 |
| Wang et al. 2026a (Turing-RL) | Wang, Zhang, Qiu, He, Li, Pentland, Levy, Kim. *Learning User Simulators with Turing Rewards*. arXiv:2606.19336, 2026 (venue TODO) | RWS; TuringRewardSim p.1 |
| Wang et al. 2026b (ProUtt) | Wang et al. *LLM-Driven Preference Data Synthesis for Proactive Prediction of the Next User Utterance …*. arXiv:2601.09713, 2026 (the two sources give different titles — TODO) | ProUtt p.1; DITTO deck |
| Wu et al. 2026 (HumanLM) | Wu, Choi, Khatua et al. *HumanLM*. arXiv:2603.03303, 2026 (full title TODO) | RWS p.19 |
| Zerhoudi 2026 (SimEval-IR) | Zerhoudi. *SimEval-IR*. SIGIR 2026 (full title TODO) | RWS p.19 |
| Zhang et al. 2026 (UserLM-R1) | Zhang, Li, Zhang et al. *Modeling Human Reasoning in User Language Models with Multi-Reward Reinforcement Learning*. arXiv:2601.09215, 2026 | RWS p.19 |
| Zhou et al. 2026 | Zhou, Sun, Ma et al. *Mind the Sim2Real Gap in User Simulation*. COLM 2026 | RWS p.19 |
| Dr. GRPO | TODO (not in the sources) | — |
| TREC 2026 UserSim | Guidelines, https://trec.usersim.ai/guidelines (2026-09-23 version) | bench metric_specs |

---

## Sources

- 自己的報告：Google Drive「TREC-UserSim」R1–R4（2026-07-13 / 07-27 / 08-10 / 08-28）。
- Internal Meeting 投影片：MingZhi（Related Work Summary、Planner_Speaker_Selector、v2fix_to_E1.6、MUSE、Agentic Conversational Search via RL …）；組員 HaoCheng / HungChun / KuanWei 的論文報告與重現。
- `/home/mzjiang/Sep-1st-Simulator/README.md`、`REPRO.md`。
- 實驗室內部 benchmark `/tmp2/hchsu/trec2026-usersim-benchmark`：`README.md`、`leaderboard/leaderboard.md`、`docs/metric_specs.md`、`protocols/*.md`、`instruments/termination_probe_v2/README.md`、`NOTICE.md`。
- 現行系統：`ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`（stop-sft-stageB 分支）。
