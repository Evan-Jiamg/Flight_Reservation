# Draft — Introduction, Related Work, Methodology & Future Work (rev. 5)

> 草稿 rev.5（2026-09-29）：依使用者決定重組——主方法是**免訓練**的 Planner–Speaker 模擬器；GRPO 移到 §4 Future Work，寫明遇到的難關（含樣本數）；Methodology 加上 PaperBanana 生成的架構圖（Figure 1）。英文正文供之後貼進 `scai_sigconf/body.tex`；`> 註` 是給作者的說明，定稿時刪除。
> **規則**：所有數字與文獻都來自文末「Sources」，文獻只照來源抄，缺的欄位標 `TODO`，定稿前請逐條對原文確認。
> 版面：模板是 2 頁 extended abstract（Intro 約 250 words、沒有 Related Work 節）。本稿（rev.5）Intro 約 950 words、Related Work 約 570 words、Methodology 約 850 words、Future Work 約 800 words（皆不含作者註），
> 放進模板時：Intro 刪到 Context 2 句 / Gap 3 句 / Approach 2 句；Related Work 併成 Intro 裡約 80 words 的一段（見 §2 末的濃縮版）。

---

## 1. Introduction

**Context.** User simulators are increasingly used to evaluate and train conversational information-access systems [Balog & Zhai 2024]. The TREC 2026 User Simulation Track targets *conversational data search*, where a researcher looks for a relevant dataset through a search interface [Kreutz et al. 2025]. A simulator either predicts the next user utterance given a partial conversation history and the user's information need (Task 1), or generates a whole conversation and must itself decide when the goal is satisfied or when to give up (Task 2). The track compares the distributions of query length, turn count and clarification requests with real logs, alongside a human Turing test.

**The gap.** We call a simulator *session-level faithful* when its per-turn decisions to stop match those of real users — *when* to stop, not only *whether* — and, as a consequence, its distribution of session lengths matches theirs. Recent LLM simulators have become strong at turn-level fidelity (intent adherence, persona consistency, style) [Naous et al. 2026; Abdulhai et al. 2025; Wang et al. 2025], but on the track's data this does not carry over to the session level. In our lab's internal benchmark (not the official evaluation), a re-implementation of the Turing-RL recipe [Wang et al. 2026a] whose act transitions are within the human noise floor and whose act distribution is, with one other system, the closest to human in the benchmark (act TVD 0.185) never ends a replayed human session when offered one more turn (0/26 sessions) and runs 9.9 user turns against a human mean of 4.4; the public UserLM-8b [Naous et al. 2026], prompted with its native end token, instead emits END at 72% of the real mid-session turns where the human continued, and its sessions last 1.04 turns; zero-shot Ditto-8B comes close to the human mean (5.0 turns). Act-level fidelity thus does not predict session behaviour: one of the two simulators closest to human act distributions never stops, while Ditto-8B, further from them (act TVD 0.305), is close to the human session length. Stopping is also sensitive to the probe setup: a newer probe that adds one shared termination instruction (together with an updated system agent) moved Ditto-8B's K+1 end rate from 0.04 to 0.48. Existing RL rewards for simulators are per-utterance judgements or, in USP, one dialogue-level profile-similarity score; none compares a simulator's session lengths or stop positions with those of real users — a gap we first address by design (§3) and then try to close by training (§4).

**Our approach.** We build a Planner–Speaker simulator in which ending the session is a separate, binding decision of a Planner (Qwen3-4B-Instruct-2507) that judges for itself whether the user's goal has been met; a frozen Speaker (Ditto-8B, which has no end-of-conversation token; a blank Speaker message also ends the episode, by the benchmark convention) writes candidate messages from the plan, from an implicit profile of how this user writes that the Planner accumulates over the conversation, and from real messages of users with the same writing style, and a length-and-style selector picks one (Figure 1). Two earlier training-free versions of this design showed why stopping needs an explicit, binding decision. In the first, the Speaker ignored the Planner's decision to stop (asked to close, it closed 8% of the time) and 51 of 64 sessions hit the 10-turn cap. A second version, which among other changes makes the Planner's decision binding, does close (K+1 end rate 0.05 → 0.48 against the first version without annotations) but ends too early (premature end rate 0.03 → 0.14). The current design removes the rule-based stop ledger and lets the Planner's own `goal_met` / `still_wanted` judgement decide, and adds the implicit profile, the style-matched examples and the Borda selector; whether this improves on the second version has not yet been measured under the same protocol. We also tried to learn stop timing with reinforcement learning (GRPO with session-level rewards); on held-out conversations it did not improve on the untrained Planner, and we report what made it hard (§4).

> 註：Findings 目前只有 fold 2 的 test（Task 2：5 段對話 × 8 seeds；Task 1：9 段對話），fold 0/1 還沒跑；與前一版（E1.6）在同一協定下的比較也還沒做（AUDIT_SPEC 的比較協定）。若要與 benchmark 其他方法同表，須先證明我們 Task 2 環境（自架 R0／ledger judge）與 benchmark 的 system agent 設定（prompt v4、length_retry_v1）一致。數字來源：`runs/pend_f2_v16/test_boot.txt`（u0 列；TEST CHECK PASSED、verify passed，2026-09-29 00:33）。

**Findings.** On the held-out test conversations of one fold (fold 2: five conversations for Task 2, each run with eight seeds, and nine for Task 1), the untrained simulator produces sessions of 5.65 user turns on average against 5.60 for the real users (turn-count W1 0.80), covers 86% of the users' requirements, and in Task 1 never ends before the real user's last message (0 of 9 conversations) but ends at it in only 3 of 9 (stop F1 0.50 under our mapping; end-probability AUC 0.83). The GRPO-trained Planner did not improve on these numbers (§4). `TODO`: the other two folds, and the comparison with the earlier version under the same protocol.

**Contributions.**
- We show, on the track's data, that act-level fidelity does not predict session-level fidelity: session behaviour ranges from never ending to ending almost immediately, independently of act fidelity, and is sensitive to the prompt and probe setup.
- We design a training-free Planner–Speaker simulator in which ending the session is an explicit, binding decision grounded in the Planner's own judgement of goal completion, with a Speaker conditioned on an implicit profile and on style-matched real messages; on the held-out conversations of one fold its mean session length matches the real users' (5.65 vs 5.60 user messages; mean absolute difference 1.8 per episode) (`TODO`: all folds).
- We report a negative result for session-level reinforcement learning: GRPO with rewards on real users' session-length distribution and stop positions did not beat the untrained Planner on held-out conversations, and we identify the obstacles — validation sets of one to four conversations that make checkpoint selection noisy, a sparse stop signal, and an auxiliary supervision whose gradient, in a smoke test, dwarfed the RL gradient.

> 註：
> 1. benchmark 是實驗室內部量測工具，NOTICE 說它「不是可引用的出版物；請引用 Track 與原始論文」。致謝 Lucas H.-C. Hsu（Sep-1st README §7），**不要引用 repo**。
> 2. Turing-RL 的數字：probe v2、三個 fold test side 共 26 個 session（K+1 0/26、rollout 自己結束 1/26、平均 9.885 輪），`instruments/termination_probe_v2/README.md` 與 `leaderboard.md`。真人雜訊地板：act TVD（F1）0.155、transition JSD（F2）0.19——Turing-RL 的 transition JSD 0.121 在地板內，act TVD 0.185 **在地板之上**（但與 A1-s1 的 0.182 並列最接近真人）。Ditto 的提示敏感度：加一條共用 TERMINATION_INSTRUCTION 讓全語料 K+1 從 0.0357 變 0.4821（probe README 49-52；v1 的 system agent 也是舊設定，所以不能全歸功於那條指示）。它的「結束」在自己的格式裡是空訊息，所以「不結束」可能部分來自重現方式——正文已寫 re-implementation，必要時再加一句 hedge。
> 3. UserLM-8b：**未入榜**，用它原生的 end token、沒有共用的 TERMINATION_INSTRUCTION（probe README 74-76）；72% = `teacher_forced_turn_with_end_decision_rate`（分母是所有真人繼續的中間輪）。Ditto-8B：K+1 0.577、4.962 輪，是 F10 family 最佳，而且就是我們的凍結 Speaker——reviewer 會問「為何不直接用 Ditto」，答案要在 Results 用「停在哪一輪」（premature、stop AUC）而不只是平均輪數來回答。
> 4. 兩個免訓練版本的數字是**我們自己在 benchmark 較早協定下**的量測（v2fix：Qwen2.5-32B planner + UserLM-8b speaker，64 episodes；E1.6：K+1 0.0536→0.4821、premature 0.0321→0.1446，出自 v2fix_to_E1.6 投影片 p.4-5），**不可和上面 probe v2 的數字並列成同一張表**。8% 與 51/64 屬於 v2fix；0.0536 屬於 v2fix 去掉 annotations 的版本（原 v2fix 是 0.0179）；v2fix → E1.6 改了五件事（annotations、stop 權限、speaker 輸入、role header、開場取樣），所以正文寫「among other changes」。兩版用的模型也不同（v2fix：Qwen2.5-32B planner＋UserLM-8b speaker；E1.6 的 speaker 來源未寫明，請確認）。
> 5. 「免訓練」＝test 中的 u0：Qwen3-4B-Instruct-2507 加上一個**初始化為零效果**的 LoRA（等於原模型）。§4 的 RL 描述對應 AUDIT_SPEC / SPEC_v16；Dr. GRPO 需要引文（`TODO cite`）。
> 6. 實驗室組員的 Unified Framework 投影片提出過 user 端的 termination reward（λ4·r_term），A1-s1 是他在同一 benchmark 的 RL 模擬器。**我們不宣稱「第一個獎勵 user 端結束」**，新穎性放在「對真人長度分佈與真人結束位置做最佳化、並把 credit 給結束決策」。若要正面比較，請確認 A1-s1 是否用了 r_term。
> 7. 官方 Task 1 是 next-utterance prediction；我們自己的 Task 1 結束決策指標（M2 mapping, decision D7）不是官方指標；benchmark 的 `termination_f1` 已於 9/26 退役。

---

## 2. Related Work

**LLM user simulators.** User simulation evolved from agenda-based [Schatzmann et al. 2007] and neural sequence models [El Asri et al. 2016; Kreyssig et al. 2018] to LLMs trained to play the user [Balog & Zhai 2024]. USP [Wang et al. 2025] conditions generation on an implicit profile extracted from each dialogue (our Planner keeps a similar implicit profile); UserLM-8b [Naous et al. 2026] flips assistant dialogues to train a user model with an end-of-conversation token, and documents prompted assistants' reluctance to end a session; HumanLM [Wu et al. 2026] aligns latent user states; MUSE [Liu et al. 2026] optimises a profile by iterative self-critique against real dialogues; ProUtt [Wang et al. 2026b] predicts the user's next intent path. Most are evaluated at the turn level; UserLM also scores termination, learning to stop by imitating real ends and needing a guardrail against ending too early.

**RL for user simulators.** ConsistentPersona [Abdulhai et al. 2025] applies multi-turn PPO with a judge-scored persona-consistency reward; USP's RLCC stage [Wang et al. 2025] rewards one dialogue-level profile-similarity (cycle-consistency) score, repeated over the user turns, plus human-likeness; UserLM-R1 [Zhang et al. 2026] combines rule and rubric rewards with GRPO and judges "timing of hanging up" with an LLM; Turing-RL [Wang et al. 2026a] uses a pairwise Turing-style judge as the GRPO reward; MUSE averages turn-level rubric rewards over the session under GRPO; DITTO [Sun et al. 2026] adds verbal feedback to GRPO. Session length is either fixed (ConsistentPersona's 10, 20, 40 or 60 turns), capped (USP, up to 10 turns) or judged by an LLM; none of these rewards compares session lengths or stop positions with real users. Steering a frozen generator with a small trained planner has precedents — Dialogue Action Tokens [Li et al. 2024], PPDPP [Deng et al. 2024] and EPO (`TODO` authors, ACL 2025) — but none of them trains a *user's* decision to stop: PPDPP plans the system's moves, drops CraigslistBargain's terminal acts and caps dialogues at 8 turns, and DAT names leaving the chat as an open direction. We therefore do not claim the planner–generator split as new, only its use for a user whose stop decision is an explicit Planner output (and, in §4, the attempt to train it).

**Evaluation and stopping.** Sim4IA-Bench [Kruff et al. 2026a] scores next-query and next-utterance prediction on real search sessions, but every task is single-step; Bernard & Balog [2024] formalise simulation objectives in conversational information access; Kruff et al. [2026b] propose a taxonomy of measures for validating query simulations; clem:todd [Chalamalasetti et al. 2025] benchmarks simulator × dialogue-system combinations, and SimEval-IR [Zerhoudi 2026] separates behavioural realism from tester reliability. Zhou et al. [2026] find that simulated users are more cooperative and disclose task information earlier than real ones, and recommend reporting behaviour, task outcome and subjective ratings separately — a session-level realism gap. In interactive IR, when users stop has long been modelled with stopping rules [Cooper 1973; Kraft & Lee 1979; Maxwell et al. 2015] and foraging theory [Charnov 1976; Pirolli & Card 1999]; LLM simulators learn to stop only by imitation (UserLM's end token) or through LLM-judged rubrics, and none is optimised against real users' session-length distribution. Finally, conversational search systems are increasingly optimised against learned or interactive feedback — reward-model reranking for query reformulation [Lai et al. 2025] and GRPO-trained agentic search [Mo et al. 2026] — so realistic session endings matter for training and evaluating them.

> 註：
> - 作者先前在 Sep-1st 試過具名 stopping rules，**沒贏過單純的輪數計數器**（F1 0.442 vs 0.524/0.559，Sep-1st README）；benchmark 的 stop judgement 也顯示 `rule_turn_count` AUC 0.809。Results 要把「停在哪一輪」和這個 turn-count / hazard 基線比（不只和 E1.6 比），reviewer 會要求。
> - UserLM 的 termination F1（論文 63.54）只出自組員轉述，要引請回原論文確認。
> - UserRL [Qian 2025]、UserSimCRS v2 [Bernard & Balog 2026]、Chopra [2026]《Beyond Cooperative Simulators》在來源裡**只有標題與 venue**，沒有內容描述，所以**已從正文移除**；讀過原文、能寫出一句正確描述後再放回（Chopra 可能與 Zhou et al. 並列，UserRL 可能放 RL 段）。
> - UnifiedFramework 說「termination reward 在 task-oriented dialogue RL 是標準做法（訓練 system 端）」但沒給出處——**不寫進正文**。

**Condensed version for the 2-page template (~80 words).** User simulators learn turn-level fidelity via SFT [Naous et al. 2026] or RL with per-utterance, rubric or dialogue-level similarity rewards [Abdulhai et al. 2025; Wang et al. 2025; Zhang et al. 2026; Wang et al. 2026a]; they stop by imitation, LLM-judged rubrics or caps. Benchmarks score single-step prediction [Kruff et al. 2026a]; simulated users are over-cooperative [Zhou et al. 2026]. IR models stopping [Maxwell et al. 2015; Pirolli & Card 1999], but no simulator's reward targets real users' session-length distribution.

---

## 3. Methodology

> 註：本節只寫**已實作且在評估路徑上**的設計（`sep-sim/` 程式與 `ops/AUDIT_SPEC_pend_grpo.md`）。Figure 1 由 PaperBanana 依本節內容生成（見圖下註）。放進 2 頁模板時保留 Figure 1、3.2 前兩句與 3.3 的指標定義。

### 3.1 Task and data

We follow the track's two tasks on the conversational data-search corpus. In **Task 2** the simulator receives a persona and a goal (topic, context, datasets the user already knows) and converses with a task agent until it ends the session or reaches the cap of T_max = 10 user messages. In **Task 1** it is conditioned on the real conversation up to message t−1 and produces message t; the same run also yields its decision to end at each real turn, which we score as a stop decision (our own mapping, not an official Task 1 measure). The task agent and the requirement ledger that scores coverage are a local gpt-oss-120b. We use the benchmark's goal- and persona-disjoint three-fold split: `train_all` (all training conversations of a fold) supplies the Speaker's few-shot examples, and `test` is read once, for the evaluation; `train` and `validation` are used only by the reinforcement-learning experiments of §4. No test conversation enters the few-shot pool.

![Figure 1](fig/method_final.png)

*Figure 1: The Planner–Speaker user simulator. At every user turn the Planner writes a structured plan, including the decision to end the session; a frozen Speaker writes candidate messages from the plan, the accumulated profile notes and same-style examples; a selector picks one message, which is sent to the task agent (a closing message receives no reply; the Speaker also sees the conversation history). Illustration generated with PaperBanana (Gemini 3.1 Pro Preview for planning and critique, Gemini 3.1 Flash Image Preview for rendering, via OpenRouter) from the authors' method description and verified by the authors.*

> 註：Figure 1 = PaperBanana 候選 `fig/out/method_0.png`（2026-09-29 生成 4 張；0 號 12 條連線與全部文字內容都符合規格（但小字如 "same style, other users"、"4 candidates"、Planner 欄位約只有規格要求字高 1/22 的一半，縮成單欄寬時可能太小，定稿前可考慮重生或放大）；1 號多出代號字母且 notes↔Speaker 雙向、2 號缺 notes→Planner 並多出 END→Planner、3 號把 notes→Planner 誤標為 append note）。使用者可從 4 張中改選；圖中每條連線與文字需逐條核對（核對清單見 `fig/fig_method_spec.txt` 的 CONNECTIONS / FORBIDDEN）。若投稿場地有 AI 生圖政策，caption 的揭露句與 AI Declaration 都要保留。

### 3.2 Simulator

**Planner.** At every turn a Planner (Qwen3-4B-Instruct-2507, used as is) reads the goal, the conversation so far and its own earlier notes, and writes a JSON plan: a one-sentence critique of its previous state, a verbalised distribution over dialogue moves with a target length for each, whether the goal has been met (`goal_met`: yes / partly / no), what the user still wants, and `end_session`, true if the message being planned is the user's last. The message's act is drawn from this distribution; a closing (*Complete*) act drawn while `end_session` is false is redrawn from the other entries, so the message and the decision agree. The first message cannot end the session. If `end_session` is true, the message's act is the Planner's own highest-probability *Complete* entry and the Speaker is told that this is the user's last message; the episode ends after it is emitted.

**Implicit profile.** From turn 2 the Planner also writes a `profile_note`: one or two sentences on how *this* person writes (length, tone, casing, phrasing) that its previous output got wrong. In Task 1 it compares its prediction of message t−1 with the real message t−1; in Task 2 it critiques its own previous message. Notes accumulate over the conversation and are passed to the Planner and the Speaker at every turn.

**Speaker and selector.** A frozen Speaker (Ditto-8B) writes four candidate messages per turn (one greedy and three sampled at temperature 0.7, top-p 0.9; at turn 1 all four are sampled at the checkpoint's own generation settings) from the Planner's plan, the notes and, for each candidate, its own k = 3 real messages of users with the same writing style, drawn from `train_all` conversations of other goals and personas. Candidates that break guards (e.g. copying eight or more consecutive words of an example) are rejected (duplicates are redrawn, and if every candidate fails, extra candidates are drawn without examples), and a Borda selector ranks the survivors, with equal weight, by closeness to the Planner's target length and by SimCSE similarity to reference texts (the user's own earlier real messages in Task 1 from turn 2; otherwise the few-shot examples). Ditto-8B has no end-of-conversation token, so a session ends through the Planner's decision or, by the benchmark convention, when the selected Speaker message is blank.

### 3.3 Evaluation

We evaluate on the test split of each fold (so far fold 2). **Task 2**: each test scenario with requirement annotations (five in fold 2) is run with eight seeds (Planner temperature 0.7); we report the mean number of user messages against the real users', the Wasserstein-1 distance between the simulated and real turn counts (real counts capped at T_max), the mean absolute difference per episode, and the LLM-judged requirement coverage. **Task 1**: on every test conversation (nine in fold 2), the greedy run gives the stop decisions, scored as a stop F1 (our mapping) and as the fraction of conversations with a premature end; in addition, at every real decision point t we compute the teacher-forced probability that the Planner ends the session, P_end = p(true) / (p(true) + p(false)) for the `end_session` value given the greedy plan's prefix, and summarise it with the balanced score

  bal_p = ½ · mean_{t = n} P_end + ½ · mean_{2 ≤ t < n} (1 − P_end)  (1)

(invalid points count as P_end = 0) and with the AUC of P_end between the real last message and the earlier ones. Two systems are compared with a paired bootstrap over conversations (10,000 resamples).

> 註：評估程式 `sep-sim/eval_test_rl.py`（與訓練時 validate() 同一程序，只換成 test 的 id）與 `eval_test_boot.py`（重算每個數字＋paired bootstrap）；fold 2 結果在 `runs/pend_f2_v16/test_boot.txt`。Task 2 的 test 只取有需求標註的對話（coverage 需要），Task 1 用 test_all。coverage 對免訓練系統不是 reward 的一部分，所以可以當評估指標；但對 §4 的 RL 版它是 reward 的一項（D3）。

---

## 4. Future Work: reinforcement learning for stop timing

> 註：本節內容取自原 §3.3–3.4（已經 reviewer 對照程式碼 ACCEPT），壓縮後加上結果與難關。數字來源：fold 2 v16 run（`runs/pend_f2_v16`：updates、validation、reselect、test_boot）、`ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`、`sep-sim/rl_controllers.py` 註解。

**What we tried.** Recent simulators are trained with PPO or GRPO on per-utterance rewards [Abdulhai et al. 2025; Zhang et al. 2026; Wang et al. 2026a]. We instead trained the Planner (LoRA rank 16) with a GRPO variant on *session-level* rewards. For free-running episodes (four training scenarios × G = 4 episodes per update) the reward is

  R = w_cov · coverage + w_dist · [log p_h(T) − log q(T)] − Σ_k λ_k · rate_k ,  (2)

where T is the number of user messages, p_h the smoothed distribution of real users' message counts in the training fold, q the same distribution over the update's rollouts (so E_q[log p_h − log q] = −KL(q ‖ p_h) is maximised when the *distribution* of lengths matches the human one), and rate_k the rates of unparsable or truncated plans. In teacher-forced stop groups (eight training conversations × G₁ = 8 samples at the real last message and one earlier message) a plan gets reward 1 if its `end_session` agrees with the real user. Advantages are group-mean differences without standard-deviation normalisation (Dr. GRPO, `TODO` cite), A_i = R_i − mean_j R_j; the part caused by the length term is credited only to the `end_session` value tokens. The loss is the clipped surrogate with a truncated importance weight for vLLM sampling and a k3 KL penalty (β = 0.04) to the starting policy, plus an auxiliary supervised loss on the real users' stop decisions whose weight never falls below 0.5; an LLM controller retunes the reward weights every five updates from training statistics. Checkpoints are validated every five updates and scored by coverage − W1(turn counts) + bal_p on the validation conversations.

**What happened.** On fold 2, training stopped after ten updates (two validations without improvement). Re-validating update 0, 5 and 10 with eight seeds selected update 5 (score 0.213 against 0.098 and 0.095), but a paired bootstrap over the four validation conversations separated none of them. On the test conversations, update 5 matched the untrained Planner on session length (turn-count W1 0.83 vs 0.80; 95% CI of the difference [−0.38, 0.65]), covered slightly fewer requirements (0.82 vs 0.86; [−0.08, +0.004]), and in Task 1 had a lower stop F1 (0.18 vs 0.50; [−0.62, 0.00]; one of nine conversations ended prematurely vs none) and lower but not separable bal_p and AUC; on seeds 0–1 alone, the seeds used for validation during training, the W1 difference points the other way (0.60 vs 1.00).

**Obstacles.**
- *Tiny validation sets.* The validation split has four conversations in fold 2 and one in fold 0, so checkpoint selection is dominated by noise: with two seeds the three candidates scored 0.651, 0.418 and 0.649, with eight seeds 0.098, 0.213 and 0.095 — the worst candidate under two seeds became the best under eight.
- *Small test sets.* Five Task 2 and nine Task 1 conversations per fold give confidence intervals wider than any effect we could expect; the three folds must be pooled before any conclusion.
- *Sparse stop signal.* A conversation has one real end; in the teacher-forced stop groups five to eight of every eight groups had identical rewards (no gradient) before we added dynamic re-sampling of groups.
- *Auxiliary supervision dominates.* In a smoke update the auxiliary stop loss produced a gradient about 130 times the RL gradient (norm 0.040 vs 0.0003), which suggests that the supervision, not the session-level reward, drives the stop tokens.
- *Length drift.* Other pressures shorten sessions faster than the length term restores them: in an earlier run, lowering the length weight collapsed the conversation length, and in the fold-2 run, although the weight could only rise, the mean length of training episodes fell from 6.4 to 3.5 user messages within ten updates, below the real mean (4.4 in the whole corpus, 5.6 in the fold-2 test conversations); we have not isolated the cause.
- *An unpredictable per-user target.* A per-user length reward is not viable: the visible number of requirements is only weakly (and negatively) related to a user's real turn count (r = −0.36; almost every user has 7 or 8 requirements), and per-user counts vary little (s.d. 1.1).
- *Cost.* Each update needs full multi-turn episodes with an LLM task agent and an LLM judge; ten updates with three validations took about eight hours on two GPUs, which limits the number of updates and seeds.

**Directions.** Pool validation and test conversations across folds (or use repeated cross-validation) so that checkpoint selection and evaluation have enough conversations; give the stop decision a denser signal (more decision points per conversation, and rewards that compare the whole predicted stop distribution with the real one); balance the auxiliary loss against the RL gradient instead of fixing its floor; and compare GRPO with a value-based baseline (PPO; our code has an untested PPO path that does not yet support the stop-token credit assignment).

> 註：
> - 「130 倍」＝0.040 / 0.0003，出自 v16 smoke test（Task-1-only update）。「6.4 → 3.5」出自 fold 2 v16 的 updates.jsonl（訓練 rollout 平均輪數）；真人 4.4（benchmark 錨點，全語料）、5.6（fold 2 test 的 5 段）。「r = −0.36、s.d. 1.1」出自 SPEC_v16 第 5 點。「5–8/8 組無梯度」出自 SPEC_v16 第 1 點（u1–u5）。「約 8 小時」：run_meta start 05:32，u10 的 validation summary 13:24（`evidence_fw3.txt`）。smoke 的梯度比只來自一次 Task-1-only smoke update；正式 run 每次 update 的 rl/aux 梯度範數有記錄但未納入證據。
> - PPO 在 `rl_algos.py` 有實作（`--algo ppo`），但從未正式跑過；它需要 `--stop-credit 0` 與 `--ablation`，而且 `prepare()` 會用 value head 的 advantage 取代所有樣本（含 Task 1 結束樣本）的 advantage。
> - 原本完整的 GRPO 方法描述（含式、所有超參數與程式對應）保留在 git 歷史 rev.4（commit f14d4ef）與 `method_en.md`，需要時可放進附錄。

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
