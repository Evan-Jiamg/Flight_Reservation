# Draft — Introduction & Related Work (rev. 3)

> 草稿 rev.3（2026-09-28，依 reviewer 第 1–3 輪意見修正）。英文正文供之後貼進 `scai_sigconf/body.tex`；`> 註` 是給作者的說明，定稿時刪除。
> **規則**：所有數字與文獻都來自文末「Sources」，文獻只照來源抄，缺的欄位標 `TODO`，定稿前請逐條對原文確認。
> 版面：模板是 2 頁 extended abstract（Intro 約 250 words、沒有 Related Work 節）。本稿 Intro 約 690 words（含小標與 TODO）、Related Work 約 560 words，
> 放進模板時：Intro 刪到 Context 2 句 / Gap 3 句 / Approach 2 句；Related Work 併成 Intro 裡約 80 words 的一段（見 §2 末的濃縮版）。

---

## 1. Introduction

**Context.** User simulators are increasingly used to evaluate and train conversational information-access systems [Balog & Zhai 2024]. The TREC 2026 User Simulation Track targets *conversational data search*, where a researcher looks for a relevant dataset through a search interface [Kreutz et al. 2025]. A simulator either predicts the next user utterance given a partial conversation history and the user's information need (Task 1), or generates a whole conversation and must itself decide when the goal is satisfied or when to give up (Task 2). The track compares the distributions of query length, turn count and clarification requests with real logs, alongside a human Turing test.

**The gap.** We call a simulator *session-level faithful* when its per-turn decisions to stop match those of real users — *when* to stop, not only *whether* — and, as a consequence, its distribution of session lengths matches theirs. Recent LLM simulators have become strong at turn-level fidelity (intent adherence, persona consistency, style) [Naous et al. 2026; Abdulhai et al. 2025; Wang et al. 2025], but on the track's data this does not carry over to the session level. In our lab's internal benchmark (not the official evaluation), a re-implementation of the Turing-RL recipe [Wang et al. 2026a] whose act transitions are within the human noise floor and whose act distribution is, with one other system, the closest to human in the benchmark (act TVD 0.185) never ends a replayed human session when offered one more turn (0/26 sessions) and runs 9.9 user turns against a human mean of 4.4; the public UserLM-8b [Naous et al. 2026], prompted with its native end token, instead emits END at 72% of the real mid-session turns where the human continued, and its sessions last 1.04 turns; zero-shot Ditto-8B comes close to the human mean (5.0 turns). Act-level fidelity thus does not predict session behaviour: one of the two simulators closest to human act distributions never stops, while Ditto-8B, further from them (act TVD 0.305), is close to the human session length. Stopping is also sensitive to the probe setup: a newer probe that adds one shared termination instruction (together with an updated system agent) moved Ditto-8B's K+1 end rate from 0.04 to 0.48. Existing RL rewards for simulators are per-utterance judgements or, in USP, one dialogue-level profile-similarity score; none compares a simulator's session lengths or stop positions with those of real users.

**Our approach.** We use a Planner–Speaker simulator in which ending the session is a separate, binding decision of the Planner (Qwen3-4B-Instruct-2507); a frozen Speaker (Ditto-8B, whose own end token is masked; a blank Speaker message also ends the episode) writes candidates that a length-and-style selector ranks. Two training-free versions of this design show why stopping must be learned. In the first, the Speaker ignored the Planner's decision to stop (asked to close, it closed 8% of the time) and 51 of 64 sessions hit the 10-turn cap. A second version, which among other changes makes the Planner's decision binding, does close (K+1 end rate 0.05 → 0.48 against the first version without annotations) but ends too early (premature end rate 0.03 → 0.14). We therefore train the Planner with a GRPO variant (group-mean baseline without standard-deviation normalisation) on session-level rewards: a log-ratio term that moves the simulated turn-count distribution towards real users' turn counts in the training fold, LLM-judged requirement coverage, and agreement with the real user's stop decision at real decision points of training conversations. The advantage of the length term is credited only to the Planner's end-session tokens, and an auxiliary supervised loss on the same stop decisions (weight floor 0.5) stabilises training.

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
