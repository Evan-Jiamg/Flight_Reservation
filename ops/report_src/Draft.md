# Draft — Introduction & Related Work

> 草稿（2026-09-28）。英文正文供之後貼進 `scai_sigconf/body.tex`；`> 註` 是給作者的說明，定稿時刪除。
> **規則**：所有數字與文獻都來自下列來源（見文末「Sources」），文獻資料只照投影片 / repo 上寫的抄，
> 缺的欄位標 `TODO`，**請在定稿前逐條對原文確認**。沒有任何文獻是憑記憶補的。
> 版面：模板是 2 頁 extended abstract（Intro 約 250 words、無 Related Work 節）。本稿刻意寫完整，
> 之後再依版面刪減；Related Work 可壓成 Intro 的一段，或放進投稿版全文。

---

## 1. Introduction

**Context.** User simulators are increasingly used to evaluate and train conversational information-access systems, because recruiting real users for every system variant is slow and expensive [Balog & Zhai 2024]. The TREC 2026 User Simulation Track makes this concrete for *conversational data search*: a researcher looks for a relevant dataset through a search interface, and a simulator must either predict the next user utterance given the real history (Task 1) or generate a whole session from a persona and a goal — *including when to stop* (Task 2). Submissions are judged on whether their behaviour matches real logs and whether human raters can tell them apart from real users.

**The gap.** Recent LLM-based simulators have made rapid progress on *turn-level* fidelity: they reproduce the distribution of user intents and dialogue acts, stay consistent with a persona, and write utterances that human-likeness judges accept [USP; UserLM; ConsistentPersona; Turing-RL]. Session-level behaviour — how long a user keeps going and when they are satisfied enough to leave — has received far less attention. Reinforcement-learning simulators score each *utterance* (persona consistency, profile recovery, a Turing-style judge), so no reward term measures a property that exists only at session scale; other work fixes the dialogue length in advance or suppresses the end token [ConsistentPersona; USP; UserLM]. Evidence from a shared internal benchmark for the track shows why this matters. The simulator with the best act-transition fidelity in the benchmark (Turing-RL; act-transition JSD 0.121, act-distribution TVD 0.185, length KS D 0.089) never ends a replayed human session when given one more turn (K+1 termination rate 0.000), ends only 3.8% of its free-running sessions, and runs 9.9 turns against a human mean of 4.4 — essentially always hitting the 10-turn cap. The opposite failure also occurs: UserLM-8b ends after 1.04 turns on average, and 72% of its stop decisions are premature. **Matching users turn by turn does not make a simulator stop like a user.**

**Our approach.** We build on a Planner–Speaker simulator in which stopping is an explicit decision rather than an absent or suppressed token. A small Planner (Qwen3-4B) reads the conversation, keeps an implicit profile of the user, judges whether the goal has been met, and decides whether to end the session; a frozen Speaker (Ditto-8B) writes candidate utterances in the user's style, and a selector picks one. Our earlier, training-free version of this design was competitive on turn-level metrics but inherited the field's session-level weakness: it ran 8.4 turns on average, 51 of 64 sessions hit the turn cap, and when the Planner decided to stop, the Speaker actually closed the conversation only 8% of the time. We therefore train the Planner with group-relative policy optimisation (GRPO) [DeepSeekMath], the RL method adopted by many recent simulator papers [UserLM-R1; Turing-RL; DITTO], but with *session-level* rewards: matching the real distribution of session lengths, covering the user's requirements, and agreeing with real users' stop decisions on held-out conversations, with the credit for the length reward routed to the Planner's end-session decision.

> 註：下一段等 fold 2 test（u5 vs u0）與其他 fold 跑完再填；目前沒有任何可引用的 v16 結果。

**Findings.** `TODO` — e.g. "On held-out conversations of the track's dataset, the trained Planner … turn-count W1 … while keeping act-level fidelity …, compared with <baselines> in the benchmark." (fill from the fold-2 test and the 3-fold pooled evaluation; report paired bootstrap intervals, not single numbers.)

**Contributions.**
- We document a gap between turn-level and session-level fidelity in current LLM user simulators for multi-turn conversational search: methods that match human intents and dialogue acts still fail to stop like humans, in both directions (never stopping and stopping immediately).
- We design a Planner–Speaker simulator in which ending the session is an explicit, trainable Planner decision, and train it with GRPO using session-level rewards (session-length distribution matching, requirement coverage, and stop agreement with real users) instead of per-utterance rewards.
- `TODO` We show that … (result), evaluated with goal- and persona-disjoint cross-validation against re-implemented and public simulators on the same benchmark.

> 註：
> 1. 「shared internal benchmark」是學長的 repo（`/tmp2/hchsu/trec2026-usersim-benchmark`），README 自己寫「our internal instrument, not the official track evaluation」。定稿時要寫清楚它是誰維護的（README §7 記載 Lucas H.-C. Hsu；請確認要怎麼致謝／引用）。
> 2. 上面 Turing-RL / UserLM-8b 的數字出自 `leaderboard/leaderboard.md`（main domain，system agent = 本地 gpt-oss-120b，probe v2）。Turing-RL 與 UserLM-R1 是**團隊重現版**（NOTICE §5：代表該配方在本域的表現，不是官方系統）——正文要寫 "re-implementation"。
> 3. Leaderboard 所有 Friedman 檢定都不顯著，所以只能逐指標描述，**不要寫總排名**。
> 4. v2fix 的數字（8.4375 turns、51/64 截斷、8.0% 關閉率）出自 Sep-1st-Simulator README §2.2/§4.1 與 PSS:39；K+1 0.0179 是舊 probe v1，v2 下每 fold 是 0.0。
> 5. 官方 Task 1 是 next-utterance prediction；我們的「Task 1 stop decision／term_f1」是自己的 M2 對應（decision D7），benchmark 的 `termination_f1` 已在 9/26 退役。正文不要把它寫成官方指標。

---

## 2. Related Work

### 2.1 LLM-based user simulation

User simulation has a long history in dialogue systems, from agenda-based simulators [Schatzmann et al. 2007] to neural sequence models [El Asri et al. 2016; Kreyssig et al. 2018]; Balog & Zhai [2024] survey its use for evaluating information-access systems. Recent work trains LLMs to play the user. USP [Wang et al. 2025] extracts an implicit profile from each dialogue and conditions generation on it; UserLM-8b [Naous et al. 2026] "flips" assistant dialogues to train a dedicated user model with an explicit end-of-conversation token; HumanLM [Wu et al. 2026] aligns latent user states; MUSE [Liu et al. 2026] evolves user profiles during the session; ProUtt [Wang et al. 2026] predicts the next user intent path. These models are evaluated mostly at the turn level (semantic and style similarity to the real next utterance, persona consistency, detectability). UserLM is the notable exception in scoring dialogue termination, yet its authors still add guardrails that stop the model from ending too early, and it is trained by supervised learning rather than RL.

> 註：UserLM 的 termination F1（論文 63.54 vs USP-8B 21.31）出自組員 UnifiedFramework:12 的轉述；Hao-Cheng 重現 60.24（UserLM8b_reproduction:5-6）。要引數字請回原論文確認。

### 2.2 Reinforcement learning for user simulators

Several recent simulators are fine-tuned with RL, using PPO [Schulman et al. 2017] or GRPO [Shao et al. 2024]. ConsistentPersona [Abdulhai et al. 2025] applies multi-turn PPO with a judge-scored persona-consistency reward; USP's RLCC stage [Wang et al. 2025] rewards profile recovery (cycle consistency) and human-likeness; UserLM-R1 [Zhang et al. 2026] combines rule-based and rubric rewards with GRPO over a dynamic profile; Turing-RL [Wang et al. 2026] uses a pairwise Turing-style judge as the GRPO reward; DITTO [Sun et al. 2026] adds verbal feedback to GRPO. In all of them the reward is computed per utterance (or, for USP, a session score copied back to each turn), and session length is either fixed in advance (ConsistentPersona's 10–60-turn dialogues, USP's 10-turn cap) or judged by an LLM rather than compared with real sessions (UserLM-R1's "timing of hanging up"). Our reward instead targets the distribution of real session lengths and real stop decisions.

Architecturally, steering a frozen generator with a small trained planner has precedents: Dialogue Action Tokens [Li et al. 2024] train a planner on a per-utterance signal, and PPDPP (ICLR 2024) and EPO (ACL 2025) plug a trained or verbalised planner into a frozen dialogue agent. These works plan the *system's* moves and do not represent ending the session (PPDPP drops terminal acts and caps dialogues at 8 turns); we therefore do not claim the planner–generator split itself as new, but its use for a *user* whose stop decision is the object of training.

> 註：
> - 「PPDPP、EPO 不可宣稱新穎」是 README FW 裡自己寫的警語，照實保留。PPDPP / EPO 的完整書目投影片沒有（只有 arXiv 2311.00262 / 2502.12486），標 TODO。
> - 組員 UnifiedFramework:9 說「termination reward 在 task-oriented dialogue RL 是標準做法（訓練 system 端）」但沒給出處——**沒有出處前不要寫進正文**。

### 2.3 Evaluating user simulators and session behaviour

Evaluation of simulators has moved from single metrics to suites. Sim4IA-Bench [Kruff et al. 2026] scores next-query and next-utterance prediction on real search sessions, but every task is a single-step prediction; Bernard & Balog [2024] formalise simulation objectives in conversational information access; Kruff et al. [2026] propose a taxonomy of measures for validating query simulations; SimEval-IR [Zerhoudi 2026] and clem:todd [Chalamalasetti et al. 2025] compare simulators through the systems they evaluate. Closest to our concern, Zhou et al. [2026] ("Mind the Sim2Real Gap") find that simulated users are more cooperative and disclose task information earlier than real users — a session-level mismatch — but do not propose a training remedy. In interactive IR, when and why users stop has long been modelled with stopping rules and foraging theory [Maxwell et al. 2015; Pirolli & Card 1999]; LLM simulators have not been trained against such behaviour. Finally, our setting is conversational *search*, where recent systems are themselves trained with RL — e.g. conversational query rewriting [AdaRewriter; Lai et al. 2025] and agentic conversational search with GRPO [Mo et al. 2026] — which makes a simulator that ends sessions realistically a prerequisite for training and evaluating them.

> 註：最後一句是論點延伸，不是任何來源的結論；若版面不夠可刪。Maxwell 2015 / Pirolli & Card 1999 只在 PSS / README 以「作者＋年份＋venue」出現，書名缺，標 TODO。

---

## References (as written in the sources; verify before use)

Bibliographic fields are copied from the Related Work Summary slide (RWS:19, "verified entry by entry"), the benchmark's `docs/metric_specs.md`, or the labmates' slides. `TODO` = not in the sources.

| Key | Entry (as in source) | Source |
|---|---|---|
| Abdulhai et al. 2025 | Abdulhai, Cheng, Clay, Althoff, Levine, Jaques. *Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning*. NeurIPS 2025. arXiv:2511.00222 | RWS:19; ConsistentPersona:1 |
| Balog & Zhai 2024 | Balog, Zhai. *User Simulation for Evaluating Information Access Systems*. Foundations and Trends in IR, 2024 | RWS:19 |
| Bernard & Balog 2024 | Bernard, Balog. *Towards a Formal Characterization of User Simulation Objectives in Conversational Information Access*. ICTIR 2024. arXiv:2406.19007 | bench metric_specs [BB24] |
| Chalamalasetti et al. 2025 | Chalamalasetti, Hakimov, Schlangen. *clem:todd*. SIGDIAL 2025 (full title TODO) | RWS:19 |
| El Asri et al. 2016 | El Asri, He, Suleman. *A Sequence-to-Sequence Model for User Simulation in Spoken Dialogue Systems*. Interspeech 2016 | RWS:19 |
| Kreyssig et al. 2018 | Kreyssig et al. *Neural User Simulation for Corpus-based Policy Optimisation*. SIGDIAL 2018 | RWS:19 |
| Kruff et al. 2026a (Sim4IA-Bench) | Kruff, Kreutz, Breuer, Schaer, Balog. *Sim4IA-Bench: A User Simulation Benchmark Suite for Next Query and Utterance Prediction*. ECIR 2026. arXiv:2511.09329 | RWS:19; bench [S4IA26] |
| Kruff et al. 2026b | Kruff, Bernard, Schaer. *Validating Search Query Simulations: A Taxonomy of Measures*. 2026. arXiv:2601.11412 (venue "ECIR 2026" not verified — TODO) | bench [KBS26] |
| Lai et al. 2025 (AdaRewriter) | Lai, Wu, Wang, Zhou. *AdaRewriter: … Conversational Query Reformulation via Test-Time Adaptation*. EMNLP 2025. arXiv:2506.01381 (exact title TODO) | RWS:19; AdaRewriter:1-2 |
| Li et al. 2024 (DAT) | Li, Wang, Viégas, Wattenberg. *Dialogue Action Tokens*. arXiv:2406.11978, 2024 (full title TODO) | RWS:19 |
| Liu et al. 2026 (MUSE) | Liu et al. *MUSE*. arXiv:2604.13828, 2026 (title TODO) | RWS:19 |
| Maxwell et al. 2015 | Maxwell et al. CIKM 2015 (title TODO) | PSS / README |
| Mo et al. 2026 (ConvAgent) | Mo et al. *Agentic Conversational Search*. ACL 2026. arXiv:2601.13115 | RWS:19 |
| Naous et al. 2026 (UserLM) | Naous, Laban, Xu, Neville. *Flipping the Dialogue: Training and Evaluating User Language Models*. ICLR 2026. arXiv:2510.06552 | RWS:19; UserLM:1 |
| Pirolli & Card 1999 | Pirolli, Card. 1999 (title/venue TODO) | PSS / README |
| PPDPP | Deng, Zhang, Lam, Ng, Chua. ICLR 2024. arXiv:2311.00262 (title TODO) | Sep-1st README |
| EPO | ACL 2025. arXiv:2502.12486 (authors/title TODO) | README |
| Schatzmann et al. 2007 | Schatzmann et al. *Agenda-Based User Simulation*. NAACL-HLT 2007 | RWS:19 |
| Schulman et al. 2017 (PPO) | Schulman 2017 (title/venue TODO) | RWS Appendix |
| Shao et al. 2024 (GRPO) | Shao et al. *DeepSeekMath*. arXiv:2402.03300, 2024 | RWS:19 |
| Sun et al. 2026 (DITTO) | Sun, Zhou, Liu et al. *Reinforcing Human Behavior Simulation via Verbal Feedback*. arXiv:2605.20506, 2026 (venue TODO) | RWS:19 |
| Wang et al. 2025 (USP) | Wang, Li, Yang, Zhou, Jiang, Li. *Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles*. ACL 2025 | RWS:19; USP:1 |
| Wang et al. 2026a (Turing-RL) | Wang et al. *Learning User Simulators with Turing Rewards*. arXiv:2606.19336, 2026 (venue TODO) | RWS:19 |
| Wang et al. 2026b (ProUtt) | Wang et al. *LLM-Driven Preference Data Synthesis for Proactive Prediction of the Next User Utterance in Human-Machine Dialogue*. arXiv:2601.09713, 2026 | DITTO deck:21 |
| Wu et al. 2026 (HumanLM) | Wu, Choi, Khatua et al. *HumanLM*. arXiv:2603.03303, 2026 (title TODO) | RWS:19 |
| Zerhoudi 2026 (SimEval-IR) | Zerhoudi. *SimEval-IR*. SIGIR 2026 (title TODO) | RWS:19 |
| Zhang et al. 2026 (UserLM-R1) | Zhang, Li, Zhang et al. *Modeling Human Reasoning in User Language Models with Multi-Reward Reinforcement Learning*. arXiv:2601.09215, 2026 | RWS:19; UserLM-R1:1 |
| Zhou et al. 2026 | Zhou, Sun, Ma et al. *Mind the Sim2Real Gap in User Simulation*. COLM 2026 | RWS:19 |
| TREC 2026 UserSim | Guidelines, https://trec.usersim.ai/guidelines (2026-09-23 version) | bench metric_specs |

---

## Sources

- 自己的報告：Google Drive「TREC-UserSim」R1–R4（2026-07-13 / 07-27 / 08-10 / 08-28）。
- Internal Meeting 投影片：MingZhi（Related Work Summary、User Simulator for TREC 2026 UserSim、Planner_Speaker_Selector、Dynamic-Reasoning Planner、v2fix_to_E1.6 …）；組員 HaoCheng / HungChun / KuanWei 的論文報告與重現。
- `/home/mzjiang/Sep-1st-Simulator/README.md`、`REPRO.md`。
- 學長 benchmark `/tmp2/hchsu/trec2026-usersim-benchmark`：`README.md`、`leaderboard/leaderboard.md`、`docs/metric_specs.md`、`protocols/*.md`、`instruments/termination_probe_v2/README.md`、`NOTICE.md`。
- 現行系統：`ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`（stop-sft-stageB 分支）。
