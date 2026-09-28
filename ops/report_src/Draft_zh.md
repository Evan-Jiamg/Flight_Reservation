# 草稿（中文版）— Introduction 與 Related Work（rev. 3）

> 這是英文稿 `Draft.md` rev.3 的中文對照，內容逐段對應（含作者註），供確認內容用。
> `> 註` 是給作者的說明，定稿時刪除；所有數字與文獻都出自文末「來源」，文獻缺的欄位標 `TODO`。
> 版面：模板是 2 頁 extended abstract（Intro 約 250 words、沒有 Related Work 節）。本稿 Intro 約 690 words（含小標與 TODO）、Related Work 約 560 words，
> 放進模板時：Intro 刪到 背景 2 句／缺口 3 句／做法 2 句；Related Work 併成 Intro 裡約 80 words 的一段（見 §2 末的濃縮版）。

---

## 1. 引言（Introduction）

**背景。** 使用者模擬器越來越常被用來評估與訓練對話式資訊存取系統 [Balog & Zhai 2024]。TREC 2026 User Simulation Track 的場景是「對話式資料集搜尋」：一位研究者透過搜尋介面尋找合適的資料集 [Kreutz et al. 2025]。模擬器要嘛在給定部分對話歷史與使用者資訊需求下預測使用者的下一句話（Task 1），要嘛生成整段對話，並且自己決定何時目標已滿足、或何時放棄（Task 2）。賽道會將 query 長度、對話輪數、澄清請求的分佈與真實紀錄比較，並搭配人類 Turing test。

**缺口。** 我們稱一個模擬器具有「整段對話層級的保真度」（session-level faithful），是指它每一輪「是否結束」的決定與真人一致——重點是**何時**結束，而不只是**會不會**結束——從而它的對話長度分佈也與真人一致。近年的 LLM 模擬器在單輪保真度（意圖遵循、persona 一致性、風格）上已相當強 [Naous et al. 2026；Abdulhai et al. 2025；Wang et al. 2025]，但在本賽道資料上，這並不會延伸到整段對話層級。在我們實驗室的內部 benchmark（不是官方評估）中，一個 Turing-RL 配方的重現版 [Wang et al. 2026a]——其 act 轉移落在真人雜訊地板以內，act 分佈則與另一個系統並列 benchmark 中最接近真人（act TVD 0.185）——在重播完整真人對話後再給它一輪時**從不結束**（0/26 個 session），平均跑 9.9 輪，而真人平均 4.4 輪；公開的 UserLM-8b [Naous et al. 2026] 以其原生 end token 提示時，則在 72% 的「真人其實還繼續」的中間輪輸出 END，對話平均只有 1.04 輪；zero-shot 的 Ditto-8B 則接近真人平均（5.0 輪）。可見 act 層級保真度無法預測整段對話行為：act 分佈最接近真人的兩個模擬器之一從不結束，而離真人較遠的 Ditto-8B（act TVD 0.305）對話長度卻接近真人。結束行為也對探針設定很敏感：新版探針加入一條共用的結束指示（同時更新了 system agent 設定），就讓 Ditto-8B 的 K+1 end rate 從 0.04 變成 0.48。現有模擬器的 RL reward 不是逐句判斷，就是（USP 的）單一個對話層級 profile 相似度分數；沒有任何一個拿模擬器的對話長度或結束位置去和真人比較。

**我們的做法。** 我們使用 Planner–Speaker 架構的模擬器，其中「結束對話」是 Planner（Qwen3-4B-Instruct-2507）一個獨立且具約束力的決策；凍結的 Speaker（Ditto-8B，它自己的 end token 被遮蔽；Speaker 輸出空白也會結束 episode）產生候選句，由「長度＋風格」selector 排序挑選。這個設計的兩個免訓練版本說明了為什麼「結束」必須被學習。第一版中，Speaker 無視 Planner 的結束決定（被要求收尾時只有 8% 真的收尾），64 段中有 51 段撞到 10 輪上限。第二版在多項改動中包括讓 Planner 的決定具約束力，於是能夠收尾（K+1 end rate 0.05 → 0.48，對照的是去掉 annotations 的第一版），但結束得太早（premature end rate 0.03 → 0.14）。因此我們用 GRPO 的一個變體（以組平均為基準、不做標準差正規化）訓練 Planner，reward 全部是整段對話層級：一個 log-ratio 項，把模擬的輪數分佈推向訓練 fold 中真人的輪數分佈；由 LLM 判斷的需求涵蓋率；以及在訓練對話的真實決策點上，與真人「是否結束」決定的一致性。長度項的 advantage 只歸給 Planner 的「結束對話」token，另有一個針對同一批結束決策的輔助監督損失（權重下限 0.5）穩定訓練。

> 註：下一段等 fold 2 test（u5 vs u0）與其他 fold 跑完再填；目前沒有任何可引用的 v16 結果。比較協定（AUDIT_SPEC）是對 E1.6 在同一批 test sessions 上重新計分，**目前沒有核准其他 baseline**；若要與 benchmark 其他方法同表，須先證明我們 Task 2 環境（自架 R0／ledger judge）與 benchmark 的 system agent 設定（prompt v4、length_retry_v1）一致。輪數要寫清楚是 W1 還是 benchmark 的平均輪數（錨 4.446），或兩個都報。

**主要發現。** `TODO`（用 fold 2 test 與三折合併評估的結果；報 paired bootstrap 區間並註明 n。）

**貢獻。**
- 我們在本賽道資料上指出：act 層級保真度無法預測整段對話層級保真度——整段對話行為從「從不結束」到「幾乎馬上結束」都有，與 act 保真度無關，而且對提示與探針設定很敏感。
- 我們讓「結束對話」成為 Planner 獨立且具約束力的決策，並以 GRPO 針對真人的對話長度與結束位置訓練它、把 credit 歸給結束決策，而不是用逐句 reward。
- `TODO`（結果；在 goal 與 persona 都不重疊的 test sessions 上，於相同協定下與我們的免訓練版本比較。）

> 註：
> 1. benchmark 是實驗室內部量測工具，NOTICE 說它「不是可引用的出版物；請引用 Track 與原始論文」。致謝 Lucas H.-C. Hsu（Sep-1st README §7），**不要引用 repo**。
> 2. Turing-RL 的數字：probe v2、三個 fold test side 共 26 個 session（K+1 0/26、rollout 自己結束 1/26、平均 9.885 輪），`instruments/termination_probe_v2/README.md` 與 `leaderboard.md`。真人雜訊地板：act TVD（F1）0.155、transition JSD（F2）0.19——Turing-RL 的 transition JSD 0.121 在地板內，act TVD 0.185 **在地板之上**（但與 A1-s1 的 0.182 並列最接近真人）。Ditto 的提示敏感度：加一條共用 TERMINATION_INSTRUCTION 讓全語料 K+1 從 0.0357 變 0.4821（probe README 49-52；v1 的 system agent 也是舊設定，所以不能全歸功於那條指示）。它的「結束」在自己的格式裡是空訊息，所以「不結束」可能部分來自重現方式——正文已寫 re-implementation，必要時再加一句 hedge。
> 3. UserLM-8b：**未入榜**，用它原生的 end token、沒有共用的 TERMINATION_INSTRUCTION（probe README 74-76）；72% = `teacher_forced_turn_with_end_decision_rate`（分母是所有真人繼續的中間輪）。Ditto-8B：K+1 0.577、4.962 輪，是 F10 family 最佳，而且就是我們的凍結 Speaker——reviewer 會問「為何不直接用 Ditto」，答案要在 Results 用「停在哪一輪」（premature、stop AUC）而不只是平均輪數來回答。
> 4. 兩個免訓練版本的數字是**我們自己在 benchmark 較早協定下**的量測（v2fix：Qwen2.5-32B planner + UserLM-8b speaker，64 episodes；E1.6：K+1 0.0536→0.4821、premature 0.0321→0.1446，出自 v2fix_to_E1.6 投影片 p.4-5），**不可和上面 probe v2 的數字並列成同一張表**。8% 與 51/64 屬於 v2fix；0.0536 屬於 v2fix 去掉 annotations 的版本（原 v2fix 是 0.0179）；v2fix → E1.6 改了五件事（annotations、stop 權限、speaker 輸入、role header、開場取樣），所以正文寫「among other changes」。兩版用的模型也不同（v2fix：Qwen2.5-32B planner＋UserLM-8b speaker；E1.6 的 speaker 來源未寫明，請確認）。
> 5. 方法描述對應 AUDIT_SPEC / SPEC_v16：Task 1 stop groups 用 train_all 對話（不是 held-out）；validation 只用來挑 checkpoint（bal_p）；coverage 由 gpt-oss-120b ledger judge 判斷；Dr. GRPO 需要引文（`TODO cite`，來源裡沒有書目）。
> 6. 實驗室組員的 Unified Framework 投影片提出過 user 端的 termination reward（λ4·r_term），A1-s1 是他在同一 benchmark 的 RL 模擬器。**我們不宣稱「第一個獎勵 user 端結束」**，新穎性放在「對真人長度分佈與真人結束位置做最佳化、並把 credit 給結束決策」。若要正面比較，請確認 A1-s1 是否用了 r_term。
> 7. 官方 Task 1 是 next-utterance prediction；我們自己的 Task 1 結束決策指標（M2 mapping, decision D7）不是官方指標；benchmark 的 `termination_f1` 已於 9/26 退役。

---

## 2. 相關研究（Related Work）

**LLM 使用者模擬器。** 使用者模擬從 agenda-based [Schatzmann et al. 2007] 與神經序列模型 [El Asri et al. 2016；Kreyssig et al. 2018]，演進到直接訓練 LLM 扮演使用者 [Balog & Zhai 2024]。USP [Wang et al. 2025] 以每段對話抽出的隱含 profile 為條件生成（我們的 Planner 也維護類似的隱含 profile）；UserLM-8b [Naous et al. 2026] 把助理對話翻轉過來，訓練出帶有「結束對話」token 的使用者模型，並記錄到被提示的助理模型「不願結束對話」的現象；HumanLM [Wu et al. 2026] 對齊使用者的潛在狀態；MUSE [Liu et al. 2026] 以迭代自我批判、對照真實對話來最佳化 profile；ProUtt [Wang et al. 2026b] 預測使用者下一步的意圖路徑。這些方法大多以單輪層級評估；UserLM 另外評分「結束」，它透過模仿真實的結束位置學會結束，且需要護欄防止太早結束。

**以 RL 訓練使用者模擬器。** ConsistentPersona [Abdulhai et al. 2025] 用多輪 PPO，reward 是判官給的 persona 一致性分數；USP 的 RLCC 階段 [Wang et al. 2025] 獎勵一個對話層級的 profile 相似度（cycle consistency）分數（複製到每個使用者輪）加上擬人程度；UserLM-R1 [Zhang et al. 2026] 以 GRPO 結合規則與 rubric reward，並用 LLM 判斷「掛斷時機」；Turing-RL [Wang et al. 2026a] 以成對比較的 Turing 式判官作為 GRPO reward；MUSE 在 GRPO 下把逐輪 rubric reward 在整段對話上平均；DITTO [Sun et al. 2026] 在 GRPO 中加入文字回饋。對話長度要嘛固定（ConsistentPersona 的 10、20、40 或 60 輪），要嘛設上限（USP，最多 10 輪），要嘛由 LLM 判斷；這些 reward 都沒有拿對話長度或結束位置去和真人比較。用小型訓練過的 planner 操控凍結的生成器已有先例——Dialogue Action Tokens [Li et al. 2024]、PPDPP [Deng et al. 2024] 與 EPO（`TODO` 作者，ACL 2025）——但沒有一個訓練的是**使用者**的結束決策：PPDPP 規劃的是系統端的行動，丟掉 CraigslistBargain 的終止類 act 並把對話上限設為 8 輪，DAT 則把「離開聊天」列為未來方向。因此我們不宣稱 planner–生成器的拆分本身是新的，新的只在於把它用在使用者端、並訓練使用者的結束決策。

**評估與結束行為。** Sim4IA-Bench [Kruff et al. 2026a] 在真實搜尋 session 上評分「下一個 query」與「下一句話」的預測，但每個任務都是單步預測；Bernard & Balog [2024] 將對話式資訊存取中的模擬目標形式化；Kruff et al. [2026b] 提出驗證 query 模擬的量測分類；clem:todd [Chalamalasetti et al. 2025] 評測「模擬器 × 對話系統」的組合，SimEval-IR [Zerhoudi 2026] 則把行為真實度與測試者可靠度分開。Zhou et al. [2026] 發現模擬使用者比真人更合作、更早透露任務資訊，並建議分開報告行為、任務結果與主觀評分——這是整段對話層級的真實度落差。在互動式資訊檢索中，使用者何時停止長期以 stopping rules [Cooper 1973；Kraft & Lee 1979；Maxwell et al. 2015] 與資訊覓食理論 [Charnov 1976；Pirolli & Card 1999] 建模；LLM 模擬器則只透過模仿（UserLM 的 end token）或 LLM 判斷的 rubric 學會結束，沒有一個是針對真人的對話長度分佈最佳化的。最後，對話式搜尋系統越來越常以學到的或互動式的回饋來最佳化——例如 query 改寫的 reward-model 重排序 [Lai et al. 2025]、以 GRPO 訓練的 agentic 搜尋 [Mo et al. 2026]——因此模擬器能否真實地結束對話，對訓練與評估這類系統很重要。

> 註：
> - 作者先前在 Sep-1st 試過具名 stopping rules，**沒贏過單純的輪數計數器**（F1 0.442 vs 0.524/0.559，Sep-1st README）；benchmark 的 stop judgement 也顯示 `rule_turn_count` AUC 0.809。Results 要把「停在哪一輪」和這個 turn-count / hazard 基線比（不只和 E1.6 比），reviewer 會要求。
> - UserLM 的 termination F1（論文 63.54）只出自組員轉述，要引請回原論文確認。
> - UserRL [Qian 2025]、UserSimCRS v2 [Bernard & Balog 2026]、Chopra [2026]《Beyond Cooperative Simulators》在來源裡**只有標題與 venue**，沒有內容描述，所以**已從正文移除**；讀過原文、能寫出一句正確描述後再放回（Chopra 可能與 Zhou et al. 並列，UserRL 可能放 RL 段）。
> - 空白的 Speaker 輸出也會結束 episode（AUDIT_SPEC），Method 節要寫清楚，避免讀者以為只有 Planner 能結束。
> - UnifiedFramework 說「termination reward 在 task-oriented dialogue RL 是標準做法（訓練 system 端）」但沒給出處——**不寫進正文**。

**2 頁模板用的濃縮版（約 80 words）。** 使用者模擬器透過 SFT [Naous et al. 2026]，或以逐句、rubric 或對話層級相似度 reward 的 RL [Abdulhai et al. 2025；Wang et al. 2025；Zhang et al. 2026；Wang et al. 2026a] 學到單輪保真度；它們的結束方式是模仿、LLM 判斷的 rubric 或上限。benchmark 評分的是單步預測 [Kruff et al. 2026a]；模擬使用者過度合作 [Zhou et al. 2026]。資訊檢索為「停止」建模 [Maxwell et al. 2015；Pirolli & Card 1999]，但沒有模擬器的 reward 是針對真人的對話長度分佈。

---

## 參考文獻

與英文稿 `Draft.md` 文末的表格相同（書目照來源抄寫，缺的欄位標 TODO），此處不重複。

## 來源

- 自己的報告：Google Drive「TREC-UserSim」R1–R4（2026-07-13 / 07-27 / 08-10 / 08-28）。
- Internal Meeting 投影片：MingZhi（Related Work Summary、Planner_Speaker_Selector、v2fix_to_E1.6、MUSE、Agentic Conversational Search via RL …）；組員 HaoCheng / HungChun / KuanWei 的論文報告與重現。
- `/home/mzjiang/Sep-1st-Simulator/README.md`、`REPRO.md`。
- 實驗室內部 benchmark `/tmp2/hchsu/trec2026-usersim-benchmark`：`README.md`、`leaderboard/leaderboard.md`、`docs/metric_specs.md`、`protocols/*.md`、`instruments/termination_probe_v2/README.md`、`NOTICE.md`。
- 現行系統：`ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`（stop-sft-stageB 分支）。
