# 草稿（中文版）— Introduction、Related Work 與 Methodology（rev. 4）

> 這是英文稿 `Draft.md` rev.4 的中文對照，內容逐段對應（含作者註），供確認內容用。
> `> 註` 是給作者的說明，定稿時刪除；所有數字與文獻都出自文末「來源」，文獻缺的欄位標 `TODO`。
> 版面：模板是 2 頁 extended abstract（Intro 約 250 words、沒有 Related Work 節）。本稿 Intro 約 690 words（含小標與 TODO）、Related Work 約 560 words，
> 放進模板時：Intro 刪到 背景 2 句／缺口 3 句／做法 2 句；Related Work 併成 Intro 裡約 80 words 的一段（見 §2 末的濃縮版）。

---

## 1. 引言（Introduction）

**背景。** 使用者模擬器越來越常被用來評估與訓練對話式資訊存取系統 [Balog & Zhai 2024]。TREC 2026 User Simulation Track 的場景是「對話式資料集搜尋」：一位研究者透過搜尋介面尋找合適的資料集 [Kreutz et al. 2025]。模擬器要嘛在給定部分對話歷史與使用者資訊需求下預測使用者的下一句話（Task 1），要嘛生成整段對話，並且自己決定何時目標已滿足、或何時放棄（Task 2）。賽道會將 query 長度、對話輪數、澄清請求的分佈與真實紀錄比較，並搭配人類 Turing test。

**缺口。** 我們稱一個模擬器具有「整段對話層級的保真度」（session-level faithful），是指它每一輪「是否結束」的決定與真人一致——重點是**何時**結束，而不只是**會不會**結束——從而它的對話長度分佈也與真人一致。近年的 LLM 模擬器在單輪保真度（意圖遵循、persona 一致性、風格）上已相當強 [Naous et al. 2026；Abdulhai et al. 2025；Wang et al. 2025]，但在本賽道資料上，這並不會延伸到整段對話層級。在我們實驗室的內部 benchmark（不是官方評估）中，一個 Turing-RL 配方的重現版 [Wang et al. 2026a]——其 act 轉移落在真人雜訊地板以內，act 分佈則與另一個系統並列 benchmark 中最接近真人（act TVD 0.185）——在重播完整真人對話後再給它一輪時**從不結束**（0/26 個 session），平均跑 9.9 輪，而真人平均 4.4 輪；公開的 UserLM-8b [Naous et al. 2026] 以其原生 end token 提示時，則在 72% 的「真人其實還繼續」的中間輪輸出 END，對話平均只有 1.04 輪；zero-shot 的 Ditto-8B 則接近真人平均（5.0 輪）。可見 act 層級保真度無法預測整段對話行為：act 分佈最接近真人的兩個模擬器之一從不結束，而離真人較遠的 Ditto-8B（act TVD 0.305）對話長度卻接近真人。結束行為也對探針設定很敏感：新版探針加入一條共用的結束指示（同時更新了 system agent 設定），就讓 Ditto-8B 的 K+1 end rate 從 0.04 變成 0.48。現有模擬器的 RL reward 不是逐句判斷，就是（USP 的）單一個對話層級 profile 相似度分數；沒有任何一個拿模擬器的對話長度或結束位置去和真人比較。

**我們的做法。** 我們使用 Planner–Speaker 架構的模擬器，其中「結束對話」是 Planner（Qwen3-4B-Instruct-2507）一個獨立且具約束力的決策；凍結的 Speaker（Ditto-8B，它沒有結束對話的 token；依 benchmark 慣例，Speaker 輸出空白也會結束 episode）產生候選句，由「長度＋風格」selector 排序挑選。這個設計的兩個免訓練版本說明了為什麼「結束」必須被學習。第一版中，Speaker 無視 Planner 的結束決定（被要求收尾時只有 8% 真的收尾），64 段中有 51 段撞到 10 輪上限。第二版在多項改動中包括讓 Planner 的決定具約束力，於是能夠收尾（K+1 end rate 0.05 → 0.48，對照的是去掉 annotations 的第一版），但結束得太早（premature end rate 0.03 → 0.14）。因此我們用 GRPO 的一個變體（以組平均為基準、不做標準差正規化）訓練 Planner，reward 全部是整段對話層級：一個 log-ratio 項，把模擬的輪數分佈推向訓練 fold 中真人的輪數分佈；由 LLM 判斷的需求涵蓋率；以及在訓練對話的真實決策點上，與真人「是否結束」決定的一致性。長度項的 advantage 只歸給 Planner 的「結束對話」token，另有一個針對同一批結束決策的輔助監督損失（權重下限 0.5）穩定訓練。

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

## 3. 方法（Methodology）

> 註：本節只寫**已實作且在正式 run 路徑上**的設計（v16，`sep-sim/` 程式與 `ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`）；每個數值都對應一個程式常數或 SPEC 值（見本節末的註）。放進 2 頁模板時保留 3.2 前兩句、式 (1)(2)(3) 與 3.4 的選擇分數，其餘移到附錄或全文版。

### 3.1 任務與資料

我們依照賽道在對話式資料集搜尋語料上的兩個任務。**Task 2** 中，模擬器拿到 persona 與目標（主題、情境、使用者已知道的資料集），和一個 task agent 對話，直到它結束對話或達到 T_max = 10 則使用者訊息的上限。**Task 1** 中，模擬器以真實對話到第 t−1 則訊息為條件，產生第 t 則；同一次執行也得到它在每個真實回合「是否結束」的決定，我們把它當作結束決策來評分（這是我們自己的對應方式，不是官方的 Task 1 指標）。訓練時，task agent 與評分涵蓋率的需求帳本（requirement ledger）都是本機的 gpt-oss-120b。我們採用 benchmark 的 goal 與 persona 都不重疊的三折切分：每個 fold 中，訓練用 `train`（有需求標註的對話，用於 Task 2 rollout）與 `train_all`（全部訓練對話，用於 Task 1 結束群組、few-shot 範例池與真人長度分佈）；`validation` 用來挑 checkpoint，並透過一個觸發條件啟動輔助結束損失的退火（§3.3）；`test` 在訓練結束後讀一次，做最終評估。validation 與 test 的對話從不用來跑產生梯度的 rollout，也不會進入 few-shot 範例池、長度分佈或控制器。

### 3.2 模擬器

**Planner。** 每一輪，Planner（Qwen3-4B-Instruct-2507 加上一個 LoRA adapter，是唯一被訓練的元件）讀取目標、目前為止的對話與它自己先前的筆記，寫出一份 JSON 計畫：一句對自己先前狀態的批評、一份文字化的 dialogue move 分佈（每個 move 附目標長度）、目標是否已達成（`goal_met`：yes / partly / no）、使用者還想要什麼，以及 `end_session`——若正在規劃的訊息是使用者的最後一則則為 true。這則訊息的 act 從這份分佈抽出；若 `end_session` 為 false 卻抽到收尾（*Complete*）act，就改從其他項目重抽，讓訊息與決定一致。第一則訊息不能結束對話。若 `end_session` 為 true，這則訊息的 act 採用 Planner 自己機率最高的 *Complete* 項目，並告訴 Speaker 這是使用者的最後一則訊息；訊息送出後 episode 結束。

**隱含 profile。** 從第 2 輪起，Planner 也會寫一段 `profile_note`：用一兩句話描述*這個人*的寫作方式（長度、語氣、大小寫、措辭）中，它前一次的輸出寫錯了什麼。Task 1 中，它比較自己對第 t−1 則的預測與真實的第 t−1 則；Task 2 中，它批評自己前一則訊息。筆記在整段對話中累積，每一輪都傳給 Planner 與 Speaker。

**Speaker 與 selector。** 凍結的 Speaker（Ditto-8B）每輪根據 Planner 的計畫、筆記，以及每則候選各自的 k = 3 則寫作風格相同的使用者真實訊息（取自 `train_all` 中其他 goal 與 persona 的對話），寫出四則候選訊息（一則 greedy、三則以 temperature 0.7、top-p 0.9 取樣；第 1 輪四則都以該 checkpoint 自己的生成設定取樣）。違反守衛（例如連續照抄範例 8 個字以上）的候選會被剔除（重複的候選會重抽；若全部失敗，改以不帶範例的方式再抽），再由 Borda selector 以相同權重，依「與 Planner 目標長度的接近程度」與「與參考文本的 SimCSE 相似度」（Task 1 第 2 輪起用使用者自己先前的真實訊息，其餘情況用 few-shot 範例）為存活的候選排序。Ditto-8B 沒有結束對話的 token，所以對話只會透過 Planner 的決定、或依 benchmark 慣例在選出的 Speaker 訊息為空白時結束。

### 3.3 以 GRPO 訓練 Planner

每次更新都以目前的政策跑兩種 rollout，兩者都從 vLLM 提供的 adapter 取樣。

**Task 2 群組。** 每次更新抽 4 個訓練情境，每個情境跑 G = 4 個 episode（Planner temperature 1.0）；同一群組的 episode 共用情境 seed（Speaker 取樣、act 抽取、few-shot 範例），所以彼此只因 Planner 的取樣與 task agent 而不同。一個 episode 的 reward 是

  R = w_cov · coverage + w_dist · [log p_h(T) − log q(T)] − Σ_k λ_k · rate_k ，  (1)

其中 T 是使用者訊息數（上限 T_max）；p_h 是該 fold `train_all` 中真人訊息數的 add-α 平滑分佈（α = 1）；q 是本次更新乾淨 rollout 的同樣平滑分佈；coverage 是 episode 結束時 task agent 已處理的使用者需求比例（由帳本 LLM 判斷）；rate_k 是每步「計畫無法解析」與「計畫撞到生成上限」的比例。因為 E_q[log p_h − log q] = −KL(q ‖ p_h)，長度項在對話長度的*分佈*與真人一致時達到最大，而不是讓每段對話都同一個長度。task agent 回覆被截斷、帳本判決遺失，或送出的訊息撞到上限的 episode，不會進入群組。

**Task 1 結束群組。** 每次更新抽 8 段 `train_all` 對話；每段中，真實的最後一則訊息，以及（對話有 n ≥ 3 則訊息時）一則較早的訊息（t ≥ 2）是決策點。在每個決策點，Planner 拿到目前政策在一次 greedy teacher-forced 執行中到達的 prompt（真實歷史，加上它自己先前的狀態與筆記），以 temperature 1 取樣 G₁ = 8 份計畫；計畫的 `end_session` 與真人在該點的實際行為一致時 reward 為 1，否則為 0（無效計畫也是 0）。若有 reward 變異的群組數少於抽到的群組數，就再從 `train_all` 抽還沒用過的對話（最多再 8 段），直到有效群組數恢復。

**Advantage 與 credit。** 群組內使用群組平均為基準、不除以標準差（Dr. GRPO）：A_i = R_i − mean_j R_j；reward 沒有變異的群組直接跳過。Task 2 的 advantage 拆成長度項造成的部分 S_i = w_dist · [log p_h(T_i) − log q(T_i)] 與其餘部分：

  A_i^stop = S_i − mean_j S_j ，  A_i^seq = (R_i − S_i) − mean_j (R_j − S_j) 。  (2)

A_i^seq 套用在該 episode 每一份計畫的每個生成 token 上（`profile_note` 的 token 除外），A_i^stop 只加在真實決策點（t ≥ 2、可解析、未撞上限）的 `end_session` 值 token 上。Task 1 群組的 advantage 只作用在 `end_session` 值的 token 上。

**目標函數。** 令每個 token 的重要性比 ρ = π_θ / π_old，一個 minibatch 的損失為

  L = (1/N) Σ_tokens [ −w · min(ρ·A, clip(ρ, 1±0.2)·A) + β · (e^{d} − d − 1) ] + L_aux ，  d = log π_ref − log π_θ ，  (3)

其中 N 是生成的 token 數；β = 0.04 是對起始政策 π_ref 的 k3 KL 估計量權重；w = min(π_old / π_vLLM, 2) 是截斷的重要性權重，用來修正以 vLLM server 取樣造成的差異；若 |log π_old − log π_vLLM| 的平均超過 0.1，該次更新會中止。輔助項

  L_aux = −(w_eff / N_aux) Σ log p_θ(y* | prompt, 自己的前綴)  (4)

在基本群組的 Task 1 決策點上，監督真人的 `end_session` 值 y*（N_aux 是這些計畫的生成 token 數）。其權重 w_eff = max(0.5, w_aux · a_u)：在 balanced end probability（§3.4）連續兩次驗證都至少達到未訓練政策 + 0.10 之前，a_u = 1；之後 a_u = max(0.5, 1 − (u − u_trig)/10)。w_aux 初始為 1，並由控制器調整，所以監督權重永遠不低於 0.5。

**權重控制器。** 每 5 次更新，一個 LLM 控制器（gpt-oss-120b）讀取*訓練* rollout 的摘要統計，可以把 w_cov、w_dist、λ_unparsed、λ_hit_max_new 與 w_aux 各乘上 {0.5, 0.8, 1, 1.25, 2} 中的一個倍數，但不得超出固定的上下限（w_dist 只能從 1 往上調）；若固定權重的影子 reward 連續兩個視窗變差，就撤回改動。

**設定。** 初始權重 w_cov = w_dist = λ_unparsed = λ_hit_max_new = 1。LoRA rank 16（α = 32）加在 attention projection 上；π_ref 是不帶 adapter 的 base model（等於起始政策）；AdamW（無 weight decay）、learning rate 2e-5；每次更新做一次 optimizer step，梯度範數裁切於 1.0；HF 模型是 learner，並重新計算 vLLM server 生成的每個 token 的機率。

### 3.4 驗證與 checkpoint 選擇

未訓練的政策（update 0）與每第 5 次更新，會在該 fold 的 validation 對話上驗證：Task 2 用取樣的 Planner（temperature 0.7、seeds 0 與 1），Task 1 用 greedy。Task 1 在每個真實決策點 t 計算 Planner 結束對話的 teacher-forced 機率：以 greedy 計畫的前綴為條件，對 `end_session` 值計算 P_end = p(true) / (p(true) + p(false))，再算平衡分數

  bal_p = ½ · mean_{t = n} P_end + ½ · mean_{2 ≤ t < n} (1 − P_end) ，  (5)

它獎勵在真實最後一則結束、在那之前繼續（無效的點視為 P_end = 0；決策有效但找不到值 token 的點，以它的 greedy 決定計）。checkpoint 分數為

  score = coverage − W1(模擬輪數, validation 真人輪數) + bal_p ，  (6)

連續兩次驗證沒有進步就停止訓練（最多 30 次更新）。若某次驗證有 episode 重跑兩次後仍不乾淨，該次不給分數。由於 validation 很小（fold 2 有 4 段對話，fold 0 只有 1 段），驗證過的 checkpoint 會以 8 個 seed（0–7）重新驗證，選分數最高者（平手取較早的 update）；選定的 checkpoint 與未訓練政策接著在 test 上各評估一次。

> 註：
> - 式 (1)–(6) 對應：`rl_reward.reward_v4`、`turn_distribution`；`rl_algos.split_group_advantages`（std_norm False）；`TorchLearner.update`（clipped surrogate × TIS、k3 KL、除以生成 token 數、aux_backward）；`train_planner_rl.aux_weight / aux_annealed`；`task1_stop.task1_prob_metrics`；`validate()` 的 selection。
> - clip：每次 update 只做 1 epoch × 1 minibatch，所以第一步的 ratio 恆為 1（程式有斷言），clip 在實際上不作用；式 (3) 仍照程式寫出。
> - 「連續兩次驗證沒有進步就停、最多 30 次 update」與「8 seeds 重新驗證」是實驗腳本（`ops/v11ops/run_v16_formal.sh`、`run_v16_reselect.sh`）的規則，不是 trainer 內建；「8 seeds」是使用者 2026-09-27 核准的。fold 2 已完成重選（u0/u5/u10 → u5）；fold 0/1 的腳本（`run_v16_fold.sh`）同樣對所有驗證過的 checkpoint 做 8-seed 重選。fold 0 的 validation 只有 1 段對話，如何挑 checkpoint 使用者尚未決定。
> - coverage 是 reward 的一項（D3，已揭露）：因此訓練後的 coverage 不能當成獨立的評估指標，報告時要說明。
> - Dr. GRPO 需要引文（TODO）。

---

## 參考文獻

與英文稿 `Draft.md` 文末的表格相同（書目照來源抄寫，缺的欄位標 TODO），此處不重複。

## 來源

- 自己的報告：Google Drive「TREC-UserSim」R1–R4（2026-07-13 / 07-27 / 08-10 / 08-28）。
- Internal Meeting 投影片：MingZhi（Related Work Summary、Planner_Speaker_Selector、v2fix_to_E1.6、MUSE、Agentic Conversational Search via RL …）；組員 HaoCheng / HungChun / KuanWei 的論文報告與重現。
- `/home/mzjiang/Sep-1st-Simulator/README.md`、`REPRO.md`。
- 實驗室內部 benchmark `/tmp2/hchsu/trec2026-usersim-benchmark`：`README.md`、`leaderboard/leaderboard.md`、`docs/metric_specs.md`、`protocols/*.md`、`instruments/termination_probe_v2/README.md`、`NOTICE.md`。
- 現行系統：`ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`（stop-sft-stageB 分支）。
