# TREC 2026 User Simulator：Ditto + Planner 停止決策 + 學習式 Goal 滿足判斷器 — 完整執行計畫

版本：2026-09-24 19:10（Asia/Taipei）。**狀態：待確認，尚未執行。** 已在跑的工作（見 §1.3）不屬於本計畫的新動作。
既有規範沿用：`TRAINING_PLAN_CLAUDE_CODE_20260923.md` 的防洩漏規則、`PREREG_STAGE_BC_20260924.md` 的事先登記紀錄。本計畫取代其中 Stage D（gate 的 GRPO / 精確梯度）與 D2（長度擬合），理由見 §1.2。

---

## 1. 目前為止的證據（全部可追到伺服器原始檔）

### 1.1 已完成且核對過

| 項目 | 結果 | 位置（245 除非註明） |
|---|---|---|
| Stage A PRISM SFT | 504 updates；epoch2 NLL 0.384；同環境在 221 重評 0.385（跨機器一致） | `grpo_planner/prism_sft_7b_v1/`、221 `stageA_orig_ep2_on221/` |
| Stage B 三折 TREC 適應 | 全選 epoch2；合併 12 session AUC 0.681；0.5 閾值下 K+1 0/12 | `trec_inner_fold{0,1,2}_7b_v1/`、`analysis_v1/` |
| Task 2 runner 修正 | 終止時序修正、scenario 層級統計、離線截斷＝線上 gated run（max \|Δp\| 3.7e-7） | `code_snapshots/stageC_v1/`、`stageC_v1/smoke_crn/equivalence.json` |
| UserLM no-gate（rep0，68 集） | 8.71 輪、coverage 0.192、complete 5/68；43% 發話為離題指令句、43% 完全重複 | `stageC_v1/rep0/` |
| Ditto no-gate（rep0，68 集） | 10.0 輪、coverage 0.705、complete 0.191；0% 離題句、7% 重複、38% 首句跨 seed 相同；從不自行結束 | `stageC_v1/rep0_ditto_shard*/` |

### 1.2 決定方向的三個發現

1. **gate 是位置計數器，不是內容判斷。** PRISM val：只看輪數的 AUC 0.849 > gate 0.814；同一輪內 AUC 0.451。TREC：遮掉 prompt 裡的輪數與 StoppingLedger 規則行後，三折 AUC 0.766→0.688、0.597→0.444、0.663→0.575。
2. **真人停止時機無法從內容預測（PRISM）。** 未訓練的 Qwen2.5-7B 特徵 + 輪數固定效應的線性 probe，12 組設定的同一輪內 AUC 都在 0.44–0.47，95% 信賴區間都含 0.5；Cox SFT epoch1 為 0.449。
   ⇒ 繼續訓練「預測真人何時停」的 gate 沒有意義，也無法跨資料集泛化。
3. **在 Ditto 上，凍結的 Planner 已經會在接近真人的時間點下結束 Act。** 66/68 集有結束 Act，平均第 5.02 輪（真人 ≈4.5），37/66 在真人輪數 ±1 內；結束理由以 satiation 43、rate_of_gain 13 為主。照 Planner 結束的反事實：與真人輪數的絕對誤差 5.47→**2.07**，coverage −0.027 [−0.048, −0.008]（略超守門線 −0.02），complete −0.015。57/66 在結束後就沒有新進展；9/66 屬過早結束。
   但 Planner 對「目標是否已滿足」的資訊來自**規則**：關鍵字比對的 `pending`，以及 StoppingLedger 的計數與停止條件。另外有 2/680 步的 prompt 超過 12000 tokens，會從尾端截斷，砍掉最新的對話（bug）。

### 1.3 尚在執行中的工作（計畫確認前不會再加新的）

| 主機 | 工作 | 用途 |
|---|---|---|
| 221 | Stage A′ Cox SFT epoch 2 | 補完發現 2 的最終數字 |
| 244 GPU0 | Stage A′ 不加位移對照組 | 同上 |
| 245 | Ditto replicate 1（68 集，4 shard） | 本計畫的第二次環境重複 |
| 245 | Ditto rep0 離線停止策略分析（gate / 明確輪數 hazard） | 本計畫的 baseline 之一 |

---

## 2. 新架構

```
Ditto-8B Speaker（凍結）
      ▲  Planner block（act / move / stop_rule / 長度…）
Qwen2.5-32B JSON Planner（凍結；PLANNER_END=1：其結束 Act 真正終止對話）
      ▲  prompt 中原本的規則行 → 換成「GOAL STATUS（學習式判斷）」
Goal 滿足判斷器：Qwen3-4B-Instruct-2507 + LoRA（新訓練）
      ▲  輸入：使用者的 goal（WHO THEY ARE / WHAT THEY CAME FOR）＋ 截至本輪的完整對話
```

- **停止由 Planner 決定。** 判斷器只提供「目標滿足到什麼程度、還缺什麼」，不直接輸出停不停。停止仍是 Planner 的 Act（settle / abandon），由 Ditto 說出收尾語。
- **不讀輪數、不讀規則。** 判斷器輸入不含輪數與 StoppingLedger 規則輸出；Planner prompt 也移除這些行（附錄 D 的遮蔽清單：turns so far、useful/unhelpful 計數、stopping condition、THEY HAVE SENT N MESSAGES、repeated offer；persona 的「gives up」只去掉括號內的規則說明）。
- **截斷修正。** Planner prompt 超過上限時改成保留 goal/state 前綴與最新對話、壓縮最舊的歷史（與 `stop_prompt.py` 的 TREC 策略相同），不再從尾端截斷。

---

## 3. 資料與標籤

### 3.1 訓練單位
每筆樣本 =（episode, 第 t 輪 R0 回覆之後）。
- 輸入：scenario 的 goal 與 persona 文字，加上第 1…t 輪完整對話（USER/ASSISTANT）。
- 輸出：`{"status": "SATISFIED|PARTIAL|NOT", "score": 0–1, "unmet": [...最多 3 條簡述]}`。

### 3.2 樣本來源（全部是已記錄的軌跡，不需要新 rollout）

| 來源 | 集數 | 約略樣本數 |
|---|---|---|
| Ditto rep0 / rep1 | 68 + 68 | ~1300 |
| UserLM rep0 / rep1（部分） | 68 + 45 | ~1000 |
| TREC 真人 gold sessions | inner 側 | ~250 |

- 只用 **inner train / inner validation** 的 scenario（union run 本來就不含任何 outer test）。
- 每折分開訓練：該折 inner train 訓練、inner validation 選模；三折彼此獨立。

### 3.3 標籤 judge（與評估尺分開）
- **主標籤：Meta-Llama-3.1-70B-Instruct**（本地、NF4、245）。
  - 和評估用的 gpt-5-mini ledger 不同家族。
  - 主域資料不送第三方。
- **judge 看的是什麼：** 只看 goal 文字與對話，**不看 benchmark 的 requirement shards**。shards 只保留給評估（coverage/complete），徹底把標籤和評估尺分開。
- **標籤品質檢查：**
  - 隨機 200 筆由 gpt-oss-120b 再標一次，報告兩者一致率與 Cohen's κ。
  - 另抽 30 筆請你人工檢查（見 §8）。
  - κ < 0.4 則停下檢討標籤定義，不進訓練。
- 標籤 prompt、模型版本、每筆 request hash 全部保存。

### 3.4 跨資料集
- MultiWOZ（本地只有官方 test）：**只做評估、絕不訓練**。對 test 對話逐輪跑判斷器，檢驗「目標滿足」能不能在同一輪內預測對話結束（見 §5.2）。
- PRISM 沒有明確 goal，不適用判斷器；只保留在發現 2 的證據裡。

---

## 4. 判斷器訓練

- 模型：Qwen3-4B-Instruct-2507，bf16，LoRA r=16、alpha=32，作用在 q/k/v/o；max length 4096（保留 goal 前綴 + 最新對話的壓縮策略）。
- 目標：生成 JSON 的交叉熵；另外記錄 status 三個類別的機率，以便評估校準度。
- 超參數起點：lr 1e-4、3 epochs、effective batch 16、seed 20260924；每 epoch 存 checkpoint。
- 選模規則（事先固定）：inner validation 上 status 的 macro-F1 最高；同分取 score 的 Brier 分數較低者。
- 主機：245（bf16）。4B 模型約 10–14GB，可以和 rollout 共存；若 245 太滿則用 244 V100（fp16），並在報告中標明。
- 「LLM 控制超參數」（你先前提過）：列為**可選**的消融。和隨機搜尋在相同預算（例如 6 組設定）下比較，只看 inner-train 的彙總數字。預設**不做**，除非你勾選（§8）。

---

## 5. 評估

### 5.1 Task 2 rollout（Ditto，34 scenarios × 2 seeds × 2 replicates，同一個 R0 與 runner）

| Arm | 說明 | 成本 |
|---|---|---|
| A0 | Ditto no-gate | 已有（rep0；rep1 執行中） |
| A1 | Ditto + Planner 結束（原本規則式 prompt） | 精確截斷，免費 |
| A2 | Ditto + Planner 結束 + 遮蔽規則行 + **學習式 GOAL STATUS** | 新 rollout，每個 replicate 約 2.5 小時 |
| A3 | Ditto + Planner 結束 + 遮蔽規則行、**不給判斷器** | 新 rollout；用來分離判斷器本身的貢獻 |
| A4 | 明確的輪數 hazard（不看內容） | 離線，免費 |

- **主要指標：** 與真人輪數 K 的絕對誤差與有號誤差；coverage；complete（gpt-5-mini ledger，只用於評估）。
- **次要指標：** 結束 Act 的類別分佈；過早結束率（結束後在對照軌跡中仍有進展的比例，只能在 A1/A4 精確量）；Act TVD、首句重複、長度 W1（benchmark scorer）。
- **統計：** scenario 層級配對差、SE、cluster bootstrap；兩個 replicate 分開報告再合併。
- **品質守門（沿用）：** 與 A0 相比 coverage 不低於 −0.02，complete 最多少 1 集；不符合就報告為取捨或失敗，不因輪數變少而宣稱改善。
- **成功的定義：** A2 的輪數誤差不劣於 A1，並且通過品質守門（A1 目前 coverage −0.027，沒過）；同時 A2 優於 A3，才能歸功於判斷器。

### 5.2 判斷器本身
- inner validation 上：與標籤 judge 的一致率、macro-F1、校準度。
- **泛化檢驗（核心）：** 判斷器的 score 在「同一輪內」能不能區分真人最後一輪（K）與之前各輪。在 TREC gold sessions 與 MultiWOZ test 上分別計算，並和只看輪數的 baseline 比較。
  - 若同一輪內 AUC 顯著高於 0.5，代表「目標滿足」確實是跨資料集可用的停止訊號。
  - 若沒有，照實報告。停止仍由 Planner 做，但不宣稱它學到了真人的停止時機。

### 5.3 outer test
- 方法（判斷器 checkpoint、Planner prompt v3、arm 定義）凍結後，才用各折 outer train 全部重訓判斷器，對 outer test 只跑一次。
- fold0 標為探索性；fold1/2 為主要估計。

---

## 6. 之後（可選，另行確認）：RL
判斷器與 Planner prompt 穩定後，才考慮對 **Planner 的停止決策**做 RL：reward = 輪數誤差 + coverage/complete 限制（Lagrangian），並用第 1 次研究報告建議的精確期望梯度或 GRPO。這一步需要 32B Planner 的 LoRA，而且每次都要重新 rollout，成本高。**本計畫不包含，確認前不做。**

---

## 7. 算力分配與時程（估計值）

| 步驟 | 主機 | 預估 |
|---|---|---|
| S0 等 §1.3 的工作跑完 | 221 / 244 / 245 | 已在跑，約 2–5 小時 |
| S1 標籤：Llama-3.1-70B NF4 對約 2500 筆逐輪樣本 | 245 GPU（另一張 GPU 給 rollout） | 約 3–4 小時 |
| S1b 標籤品質：gpt-oss-120b 標 200 筆 + 人工 30 筆 | 245 / 你 | 約 1 小時 + 你的時間 |
| S2 判斷器三折訓練 | 245（或 244 V100） | 每折約 30–60 分鐘 |
| S3 Planner prompt v3 + 截斷修正 + 單元測試 + smoke | 本機 / 245 | 約 1 小時 |
| S4 A2 / A3 rollout × 2 replicates（4 shard 平行） | 245 | 約 10 小時 |
| S5 分析、MultiWOZ 泛化檢驗、報告 | 245 CPU / 221 | 約 2 小時 |
| S6 方法凍結後 outer test（一次） | 245 | 約 5 小時 |

221 的 P40 與 244 的 V100 只能跑 7B 以下的模型；Task 2 rollout（32B + 8B）只能在 245 跑。238/243 沒有可用的 GPU。

---

## 8. 需要你確認的決定

1. **架構**：§2（判斷器只提供資訊，由 Planner 決定停止；PLANNER_END=1）是否同意？
2. **標籤 judge**：主標籤用 Llama-3.1-70B-Instruct、交叉檢查用 gpt-oss-120b，judge **不看** requirement shards，是否同意？
3. **人工檢查**：你是否願意檢查 30 筆標籤？（不願意的話，就只靠兩個 judge 的一致率。）
4. **遮蔽清單**：§2 列出要從 Planner prompt 移除的規則行，是否同意？是否還有其他欄位你認為屬於「寫死的規則」？
5. **LLM 控制超參數**：要不要做成 §4 的可選消融？
6. **UserLM 軌跡**：你已決定不再跑新的 UserLM。已經跑完的 UserLM 軌跡能不能當判斷器的訓練樣本（多約 1000 筆、內容類型較雜）？
7. **RL（§6）**：同意先不做嗎？

---

## 9. 風險與已知限制

- **判斷器學到標籤 judge 的偏誤：** 用第二個 judge 的一致率和人工檢查監控。
- **Planner 可能忽略新的 GOAL STATUS 行：** A2 對 A3 的比較會直接顯示；若沒有差異，就照實報告。
- **Ditto 的既有問題：** 38% 首句跨 seed 相同、長度較長、比 UserLM 容易被認出不是真人（benchmark AUC 0.769）。本計畫不處理，照實報告。
- **環境差異：** Ditto 只能在 consistent-test 環境跑（transformers 5.7.0），Planner 在不同環境下的輸出會不同（只有 14/68 集第一輪相同）。所以所有 Ditto arm 都在同一個環境跑，不和 UserLM 做跨環境的直接比較。
- **樣本很少：** 每折 inner validation 只有 1–4 個 scenario；主要結論以合併三折、兩個 replicate、scenario 層級的區間為準。
- **R0 是 gpt-5-mini 模擬的 agent：** 結論限定在這個環境下成立。
