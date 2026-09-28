# 架構 v3：完整過程、所有修正與證據、未改動項目的揭露

日期：2026-09-25。**狀態：架構程式已完成並通過測試；尚未執行任何實驗**（標籤產生、判斷器訓練、rollout 都還沒跑）。
本文件取代 `AUDIT_AND_PLAN_v2_20260925.md` 的架構部分。RL 與長跑流程的整體 Plan，等你確認本文件後再寫。
程式碼：worktree `C:\Users\User\.claude\worktrees\stop-sft-stageB\sep-sim\`，分支 `stop-sft-stageB`，commit `92fff37`；伺服器測試副本：245 `grpo_planner/dev_v3/`。

---

## 1. 完整過程（時間順序）

| # | 做了什麼 | 結果 | 現在的地位 |
|---|---|---|---|
| 1 | 核對交接的 Stage A（PRISM stop SFT） | 504 updates，epoch2 NLL 0.384 | 保留為紀錄；#6 證明它學到的是位置 |
| 2 | 三折 prompt parity 稽核 | 346 rows 全過 | 保留 |
| 3 | Stage B（TREC LoRA，三折） | 全選 epoch2；0.5 閾值下 K+1 0/12 | 保留為紀錄；gate 路線已停 |
| 4 | Task 2 runner v1（終止時序、scenario 統計、離線截斷推導） | 單元測試、線上/離線 smoke 一致 | 被 v3 取代；它繼承了 sepsim 的截斷 bug |
| 5 | UserLM no-gate 68 集 | coverage 0.192；43% 離題句 | 探索性；你已決定不再用 UserLM |
| 6 | 位置捷徑稽核、遮蔽消融、PRISM 內容 probe | 同一輪內 AUC 0.44–0.47（CI 含 0.5） | **主要研究發現**：真人停止時機無法從內容預測 |
| 7 | Stage A′（Cox）與對照組 | epoch1 同一輪內 AUC 0.449 / 0.467 | 同 #6；不再投入 gate 方向 |
| 8 | Ditto no-gate rep0 + rep1（各 68 集） | coverage 約 0.70 | 探索性（舊截斷 bug）；可作為判斷器的訓練樣本（§5-10） |
| 9 | Planner 結束稽核、架構稽核（讀完 sepsim 原始碼） | 找到 D1–D10 及三處寫死的決策規則 | 由本文件的修正處理 |
| 10 | 修正全部缺陷、移除寫死的決策規則 | §3 的測試全部通過 | **目前的架構** |

---

## 2. 最終架構（A2；A0 為不改動的參照）

```
Ditto-8B Speaker（凍結）
  ▲ Speaker block：Planner 的 act/state/length，加上判斷器的 unmet 清單（取代關鍵字 agenda）
Qwen2.5-32B JSON Planner（凍結；system prompt v3；read_plan v3）
  ▲ GOAL STATUS：判斷器每回合的輸出
Goal 滿足判斷器：Qwen3-4B-Instruct-2507 + LoRA（cross-fitted）
  ▲ 輸入：scenario_text（和 Speaker 相同）＋ 截至上一回合 agent 回覆的完整對話
    （不含輪數，也不含任何 ledger 或 agenda 的規則輸出）
```

**A2 中每一個決策由誰決定（全部是動態推理，沒有寫死的門檻）：**

| 決策 | 由誰決定 | 先前的寫死規則（已移除） |
|---|---|---|
| 是否結束對話 | Planner 的 `end_session`，先寫出離開與繼續兩方理由再決定 | 寫出非 weak 規則名稱 → 程式強制 Complete（D1）；strong/weak 權重；「先達成者優先」組合規則 |
| 目標滿足到什麼程度 | 學習式判斷器 | 關鍵字 agenda（D3）；Task 1 專用的 ledger 計數（D2） |
| 這一輪寫多長 | Planner（依 persona 風格與此人自己先前的訊息） | persona→固定字數區間（例如 Concise 對應 8–22 字），並把 Planner 的數字夾回區間 |
| 耐心還剩多少 | Planner（JSON 新欄位 `patience`） | 原版 JSON 沒有此欄位，所以永遠顯示「patience full」 |
| 下一句的 act | Planner 的機率分佈抽樣（verbalized sampling，原本就是動態的） | — |

**結束的執行：** `end_session=true` 時，在 Speaker 發話之前**靜默離開**：不發話、不算一輪、不呼叫 R0。與 benchmark「空訊息＝終止」的口徑相同（D6）。

**截斷（D5）：** Planner 上限 32104、Speaker 32504、判斷器 16384 tokens。只有超過時才刪最舊的內容，並一律保留靜態前綴、最新對話和生成提示。原本就沒超長的 prompt，token 與舊做法逐位元相同。每次呼叫都記錄長度與是否壓縮。

**評估與防洩漏（D7、D8）：**
- **cross-fitting：** 每個外層折的 inner scenario（15/16/18 個）分成 3 組；判斷器 (f, g) 只用其他組訓練，只在 g 組評估。
- **leak gate：** runner 讀取判斷器的 `train_manifest.json`，若與評估組重疊就拒絕執行。
- **outer test：** 不碰。
- **主要時機指標：** 模擬輪數分佈與真人 K 分佈之間的 W1，以及平均差。

---

## 3. 每個修正與它的證據

| 項目 | 修正內容 | 證據（測試） |
|---|---|---|
| D1 停止覆寫 | A2 設 `SEPSIM_ACT_PRIOR=nostopclobber`；停止改由 `end_session` 決定；runner 斷言兩個 arm 的設定 | `test_planner_prompt_v3`：寫出 satiation 但 `end_session=false` 時 act 不變、不會停；`true` 才停；無效值記錄為無效且不修補 |
| D2 ledger 錯誤事實 | Planner prompt 只保留 `turns so far` 和 Planner 自己的 gain | 30 個 prompt 與原版逐字比對：只有文件列出的位置不同 |
| D3 關鍵字 agenda | 以判斷器的 GOAL STATUS 取代（Planner prompt 與 Speaker block 都改） | 同上；Speaker block 只有 pending 那一行不同 |
| D4 Planner 看不到完整 goal | 判斷器用與 Speaker 相同的 scenario 文字 | 程式與測試 |
| D5 截斷 | 見 §2 | 伺服器：678/680 個真實 prompt 與舊做法相同，其餘 2 個現在完整；合成超長 prompt 保留前綴、最新對話、生成提示；Ditto 同樣通過 |
| D6 settle 那一輪問新問題 | 靜默離開 | `test_task2_episode`：10 種情況；A1 離線推導 = 線上結果（5 種模式） |
| D7 在訓練 scenario 上評估 | cross-fitting + leak gate | `make_crossfit_groups`：斷言 inner ⊆ 外層 train、與外層 test/dropped 不相交、每個都有已記錄的軌跡 |
| D8 逐集比 K 是代理指標 | W1 + 平均差為主要指標 | `test_analyzers`：人工可算的例子 |
| D9 complete 有上限 | 報告時揭露，只做配對比較 | — |
| D10「USER 是真人打的」 | A2 的 system prompt 改為「this person's messages so far」 | system prompt 逐字比對測試 |
| 新：寫死的長度區間與夾回 | 移除區間，Planner 自己決定長度 | `read_plan_v3` 測試：Concise persona 寫 150 字也保留 |
| 新：patience 永遠是 full | JSON 加入 `patience` 欄位 | 測試：`patience thin` 會出現在渲染出的 state 中 |
| 新：判斷器解析失敗時顯示「(nothing identified)」 | 改為「(not available this turn…)」 | Speaker block 與 GOAL STATUS 測試 |
| 新：Llama 標籤 judge 會有兩個 BOS | 一律 `add_special_tokens=False` | 伺服器：Llama 恰好一個 BOS；假 tokenizer 會斷言不可再加 BOS |
| 新：判斷器 LoRA 以 bf16 更新 | 可訓練參數轉 fp32 | 程式 |
| 新：run_meta 寫兩次 | 只寫一次，含 system prompt 與判斷器 prompt 的 SHA | 程式 |

**所有測試（伺服器，正式 sepsim 原始碼，consistent-test 環境）全部通過：**
`test_task2_episode`、`test_planner_prompt_v3`、`test_goal_judge`、`test_derive_gate_arms`、`test_analyzers`、`test_fit_prompts`、`test_judge_tokenization`。

---

## 4. 仍保留、未改動的部分（完整揭露）

這些不是停止或目標判斷的決策規則，但屬於固定設定。我選擇保留，理由如下；你若認為也要改，請指出。

| 項目 | 內容 | 保留的理由 |
|---|---|---|
| v2fix 生成過濾 | 每輪 1 個 greedy + 3 個抽樣候選；最多重抽 4 次；拒絕抄模板或逐字重複的候選；選長度最接近 Planner 目標的候選 | 品質過濾，選擇的目標來自 Planner（動態）；更改會同時改變 Speaker 的行為 |
| persona 文字對應 | 熟悉度→「can be: …」描述、goal stage→起始階段 | schema 欄位轉成文字，不是數值門檻 |
| gain 摘要行 | 「本輪 gain 對上本次平均」，前 2 輪不顯示 | 顯示 Planner 自己估計的數字，不做判斷 |
| act 詞彙與 benchmark 對應 | acts.py 的類別清單 | 標籤的詞彙表 |
| 停止理由的名稱與描述 | 6 個名稱，只保留描述句的第一句 | 只作為描述理由的詞彙；已移除權重與效能評論 |
| T_max | 10 | benchmark 協定 |
| 判斷器輸出格式 | 三種狀態 + 最多 3 項 unmet | 格式定義 |
| 靜默離開的取捨 | 模擬使用者不會說「謝謝、再見」 | TREC 真人約 5/34 會說；這是協定口徑的選擇 |
| 原作者在 TREC 上開發 | persona 與 prompt 的原始設計沒有 hold-out | 已揭露；A2 已移除其中的數值規則（長度區間），其餘文字仍在 |
| Ditto 的已知問題 | 38% 首句跨 seed 相同；benchmark 可辨識度 AUC 0.769 | 本架構不處理 |

---

## 5. 需要你確認的事（架構層面）

1. §2 的決策分工與靜默離開。
2. §4 保留項目中，有沒有你認為也屬於「寫死的規則」、需要改的。
3. **標籤 judge**：主標籤用 Llama-3.1-70B（本地），交叉檢查用 gpt-oss-120b；κ < 0.4 就停。你是否要人工檢查 30 筆？
4. **判斷器訓練資料**：Ditto rep0 + rep1 的逐輪樣本（約 1300 筆，來自舊 runner，第 7–10 輪的 Speaker 曾受截斷影響）加上 TREC 真人對話（約 150 筆）。另一個選項是先用修正後的 A0 重新產生軌跡，比較乾淨，但要多 1 輪 rollout 的時間。
5. **分佈落差：** 判斷器用 no-gate 軌跡訓練，但部署在 A2 的軌跡上（A2 的 Planner prompt 不同）。我建議接受並監測（記錄判斷器在 A2 上的狀態分佈），你同意嗎？
6. **Planner 規模**（先前討論的 8B/9B 候選）：要在架構確認之後、正式實驗之前做「逐步 prompt 比較」嗎？

確認後，下一份文件是**整體 Plan**：標籤 → 判斷器 → A2 評估 → RL 訓練，以及可以穩定長跑的流程設計（排程、續跑、監控、失敗處理、版本凍結）。
