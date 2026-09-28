# 架構稽核與修正後計畫 v2（Ditto + Planner + Goal 滿足判斷器）

日期：2026-09-25。**狀態：待你確認，尚未執行任何新工作。** 本文件取代 `PLAN_DITTO_GOALJUDGE_20260924.md`。
寫法原則：每個數字都標出處；凡是參考過 TREC 語料才做的設計決定，都在 §5 逐項揭露；我自己先前說法不精確或錯誤的地方，列在 §2.3。

---

## 1. 完整過程（目前為止做過什麼、結論現在還算不算數）

| # | 步驟 | 關鍵數字 | 出處（245 `grpo_planner/` 除非註明） | 現在的狀態 |
|---|---|---|---|---|
| 1 | 核對 Stage A（PRISM stop SFT，交接前已跑完） | 504 updates；epoch2 val NLL 0.384 | `prism_sft_7b_v1/metrics.jsonl` | 有效，但見 #6：學到的是位置 |
| 2 | 三折 prompt parity 稽核 | 346 rows 全過 | `audits/stop_prompt_parity_audit.json` | 有效 |
| 3 | Stage B（三折 TREC LoRA） | 全選 epoch2；合併 AUC 0.681；0.5 閾值 K+1 0/12 | `trec_inner_fold*_7b_v1/`、`analysis_v1/` | 有效，但見 #6 |
| 4 | Task 2 runner 重寫（終止時序、scenario 層級統計、離線截斷） | 5 種結束情況單元測試通過；線上/離線 max \|Δp\| 3.7e-7 | `code_snapshots/stageC_v1/`、`stageC_v1/smoke_crn/` | 有效；但**繼承了 sepsim 的截斷 bug**（§2.1 D5） |
| 5 | UserLM no-gate rep0（68 集）+ gate 的離線 arm | 8.71 輪、coverage 0.192；43% 離題句 | `stageC_v1/rep0/` | 探索性（截斷 bug）；UserLM 路線已停 |
| 6 | 位置捷徑稽核 + 遮蔽消融 + PRISM 內容 probe | PRISM 同一輪內 AUC 0.44–0.47（CI 含 0.5）；只看輪數 0.849 > gate 0.814 | `mask_ablation_v1/`；244 `prism_probe_v1/` | **有效，是主要研究發現**：真人停止時機無法從內容預測 |
| 7 | Stage A′ Cox SFT / 不加位移對照組 | Cox ep1 同一輪內 AUC 0.449；對照組 ep1 0.467 | 221 `stageA_cox_v1/`；244 `stageA_nooffset_244_v1/` | ep2 跑完即可結案；不再往 gate 方向投入 |
| 8 | Ditto no-gate rep0（68 集）；rep1 執行中 | 10.0 輪、coverage 0.705、complete 0.191 | `stageC_v1/rep0_ditto_shard*/` | 探索性：截斷 bug（第 7–10 輪），且與 UserLM 不同環境 |
| 9 | Ditto 上的 Planner 結束稽核 | 66/68 有結束 Act；照它停時誤差 5.47→2.07，coverage −0.027 | `stageC_v1/planner_stop_audit/` | **機制描述要更正**（§2.3 M1、M2） |

---

## 2. 稽核發現

### 2.1 缺陷（D = defect；每項都有原始碼行號或資料證據）

| ID | 缺陷 | 證據 | 影響 | 修正 |
|---|---|---|---|---|
| **D1** | **結束是規則覆寫出來的。** Planner 在 `stop_rule` 寫出非 weak 的規則名稱時，`read_plan` 會把它抽樣出的 act 覆寫成 Complete/settle 或 Complete/abandon。 | `planner_prompt.py` 第 299–326 行；runner 設 `SEPSIM_ACT_PRIOR=off`（覆寫開啟）。Ditto rep0 第一次結束的 66 次中，64 次屬此機制。 | 我先前說「Planner 下了結束 Act」，實際是「Planner 寫出一個停止規則名稱 → 程式強制結束」。規則名稱由 LLM 自己讀對話決定（見 D2），**不是固定門檻**，但這個轉換必須明講。 | 保留這個機制（否則 Planner 幾乎不會停：原作者量過，抽樣本身在 CLOSE 上只放 0.054 的機率），但在文件和報告中定義清楚：**停止決策 = Planner 寫出非 weak 的 stop_rule**。 |
| **D2** | **StoppingLedger 在 Task 2 提供錯誤事實。** 有用/無用計數、最佳品質、satiation、disgust 都需要 Task 1 的真人標註；Task 2 沒有標註，這些值永遠停在初始值。 | Ditto rep0 的 680 步 Planner prompt 全部是「useful replies: 0, unhelpful: 0」「best offered so far: nothing yet」；唯一出現的停止條件是關鍵字規則 difference_threshold（129 步）。 | Planner 一直被告知「0 次有用回覆」。它的 satiation 判斷是靠自己讀對話，而且是在錯誤資訊干擾下做出的。 | 在 Planner prompt 中移除這些 Task 1 專用的行，換成判斷器的 GOAL STATUS（§3）。 |
| **D3** | **agenda「pending」是關鍵字規則。** 從 goal 文字抽出 8 個詞，agent 回覆中出現該詞就算已滿足，而且會觸發「新 offer 重設」（>0.5 詞彙新穎度）。 | `agenda.py` 第 59–113、177–200 行；它同時出現在 Planner prompt（「WHAT THEY STILL WANT」）與 Speaker block（「- pending:」）。 | 正是你所說的寫死規則，並且直接影響 Speaker 要問什麼。 | Planner prompt 和 Speaker block 都換成判斷器輸出的「尚未滿足」清單（兩邊一致）。 |
| **D4** | **Planner 看不到完整的 goal。** Planner 只看 `goal.topic`；Speaker 看 persona + `goal.context` + `goal.topic` + 已知資料集。 | `planner_prompt.py` 第 221 行；`pipeline.py` 第 21–39 行。 | Planner 判斷「目標是否達成」的依據比 Speaker 少。 | 判斷器的輸入用與 Speaker 相同的 `scenario_text`。Planner 的 goal 欄位**不改**，以減少同時變動的因素；判斷器的輸出會帶進這部分資訊。 |
| **D5** | **從尾端截斷。** Planner 12000、Speaker 7000 tokens，`truncation=True` 從右側截，砍掉最新對話與生成提示。 | `models.py` 第 248、365、405 行。Ditto rep0：Planner prompt 2/680 步超過 12000；Speaker prompt 超過 7000 介於 57（8.4%）與 88（13%）步之間，全部在第 7–10 輪。 | 舊的 A0 基準在第 7–10 輪受汙染。 | 上限放寬到模型 context（Qwen2.5-32B 為 32,768），超過時才壓縮最舊的歷史、保留生成提示。回歸測試：原本沒超長的 prompt，新舊 token 逐位元相同。**A0 重跑。** |
| **D6** | **Ditto 在 settle 那一輪常問新問題。** | Ditto rep0 第一次結束的 66 次中，25 次含問號（18 次沒有任何收尾字眼）。 | 若照協定讓「結束語」算一輪，約 4 成的對話會停在一個沒人回答的問題上，不像真實使用者。 | 結束改為**靜默離開**：Planner 決定停止的那一步不再生成發話、不算一輪，與 gate 停止的語意相同。這也是 benchmark 對「沒有原生結束 token 的方法」的公平口徑（t2_rollout.md：空訊息＝終止）。 |
| **D7** | **用判斷器自己的訓練 scenario 評估它。** 舊計畫 §5.1 在全部 34 個 union scenario 上跑 A2，但每一折的判斷器都用該折 inner train 訓練過。 | 我舊計畫本身的設計錯誤。 | 結果會偏樂觀（在訓練資料上評估）。 | **cross-fitting**：每個外層折內，把該折的 inner scenario 分成 3 組，每組都用「沒看過該組」的判斷器評估（§4）。 |
| **D8** | **逐集比對真人 K 是代理指標。** 真人對話的 agent 是平台上的真 agent，模擬對話的 agent 是 R0，兩者內容不同，該停的時間也不必相同。 | 設計層面的問題。 | 逐集絕對誤差會把「對話內容不同」也算成誤差。 | 主要指標改成**分佈層級**：模擬輪數與同一組 scenario 真人 K 的平均差、分佈距離（W1）。逐集誤差只當次要指標。 |
| **D9** | **complete 有上限。** requirement shards 是從真人對話推回來的，其中可能有 goal 文字沒寫到的需求。 | 設計層面的問題。 | complete 偏低不一定是停止造成的。 | 報告時說明；coverage/complete 都只做同一組 scenario 的配對比較。 |
| **D10** | **Planner system prompt 寫著「每則 USER 訊息都是真人打的」**，但在 Task 2 裡是模擬的。 | `planner_prompt.py` 第 85–86 行。 | 可能影響 Planner 的語氣判斷，影響方向未知。 | **不改**（A0 與 A2 都一樣），在報告中揭露。 |

### 2.2 設計上沒有問題、但需要你知道的
- Ditto 的首句跨 seed 有 38% 完全相同（benchmark 也記錄過）；本計畫不處理。
- Ditto 只能在 `consistent-test` 環境跑（transformers 5.7.0）；所有 Ditto arm 都在這個環境，不和 UserLM 做跨環境比較。
- Planner JSON 解析失敗：2/680。

### 2.3 我先前說法不精確或錯誤的地方
- **M1：** 我說「Planner 已經會在接近真人的時間點下結束 Act，結束理由多是以內容為準的判斷」。更正：結束的機制是 D1 的覆寫；規則名稱（satiation 等）確實由 LLM 讀對話決定，因為 ledger 在 Task 2 從來不會提供 satiation 候選；但它是在 D2 的錯誤事實干擾下判斷的。
- **M2：** 「照 Planner 停」的誤差 2.07，是用「結束語算一輪」計算的，並且基於有截斷 bug 的 A0。改成靜默離開後，數字會不同，需要重算。
- **M3：** 我先前提出的 D2 長度擬合項會把 TREC 的平均長度直接寫進訓練目標。你指出之後已經取消，沒有執行。
- **M4：** Ditto 對 UserLM 的比較混入了環境差異（只有 14/68 集的第 1 輪 Planner 輸出相同）。我當時已說明，這個結論不採用。

---

## 3. 修正後的架構

```
Ditto-8B Speaker（凍結；block 中的 pending 換成判斷器的 unmet 清單）
   ▲ Planner block
Qwen2.5-32B Planner（凍結；截斷修正；prompt 改動見下）
   ▲ GOAL STATUS（每回合，由判斷器產生）
Goal 滿足判斷器：Qwen3-4B-Instruct-2507 + LoRA
   ▲ 輸入：scenario_text（和 Speaker 相同）＋ 到本回合 agent 回覆為止的完整對話
     （不含輪數、不含任何 ledger 或 agenda 的規則輸出）
```

**Planner prompt 的確切改動（只改這些）：**
1. 刪除「WHAT HAS ACTUALLY HAPPENED」區塊中的：useful/unhelpful 計數、best offered、repeated offer、condition met first、stopping conditions available / no stopping condition（D2）。
2. **保留** `turns so far` 和 Planner 自己的 gain 行。前者是事實（使用者知道自己聊了多久），不是規則；後者是 Planner 自己的估計。Planner 沒有用停止標籤訓練，所以不會學到資料集專屬的位置先驗。這一點請你決定（§6-3）。
3. 「WHAT THEY STILL WANT – pending」換成新區塊：「GOAL STATUS (assessed from the conversation by a separate model) – status: SATISFIED/PARTIAL/NOT – still unmet: …」（D3）。
4. persona 那一行 `gives up: X (so about N unhelpful replies…)` 刪掉括號內的規則說明（因為 D2 的計數已移除）。
5. system prompt、停止規則清單、JSON 格式都不變。

**Speaker block：** 「- pending:」換成判斷器的 unmet 清單，其他不變。

**結束執行：** `PLANNER_END=1` 加上靜默離開（D6）。在第 t 步，若 Planner 的 stop_rule 是非 weak 規則，就在 Speaker 發話之前結束：發話輪數 = t−1，coverage/complete 取第 t−1 步後的值。

**截斷：** Planner 與 Speaker 都用 D5 的修法，每次呼叫記錄長度與是否壓縮。

---

## 4. 實驗設計（只有一個新 arm，另加一個必要的參照）

| Arm | 內容 | 為什麼需要 |
|---|---|---|
| **A2（新）** | Ditto + Planner（prompt v3）+ 判斷器 + 靜默離開 + 截斷修正 | 你要看的主角 |
| A0-fix | Ditto no-gate + 截斷修正（其他全照舊） | coverage/complete 需要一個參照，才能判斷停止是否犧牲了任務品質；舊的 A0 有截斷 bug，不能用 |
| A1（免費） | 從 A0-fix 以精確截斷推導「規則式 prompt + 靜默離開」 | 不需要 rollout；可以比較「換成判斷器」本身的效果（A2 對 A1） |

**折與 cross-fitting（D7）：**
- 對每個外層折 f，取它的 inner scenario 中有 requirement shards 的：fold0 15、fold1 16、fold2 18 個。
- 把它們按 scenario 分成 3 組。每組用「用其他 2 組訓練的判斷器」跑 A2。
- 共 9 個判斷器（3 折 × 3 組）。所有 inner scenario 都拿到 out-of-fold 評估。
- outer test 完全不碰。

**Rollout 數量：**
- A2：49 個（scenario, 折）組合 × 2 seeds × 2 replicates = 196 集。
- A0-fix：34 個 scenario × 2 seeds × 2 replicates = 136 集（不依賴折）。
- 兩個 arm 在同一個佇列中按 scenario 交錯執行，避免 R0 隨時間漂移和 arm 對齊。

**主要指標（事先固定）：**
1. 模擬輪數的平均值與同一組 scenario 真人 K 平均值之差；兩者分佈的 W1 距離。
2. coverage、complete（gpt-5-mini ledger，只用於評估）。與 A0-fix 配對，scenario 層級，cluster bootstrap。
3. **品質守門（沿用）：** coverage 差 ≥ −0.02，而且 complete 最多少 1 集。

**次要指標：** 逐集 \|輪數−K\|；結束規則的分佈；過早結束率（只能在 A1 上精確量）；判斷器與標籤 judge 的一致率。

**成功的定義（事先固定）：**
- A2 在指標 1 上比 A0-fix 更接近真人，並且通過品質守門。
- 另外報告 A2 對 A1 的差異，以判斷「判斷器取代規則」本身是否有幫助。

**泛化檢驗（核心，只評估不訓練）：**
- 判斷器的 status/score 能否在同一輪內區分「真人最後一輪」與「之前各輪」。
- 在 TREC gold sessions（同樣 out-of-fold）以及 MultiWOZ test 上分別計算，並與只看輪數的 baseline 比較。

**outer test（方法凍結後只跑一次）：**
- 每個外層折用全部 inner scenario 重訓判斷器。
- 在 outer test 中有 shards 的 scenario 上跑 A2 與 A0-fix：fold0 7、fold1 2、fold2 4 個。
- **樣本非常少，只能報告描述性結果，不能當成強結論。** fold0 的 outer test 先前看過，標為探索性。

---

## 5. 所有受 TREC 語料影響的設定（完整揭露）

| 設定 | 值 | 來源 | 是否擬合 TREC |
|---|---|---|---|
| T_max | 10 | benchmark 協定 | 否 |
| 品質守門門檻 | coverage −0.02、complete 1 集 | 我在看到任何 Task 2 結果前寫下（PREREG） | 否 |
| 靜默離開 | Planner 停止＝不再發話 | benchmark「空訊息＝終止」口徑；**TREC 真人 29/34 最後一句不是收尾語，與此一致** | 參考過 TREC 的觀察，決定理由是協定口徑 |
| persona 規則：長度區間（8–22 字等）、耐心、dispute prior | 見 `persona.py`、`planner_prompt.py` | 原作者宣告的先驗 | **原作者在同一份 TREC 語料上開發，且沒有 hold-out**（原始碼註解明寫參考過整個語料 190 個 turn 的量測）。本計畫不改動，但所有 TREC 評估都要揭露這一點 |
| 停止規則清單與 weak/strong 權重 | 見 `stopping.py` | 文獻（Kraft & Lee、Maxwell 等） | 否（原作者宣告） |
| 判斷器標籤 prompt | 我撰寫 | 在看任何結果前固定，只寫一個版本 | 否 |
| 判斷器超參數 | LoRA r16、lr 1e-4、3 epochs | 事先固定，不做選模 | 否 |
| 評估用的 requirement shards | benchmark 提供 | 從真人對話推回，只用於評估 | 評估工具，不進訓練 |

---

## 6. 需要你決定的事

1. **靜默離開（D6）：** 同意 Planner 停止時不再生成發話、不算一輪嗎？另一個選項是維持「結束語算一輪」，但約 4 成會停在問句上。
2. **標籤 judge：** 主標籤用 Llama-3.1-70B（244 的兩張 V100 或 245 GPU1），200 筆用 gpt-oss-120b 交叉檢查（需要 245 的 GPU），κ < 0.4 就停下來。另外，你是否願意人工檢查 30 筆？
3. **`turns so far`：** 要保留在 Planner prompt 嗎？我建議保留（它是事實，不是規則）。
4. **判斷器訓練資料：** 只用 Ditto 軌跡加 TREC 真人對話，還是也加入已跑完的 UserLM 軌跡？
5. **GPU：** 245 的 GPU0 目前讓給你的組員。只用 GPU1 的話，rollout 時間約加倍（約 20 小時），要等 GPU0 釋出嗎？
6. **MultiWOZ 泛化檢驗：** 要做嗎？需要把本機的 MultiWOZ test 傳到伺服器，而且只做評估。

---

## 7. 執行順序（每一步都有檢查點，沒過就停下來回報）

1. **程式修改：**
   - 截斷修正、Planner prompt v3、Speaker block 改動、靜默離開、判斷器接入。
   - 單元測試：截斷回歸（token 逐位元相同）、prompt 內容檢查（該刪的都刪了、該留的都在）、結束語意 5 種情況。
2. **標籤：**
   - 把 Ditto rep0/rep1 與 TREC 真人對話切成逐輪樣本，只含 inner scenario。
   - Llama-70B 產生標籤；gpt-oss-120b 交叉檢查；**檢查點：κ ≥ 0.4。**
3. **訓練 9 個判斷器（cross-fit）：** 報告 out-of-fold 的標籤一致率。
4. **Smoke：** A2 與 A0-fix 各跑 2 集，人工檢查 prompt 與 block 內容。**檢查點：沒有空白或截斷的 prompt，GOAL STATUS 確實出現在 prompt 中。**
5. **完整 rollout：** A2 196 集 + A0-fix 136 集，交錯執行。
6. **分析：** §4 的主要與次要指標、泛化檢驗；報告寫明每個數字的出處。
7. **（你確認之後）** outer test。
