# PLAN（Project Operating Mode：可斷點續跑的主計畫）

最後更新：2026-09-25 19:05（Asia/Taipei）。

## Goal（使用者定案）
在 E1.6 已經解決 E1 問題的基礎上，讓 **Task 2 指標（尤其回合數）接近真人**，同時 **Task 1 指標不退化**。方法以 **GRPO RL** 為主，**零 data leakage**。

## 架構：`pend` arm（Planner + Ditto + Selector，No Annotation，不用 goal judge）
- Planner = **Qwen3-4B-Instruct-2507** + LoRA（使用者要求縮小；伺服器上現成、原生 262K context、非思考版）。
- E1.6 樹 `trees/e1r_cf19400` + 修正：override 關、只呈現 Task 2 為真的事實、完整 goal（topic+context）、沒有 band/clamp（保留每個 act 各自的長度）、prompt 不截斷、patience 由 Planner 判斷。
- Planner 輸出 `goal_met` / `still_wanted`，`end_session=true` = 這一句是最後一句：act 取自 Planner 自己的 Complete 項目，Ditto 寫收尾（不必道謝），送出後結束（E1/E1.6 的 PLANNER_END 語意）；第 1 回合不能結束。
- Ditto 的 pending = Planner 的 `still_wanted`。

## 環境（要揭露）
- OpenAI 額度用完 → R0（Task Agent）與 requirement-ledger 判斷器改用 **gpt-oss-120b**（hcwang 在 245 GPU0 的 vLLM，port 8019）。benchmark client 在這個 dialect 不送 reasoning_effort（伺服器預設）。
- Ledger 判斷器 `max_tokens=200` 在推理型模型下會回空答案（coverage 永遠 0）→ 非 gpt-5 dialect 下限提高到 4000，空回應會計數，verify 在 ledger 從未記到任何需求時判為失敗。

## GRPO（`train_planner_rl.py --arm pend`）
- Task 2 rollout：每次更新 4 個 train 情境 × G=4；reward v3 = coverage − |模擬回合 − 真人回合| / 10 − 格式違規（unparsed、hit_max_new，λ=1）。
- **Stop credit**：回合差的 advantage 只作用在每一步 `end_session` 的值 token；coverage/格式的 advantage 作用在整段。
- **Task 1 停止組**：每次更新抽 4 段 train_all 的真實對話，在真人最後一回合和一個較早回合各取 G 個 Planner 決策，reward = 和真人是否結束一致（只作用在 `end_session` token）。
- **停止 token 輔助監督**（權重 1）：未訓練的 Planner 在真實對話上從不結束，GRPO 組內標準差為 0、學不到 → 加上人類停止標籤的監督項（報告要寫成 GRPO + 輔助監督，不稱純 RL）。
- KL 0.04（對未訓練的原始模型），lr 1e-5，T=0.7 / top_p=0.8（Qwen3 官方值）。
- 選 checkpoint：validation 的 Task 2 reward + Task 1 term_f1（權重 1，事先定好）。
- 跨 episode 的批次生成（Planner、Ditto）：驗證過 padded 與 unpadded 的 log-prob 平均差 0.007，差距 >0.5 的 token 佔 0%。

## Splits（`splits_v1.json`，官方 goal+persona 三折）
fold2：train 14（Task 2）/ train_all 17（Task 1）、validation 4、test 5（test_all 9）。語料只有 56 段已完成的 session；PRISM/MultiWOZ 是 benchmark 的 OOD 測試集，不拿來訓練。

## 評估
- Task 1：`task1_v4.py`（teacher-forced，56 段 + K+1），用 E1.6 當時的同一條指令評分（`score_method.py --no-llm` + `reduce_k1.py`），和 E1.6 並列比較。
- Task 2：validation / test 回合數 vs 真人（平均差、W1）、coverage、complete，對照 E1 的 32B 數據（保留，不重跑；R0 環境不同要註明）。

## 時程
- 9/25 18:58 GRPO fold2 開始（30 次更新，每 5 次 validation）；每次更新時間待實測後校正。
- 9/25 約 21:00 Task 1 未訓練基準出爐。
- 之後：選 checkpoint → Task 1 完整評估（訓練後）→ test 一次 → 視時間補 fold0/fold1。
