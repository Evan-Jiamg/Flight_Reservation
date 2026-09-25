# PLAN（Project Operating Mode：可斷點續跑的主計畫）

最後更新：2026-09-25 15:10（Asia/Taipei）。每輪重大變更後更新本檔與 `TASKS.md`。

## Goal（使用者 2026-09-25 定案）
以 **E1.6**（最新版 E1，已解決 E1 在 Task 1 的問題）為基礎，修掉 E1.6 既有的架構缺陷，並加上 **RL 訓練**，讓 **Task 2 指標（含回合數）重新接近真人**。Speaker 用 **Ditto-8B**（移植 E1.6 的 Planner 端）。**零 data leakage**。

- E1.6 = `/home/mzjiang/Sep-1st-Simulator-e1r` @ `cf19400`（沒有未提交修改），設定見 `endfix/stage_e16.sh`。唯讀副本放在 `grpo_planner/trees/e1r_cf19400`（附 SHA256SUMS）。
- E1.6 當時用的是 UserLM-8b，而且 **從來沒有 Task 2 rollout**；目前 Task 2 的數字（8.44 回合 vs 真人 4.45）是 v2fix 的。
- E1.6 的 config z123 是看過評估語料分數後才選的：**報告時必須揭露**。

## Only two systems are compared（同一個 Task2Env，同一份 split）
| 系統 | arm | 內容 |
|---|---|---|
| **Baseline：E1.6 in Task 2** | `e16` | E1.6 Planner 端照原樣：NO_ANN（載入時就剝除標註）、ACT_FULL、SELF_JUDGE + STOP_LEDGER + override、PLANNER_END（Complete act 說完這句就結束）、T1_SAMPLE；Ditto；截斷修正 |
| **Final** | `final` | E1.6 + 缺陷修正（D1 override 關、D2 只留 Task 2 為真的事實、D3/D4 改由目標判斷器提供 GOAL STATUS、D5 不截斷、D10、patience 由 Planner 自己決定、長度沒有 band 也不 clamp，但保留 ACT_FULL「每個 act 各自的長度」）+ `end_session` 靜默離開 + 判斷器 unmet 傳給 Ditto + **GRPO RL**（Planner LoRA，只用 train） |

另外報告「RL 前」的中間點（`final` arm、尚未 RL），用來歸因。

**E1.6 裡沒有移植、需要揭露的部分：**
- 只適用 UserLM 的開關（ROLESTOP 以外）：INTENT_PROSE、NEXTSTEP_INLINE、INTENT_CACHE、ENDGATE。Ditto 沒有 end token，這些關閉；`ditto_e16.DittoSpeaker.load` 在它們被打開時會直接報錯。
- ROLESTOP 的 Ditto 對應版本：`<|im_start|>` 也當停止 token。
- T1_SAMPLE 的溫度改用 Ditto checkpoint 自己的 `generation_config`（T=0.7、top_p=0.8），E1.6 用的是 UserLM 卡片上的 1.0／0.8。
- SELF_JUDGE、STOP_LEDGER 只留在 Baseline；Final 以判斷器取代它們。

## Metrics
- **主要（Task 2，對真人）：** 模擬回合數與真人回合數的平均差與分佈 W1；coverage；complete。
- **守門：** Final 對 Baseline，coverage 差 ≥ −0.02，complete 最多少 1 集。
- **Task 1 不退化：** teacher-forced 的 Act TVD、transition JSD、intent adherence，對照 E1.6 + Ditto。
- **泛化：** MultiWOZ（只評估判斷器）。

## Splits（`grpo_planner/splits_v1.json`）
| 分割 | fold0 / 1 / 2 | 用途 |
|---|---|---|
| train | 14 / 12 / 14 | 判斷器 SFT、RL rollout、所有統計 |
| validation | 1 / 4 / 4 | 只用於選 checkpoint |
| test | 9 / 5 / 5 | 方法凍結後只跑一次（`--final`） |

## Steps
| 步驟 | 內容 |
|---|---|
| 0 | E1.6 移植（`ditto_e16.py`、`task2_env` 的 e16/final、v3 的 ACT_FULL、verify）→ 245 smoke（進行中） |
| 1 | 標籤（Llama-70B，進行中）→ κ 交叉檢查 → 判斷器 SFT × 3 折 |
| 2 | 小 Planner 測試（進行中）：Qwen2.5-7B、Llama-3.1-8B，並用 **同一 prompt 的 32B NF4** 當參考。v3 teacher-forced 200 步，看 Planner 是否及時結束；過關者再做 Task 2 rollout（`final` arm）與 32B 比較 |
| 3 | Baseline `e16` rollout（validation；test 等 Final 凍結後一起跑）＋ Pipeline 驗證 + GRPO 可行性（每折 5 次更新） |
| 4 | GRPO 正式訓練（三折，reward v2） |
| 5 | Validation 選 checkpoint → Test 只跑一次（Baseline vs Final）+ Task 1 不退化 + MultiWOZ |

## GPU
245 GPU1（H100 97GB）是唯一能跑 Ditto／bf16 的卡；GPU0 是 hcwang 的 vLLM，不動。244 是 V100（沒有 bf16，也沒有 Ditto），用來跑判斷器 SFT 與評估的非 Ditto 部分。

## ETA
約 **9/28**。RL 是最長的一段，約 1–1.5 天；245 GPU1 同時只能跑 1–2 個 rollout worker。
