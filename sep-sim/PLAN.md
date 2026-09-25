# PLAN（Project Operating Mode：可斷點續跑的主計畫）

最後更新：2026-09-25 14:40（Asia/Taipei）。每輪重大變更後更新本檔與 `TASKS.md`。

## Goal
在 TREC 2026 UserSim Task 2 上，用 **RL 訓練**讓 Ditto + Planner 模擬使用者，在與真人相近的時機停止，同時不犧牲需求覆蓋與語言品質。
- 嚴格的 Train / Validation / Test 切分，**零 data leakage**。
- 跨資料集泛化：MultiWOZ（只評估）。

## Operating rules
- **Autonomy：** 只寫入本 worktree 的 `sep-sim/`、`ops/` 與伺服器 `/tmp2/mzjiang_usersim/`；不改 sepsim 原始碼，不寫入 home 目錄。
- **Parallelism：** 同時最多 4 個並行任務（sub-agent 或 GPU job），其餘排隊。
- **Auto-accept plan：** 依本計畫自動執行；只有閘門（Gate）沒過才停下來等使用者。
- **Validation first：** 每個子任務完成後都要跑對應的測試或驗證；沒過就自行修正後再往下。
- **SSH：** 只有主協調者會連伺服器，一次一條連線、間隔 ≥ 15 秒；sub-agent 不連伺服器。
- **GPU：** 先用 cfda5（245）GPU1，再用 cfda4（244）；245 GPU0 只在確認空出後使用。

## Data splits (leakage control) — `grpo_planner/splits_v1.json`, sha a3591933…
每折 f 各自獨立：

| 分割 | 有 shards 的 scenario 數（fold0/1/2） | 用途 |
|---|---|---|
| train | 14 / 12 / 14 | SFT、RL rollout、reward 與正規化統計、控制器看到的彙總 |
| validation | 1 / 4 / 4 | 只用於選 checkpoint 與閘門 |
| test | 9 / 5 / 5 | 方法凍結後只跑一次 |

- 所有統計都只從該折的 train 計算；validation 與 test 列在 `forbidden_for_training`，任何訓練程式都必須斷言沒有用到。
- MultiWOZ 與 PRISM 保留的 60 位 user：只評估。
- 標籤 judge（Llama-3.1-70B）看不到 requirement shards；shards 只用於評估。

## Architecture (frozen v3, tested)
- **Speaker：** Ditto-8B（凍結）。
- **Planner：** policy，LoRA；v3 prompt，停止 = `end_session`，長度與 patience 由 Planner 決定。
- **Goal 滿足判斷器：** Qwen3-4B + LoRA（每折 1 個，只用 train 訓練）。
- **結束：** 靜默離開。
- **截斷：** Planner、Speaker、判斷器都先擬合再生成，程式會斷言不超過上限，不會從尾端截斷。
- **共用環境：** `task2_env.py` 的 `Task2Env` 同時服務評估與 RL，確保兩者行為一致。

## Stages and gates
| # | 階段 | 閘門（沒過就停） |
|---|---|---|
| S0 | 標籤（Llama-70B，1514 筆）；小 Planner v3 比較 | 標籤解析失敗 < 2% |
| S1 | 標籤品質：200 筆用 gpt-oss-120b 交叉檢查；準備 30 筆給使用者人工檢查 | κ ≥ 0.4 |
| S2 | 判斷器 SFT × 3 折（只用 train） | validation 上 macro-F1 ≥ 0.5；解析失敗 < 2% |
| S3 | 選定小 Planner；需要時做 SFT warm start | JSON 解析率 ≥ 98%；`end_session` 有效率 ≥ 98% |
| S4 | Pipeline 完整性驗證（每個設計點都出現在實際 log 中，沒有任何截斷）；RL 可行性測試：每折 5 次更新 | 驗證器全部通過；reward 有組內變異；能從 checkpoint 續跑 |
| S5 | RL 正式訓練：每折 GRPO；fold1 上同預算比較 RLOO、PPO；三種控制器（固定、對偶上升、LLM） | 每次更新都存 checkpoint；validation 用來選 checkpoint |
| S6 | 評估：validation 上比較 A0-fix、A1、A2、A2+RL；凍結 → test 只跑一次；MultiWOZ 泛化 | 品質守門：coverage 差 ≥ −0.02，complete 最多少 1 集 |

## Reward v2（Planner RL；只用 train 統計）
- **目標達成：** 判斷器在最後一輪的 P(SATISFIED) + 0.5·P(PARTIAL)。
- **停止一致性：** 判斷器已判定滿足後還繼續的每一輪扣分；在 NOT 狀態、沒有 abandon 類理由時就離開則扣分。
- **限制（Lagrange）：** JSON 解析失敗、生成長度打到上限、候選全被 guard 擋下、判斷器輸出 UNKNOWN。
- **KL：** 對起始 policy 的 KL。
- **只評估、不進 reward：** coverage／complete（gpt-5-mini ledger）、與真人輪數分佈的 W1、Act TVD。

## Timeline (estimate, wall clock)
| 階段 | 預估 |
|---|---|
| S0 | 約 3.5 小時（進行中） |
| S1 | 約 1 小時 |
| S2 | 約 2 小時 |
| S3 | 約 2 小時 |
| S4 | 約 4 小時 |
| S5 | 約 1.5–2 天 |
| S6 | 約 0.5 天 |
| **合計** | **約 3 天** |

Pipeline 程式的開發與 S0–S2 並行進行。
