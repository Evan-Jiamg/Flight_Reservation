# TASKS（斷點續跑用；狀態：TODO / RUNNING / DONE / BLOCKED）

最後更新：2026-09-25 14:40

| ID | 任務 | 負責 | 狀態 | 產出／位置 | 驗證 |
|---|---|---|---|---|---|
| T01 | 架構 v3（prompt、截斷、結束語意、判斷器 builder） | main | DONE | commit 92fff37 | 伺服器 7 組測試全過 |
| T02 | `splits_v1.json`（train/val/test，防洩漏） | main | DONE | 245 `grpo_planner/splits_v1.json` sha a3591933 | 建檔時的斷言全過 |
| T03 | 標籤：Llama-70B 產生 1514 筆 | 245 GPU1 | RUNNING | `v3_run1/judge/labels_llama70b.jsonl` | 解析失敗率 |
| T04 | 小 Planner v3 比較（4 個模型，teacher-forced 200 步） | 245 GPU1 | RUNNING | `v3_run1/probe_v3/summary.txt` | S3 閘門 |
| T05 | `Task2Env`（評估與 RL 共用） | main | DONE (code) | `task2_env.py` | 待 S4 的伺服器 smoke |
| T06 | RL 訓練器：GRPO／RLOO／PPO、reward v2、控制器、checkpoint 與續跑 | sub-agent A | TODO | `rl_*.py`、`train_planner_rl.py` | 單元測試 + 伺服器 smoke |
| T07 | 判斷器：κ 交叉檢查、每折 SFT（依 splits）、狀態機率、MultiWOZ 泛化 | sub-agent B | TODO | `judge_*.py`、`eval_multiwoz_judge.py` | 單元測試 + 伺服器 |
| T08 | 長跑佇列（續跑、心跳、重試）與 Pipeline 完整性驗證器 | sub-agent C | TODO | `jobq.py`、`verify_pipeline.py` | 單元測試 |
| T09 | S1 標籤品質（κ）＋ 30 筆人工檢查清單 | main | TODO | — | κ ≥ 0.4 |
| T10 | S2 判斷器 SFT × 3 折 | GPU | TODO | — | S2 閘門 |
| T11 | S3 選定 Planner（+ 需要時做 SFT） | main | TODO | — | S3 閘門 |
| T12 | S4 完整性驗證與 RL 可行性測試 | main | TODO | — | S4 閘門 |
| T13 | S5 RL 正式訓練 | GPU | TODO | — | — |
| T14 | S6 評估、test、MultiWOZ、報告 | main | TODO | — | — |
