# TASKS（斷點續跑用；狀態：TODO / RUNNING / DONE / BLOCKED）

最後更新：2026-09-25 15:10

| ID | 任務 | 負責 | 狀態 | 產出／位置 | 驗證 |
|---|---|---|---|---|---|
| T01 | 架構 v3（prompt、截斷、結束語意、判斷器 builder） | main | DONE | commit 92fff37 | 伺服器 7 組測試全過 |
| T02 | `splits_v1.json`（train/val/test，防洩漏） | main | DONE | 245 `grpo_planner/splits_v1.json` sha a3591933 | 建檔時的斷言全過 |
| T03 | 標籤：Llama-70B 產生 1514 筆 | 245 GPU1 | RUNNING (375/1514, 0 unparsed) | `v3_run1/judge/labels_llama70b.jsonl` | 解析失敗率 |
| T04 | 小 Planner 測試：Qwen2.5-7B、Llama-3.1-8B，加同 prompt 的 32B NF4 參考 | 245 GPU1 | RUNNING | `v3_run1/probe_v3/summary_with32.txt` | 對 32B 的差距 |
| T05 | `Task2Env`（評估與 RL 共用） | main | DONE | `task2_env.py` | — |
| T05b | 評估 CLI `rollout_v4.py`（leak gate、test 需 --final、--smoke） | main | DONE | `rollout_v4.py` | — |
| T05c | **E1.6 移植**：唯讀樹 `trees/e1r_cf19400`、`ditto_e16.py`、arm e16/final、v3 的 ACT_FULL、verify 的 e16/final 檢查 | main | DONE (code) | code_snapshots/e16port_v1 | 本機與伺服器 test_e16_port 8/8、verify 測試 OK |
| T05d | E1.6 移植 smoke（e16 → final，fold1 val，3 集，7B Planner，未訓練判斷器） | 245 GPU1 | QUEUED（等 ≥45GB） | `e16port_smoke_v1/`、`run_e16smoke.log` | verify_pipeline 全過 |
| T06 | RL 訓練器：GRPO、reward v2、控制器、checkpoint 與續跑（預設 arm=final） | sub-agent A | DONE (code) | `rl_*.py`、`train_planner_rl.py` | 本機 21/21；伺服器 smoke 待跑 |
| T07 | 判斷器：κ、每折 SFT、狀態機率、MultiWOZ | sub-agent B | DONE (code) | `judge_*.py`、`train_goal_judge_fold.py` | 本機測試過；伺服器待跑 |
| T08 | 長跑佇列與 Pipeline 完整性驗證器 | sub-agent C | DONE (code) | `jobq.py`、`verify_pipeline.py` | 本機測試過 |
| T09 | 標籤品質（κ）＋ 30 筆人工檢查清單 | main | TODO（等 T03） | — | κ ≥ 0.4 |
| T10 | 判斷器 SFT × 3 折 | GPU | TODO（等 T09） | — | S2 閘門 |
| T11 | 選定 RL 用的 Planner | main | TODO（等 T04、T05d） | — | 對 32B 不明顯退步 |
| T12 | Baseline e16 validation rollout + RL 可行性 | main | TODO | — | verify 全過 |
| T13 | RL 正式訓練 | GPU | TODO | — | — |
| T14 | 評估、test、MultiWOZ、報告 | main | TODO | — | — |
