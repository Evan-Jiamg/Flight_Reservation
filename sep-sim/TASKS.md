# TASKS（斷點續跑用；狀態：TODO / RUNNING / DONE / BLOCKED）

最後更新：2026-09-25 19:05

| ID | 任務 | 狀態 | 產出／位置（245 `/tmp2/mzjiang_usersim/grpo_planner/`） | 驗證 |
|---|---|---|---|---|
| P1 | E1.6 樹唯讀副本 | DONE | `trees/e1r_cf19400`（SHA256SUMS） | cf19400、無未提交修改 |
| P2 | pend arm（prompt、read_plan、Ditto 收尾、第 1 回合不結束） | DONE | commit 623ae8a 起 | test_e16_port 12/12（本機＋伺服器） |
| P3 | 兩個悄悄失敗修正：ledger max_tokens、Planner max_new 600→1536 | DONE | d7a3c9a | smoke v2 coverage 0.94、unparsed 0 |
| P4 | reward v3、Task 1 停止組、stop credit、停止 token 輔助監督 | DONE | fdbb06e、9c5c9a3、f9398f4 | 本機 dry-run＋伺服器 torch 測試（第 10、11 項）全過 |
| P5 | 批次生成（Planner、Ditto） | DONE | 027247e、fe7a24e | 批次 vs 單筆 log-prob 平均差 0.007、>0.5 佔 0% |
| P6 | smoke v2 / v3 ＋ verify_pipeline | DONE | `pend_smoke_v2`、`pend_smoke_v3` | PIPELINE VERIFICATION PASSED ×2 |
| T1 | Task 1 未訓練基準（56 段＋K+1，E1.6 同一評分指令） | RUNNING | `task1_pend_base_v1/`（run_pend_v4.log 的 COMPARE） | 比較表 |
| R1 | GRPO fold2（30 次更新，監督式自動續跑） | RUNNING | `pend_rl_fold2_v1/`、`run_rl_fold2.log` | 每 5 次 validation；verify --rl-dir |
| R2 | 選 checkpoint → 訓練後 Task 1 完整評估（fold2 test 以外不可用 adapter 評全部 56 段） | TODO | — | 對 E1.6 不退化 |
| R3 | Task 2 test 一次（`rollout_v4.py --arm pend --final`） | TODO | — | 回合數 vs 真人、vs 32B |
| R4 | fold0 / fold1 GRPO（時間允許） | TODO | — | — |
| D1 | 待使用者決定：自己生成需求切片 / goal-only 切法 | BLOCKED | — | — |
