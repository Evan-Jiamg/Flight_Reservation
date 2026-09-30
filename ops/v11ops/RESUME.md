# pend fold 2 GRPO 訓練紀錄：已停在 u25，可接續

封存時間：2026-09-27（使用者：「先不要訓練到 u30，但請你將之前的訓練紀錄存好，若日後要接續訓練也能繼續」）

## 狀態
- 訓練在 **u25** 依停止規則結束（B 下 u20、u25 兩次驗證都沒有超過 u5）。最後一個 checkpoint：`ckpt/u00025`（`ckpt/LATEST.json` = 25）。
- `u00025` 含有接續訓練需要的全部內容：`adapter/`、`optimizer.pt`、`rng.pt`、`state.json`（cfg、控制器狀態、history、best、task1_base、aux_anneal_start、介入紀錄）、`rl_manifest.json`。
- 較早的 checkpoint 只保留 adapter（optimizer 依 `--keep-optimizer-last` 已刪），可用來評估，但無法從它們接續訓練。
- 介入 B：從 u16 起 `w_dist = 1.0`，控制器下限為 1.0（`intervention_w_dist_v13.json`，sha 記在 `run_meta.jsonl` 和每個 checkpoint）。**接續時必須傳同一個檔案**，否則程式會拒絕執行。
- 訓練用過的程式快照：u1–u10 用 `pend_v11`，u11–u15 用 `pend_v12`，u16–u25 用 `pend_v13`。之後的 `pend_v14`、`pend_v15` 只加了重新評估功能，訓練行為與 v13 相同。

## 這個封存裡有什麼
- `run/`：`runs/pend_f2_v11` 在封存當下的完整副本，包括 rollouts、updates、validation、run_meta、llm_controller、best.json、所有 ckpt 和 log。
- `code_snapshots/`：`pend_v11`、`pend_v12`、`pend_v13`、`pend_v14`、`pend_v15`。
- `ops/`：續跑、guard、佔位和伺服器啟動腳本，以及介入檔案。
- `splits_v1.json`。
- `MANIFEST.sha256`：封存內每個檔案的 sha256，用 `sha256sum -c --quiet MANIFEST.sha256` 檢查。
- 封存設成唯讀，請不要在這裡直接接續訓練。

## 接續訓練（例如到 u30）
在**原本的**執行目錄 `/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11` 接續（訓練過程會在這裡繼續寫入）：

1. 伺服器：`bash /tmp2/mzjiang_usersim/grpo_planner/start_servers5.sh`（gpt-oss 在 8029，Planner vLLM 在 8031）。
2. GPU：用佔位程式 `gpu_holder2.py` 拿一張有 45 GiB 空間的卡。
3. 在 `code_snapshots/pend_v15` 目錄內執行（環境變數同 `run_v13_continue.sh`）：

```
python train_planner_rl.py --fold 2 --planner-path $Q4 --gpu <GPU> --G 4 --scenarios-per-update 4 --task1-convs 4 --updates 30 --val-every 5 --rollout-workers 4 --out /tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11 --resume --allow-code-change --intervention /tmp2/mzjiang_usersim/grpo_planner/intervention_w_dist_v13.json
```

   `run_v13_continue.sh` 已經包好這個流程：從 LATEST 接續，以 5 次更新為一段，每段跑 verify，套用停止規則，最多到 30。

4. 接續完成後，跑一次 `verify_pipeline.py --rl-dir ...`，必須 PASSED。

如果原本的執行目錄已經遺失，就把 `run/` 複製回一個新位置（`cp -a`，再 `chmod -R u+w`），並把 `--out` 指向那裡。checkpoint 內的路徑都是相對於執行目錄。

## pend_v17 程式快照（fix round 1, 2026-09-30）
- `ops/local_sha_v17.txt` 是 **working copy 的位元組** 的 sha256（本機 `core.autocrlf=true`，.py 可能是 CRLF）。
  因此 `code_snapshots/pend_v17` 必須用 working copy 打包（tar）上傳，不可在別台機器重新 checkout 後再比對。
- `.sh` 一律 LF（`.gitattributes`: `*.sh text eol=lf`）。
- 腳本：`run_v17_smoke.sh`（GPU smoke，先跑）→ `run_v17_fold.sh F`（訓練＋verify）→ `run_v17_test.sh F`（test、boot、verify、threshold control）。
- 佔位程式（fix round 2）：`run_v17_fold.sh` 需要 `gpu_holder2.py` 已在執行，否則直接停止。`run_v17_smoke.sh` 若是自己啟動佔位程式，
  結束時會把它（以及它自己啟動的 vLLM servers）關掉；之後要先重新啟動佔位程式再跑 fold：
  `rm -rf /tmp2/mzjiang_usersim/grpo_planner/hold && mkdir -p /tmp2/mzjiang_usersim/grpo_planner/hold && PYTHONNOUSERSITE=1 setsid nohup /home/mzjiang/miniconda3/envs/consistent-test/bin/python /tmp2/mzjiang_usersim/grpo_planner/gpu_holder2.py /tmp2/mzjiang_usersim/grpo_planner/hold >> /tmp2/mzjiang_usersim/grpo_planner/gpu_holder2.log 2>&1 < /dev/null &`
  若佔位程式不是 smoke 啟動的，smoke 只把訓練 GPU 還給它，不會關掉佔位程式或已在執行的 servers。
- fold 2 的 base（fix round 2，使用者決定）：`run_v17_test.sh` 對每個 fold（包括 fold 2）都用 `--include-base` 以 v17 定義重新評估 base
  （fp32 P_end、human turns 以 t_max 截斷）；v16 `runs/pend_f2_v16/test_boot.txt` 的 u0 列只以「v16-base-ref」參考列印出，不當作 base。
