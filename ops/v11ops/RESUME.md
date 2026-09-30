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
- 佔位程式（fix round 2；2026-09-30 起改用 `gpu_holder3.py` + `start_servers6.sh`）：`run_v17_fold.sh` 需要 `gpu_holder3.py` 已在執行，否則直接停止；
  舊的 `gpu_holder2.py` 若還在跑，三支 run_v17 腳本都會停下（兩個佔位程式不可並存，先 `touch hold/stop`）。`run_v17_smoke.sh` 若是自己啟動佔位程式，
  結束時會把它（以及它自己啟動的 vLLM servers）關掉；之後要先重新啟動佔位程式再跑 fold：
  `rm -rf /tmp2/mzjiang_usersim/grpo_planner/hold && mkdir -p /tmp2/mzjiang_usersim/grpo_planner/hold && CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 setsid nohup /home/mzjiang/miniconda3/envs/consistent-test/bin/python /tmp2/mzjiang_usersim/grpo_planner/gpu_holder3.py /tmp2/mzjiang_usersim/grpo_planner/hold >> /tmp2/mzjiang_usersim/grpo_planner/gpu_holder3.log 2>&1 < /dev/null &`
  若佔位程式不是 smoke 啟動的，smoke 只把訓練 GPU 還給它，不會關掉佔位程式或已在執行的 servers。
- `gpu_holder3.py`（使用者 2026-09-30：「不要堅持用哪張卡，兩張加起來夠就跑」）：以 1 GiB 為單位逐步佔記憶體（保留 1 GiB 空閒），
  每 0.25 秒在兩種配置 × 兩種 GPU 順序中選缺口最小者（缺口相差 ≤ 2 GiB 時，選缺口集中在較少張卡上的配置）：P1 = 一張卡放 gpt-oss +
  Planner vLLM（91 GiB）、另一張訓練（45）；P2 = 一張卡只放 gpt-oss（77）、另一張放 Planner vLLM + 訓練（16 + 45 = 61）。
  整個配置都佔滿才 COMMIT，寫 `role_oss`、`role_planner`、`role_train`（最後寫，代表已 commit）與 `role_server`（= role_oss）。
  大小可用 `SERVER_NEED_GIB / OSS_NEED_GIB / PLANNER_NEED_GIB / TRAIN_NEED_GIB / HOLD_MARGIN_GIB` 調整；目前配置見 `hold/plan`（每個 tick 重寫，
  run 腳本以它確認佔位程式還活著、HOLD_DIR 正確）。啟動時會清掉舊的 role_*/replan/servers_up/stop。
- `start_servers6.sh`：啟動前只釋放讓 free ≥ util × total + slack 的量（gpt-oss 3 GiB、planner 2 GiB；P1 的 planner 拿走伺服器卡上剩下的全部；
  P2 的訓練卡一律保留 45 GiB），寫好 serve.sh 後立刻啟動；若釋放後 free < util × total + 1 GiB 就不啟動、改為重新配置。
  啟動失敗算「記憶體被搶」只在兩種情況：vLLM 啟動檢查的 "less than desired GPU memory utilization"，或 OOM 類錯誤且 nvidia-smi 顯示
  啟動後該卡 free 變少／出現別人的新 process。此時只殺掉自己剛啟動的 server（只殺該 session），立刻把該卡的 target 設回交接前的值（佔位程式馬上收回釋放出的記憶體），
  寫 `hold/replan` 讓佔位程式重新配置；退避等待（60、120、300 秒，之後每次 600 秒；使用者決定：有證據的搶卡永不停止）放在下一次 commit 之後、
  下一次啟動之前，這段時間記憶體都由佔位程式佔著。釋放後記憶體仍不夠、因而沒有啟動的情況（不確定原因），
  同一 server 同一張卡 3 次就 exit 1；其他錯誤（模型路徑、設定、沒人搶卻 OOM）也 exit 1。run 腳本在 `take_train_gpu` 會先把訓練卡 target
  設回 45 再等。servers 已在執行但佔位程式剛重啟時，它寫 `hold/servers_up`，佔位程式只配置訓練 GPU。測試：`test_start_servers6.py`（Git Bash）。
- 從舊佔位程式切換：`nohup bash cutover_holder3.sh > cutover_holder3.log 2>&1 &`（在 cfda5 上）。先檢查 `hold/role_server`、`hold/role_train`
  存在且 `gpu_grab.py` 的 pid 只在 GPU1 的 UUID 上，否則什麼都不動就中止；接著先把 `grab1/target`、再把 `hold/target_<g>` 固定在目前大小，
  在 `hold3/` 啟動 holder3。每秒（以及每一步之後）寫 `hold3/extra_<g>`（舊佔位程式在該卡還佔多少，取 min(status, target)；holder3 選配置時算作可用，
  實際只佔真正空出的記憶體；超過 10 秒沒更新就不算），
  只在 holder3 在那張卡的 target 大於已佔量時把一個舊佔位程式降 2 GiB，並等 holder3 吸收；10 秒沒吸收就全部凍結並警告，120 秒仍沒吸收就中止
  （舊的維持凍結、holder3 繼續跑、extra 檔仍正確，手動停掉舊的之後要 `rm hold3/extra_*`）。holder3 commit 後，剩下的舊記憶體是多餘的，直接放掉；
  最後 touch 舊的 stop 檔並刪掉 extra 檔。兩個佔位程式不可共用 `hold/`，之後所有 run_v17 腳本都要帶
  `export HOLD_DIR=/tmp2/mzjiang_usersim/grpo_planner/hold3`；舊佔位程式還在跑時，三支腳本會直接停下。模擬測試：`test_cutover_holder3.py`（約 2.5 分鐘）。
- 部署：`gpu_holder3.py`、`start_servers6.sh`、`cutover_holder3.sh` 要複製到 `/tmp2/mzjiang_usersim/grpo_planner/`（run 腳本從那裡呼叫）；`start_servers5.sh` / `gpu_holder2.py` 保留給封存的舊 run。
- fold 2 的 base（fix round 2，使用者決定）：`run_v17_test.sh` 對每個 fold（包括 fold 2）都用 `--include-base` 以 v17 定義重新評估 base
  （fp32 P_end、human turns 以 t_max 截斷）；v16 `runs/pend_f2_v16/test_boot.txt` 的 u0 列只以「v16-base-ref」參考列印出，不當作 base。

- GPU 編號（audit B round 4, R4-S1）：cfda5 的兩張卡是同一型號（NVIDIA RTX PRO 6000 Blackwell Server Edition，PCI 0A / AE，2026-09-30 以 nvidia-smi 確認），所以 PCI_BUS_ID 與 torch 預設排序的編號相同。手動啟動佔位程式時仍請帶 `CUDA_DEVICE_ORDER=PCI_BUS_ID`（上方指令已含），與 run_v17_*.sh 一致。
- verify 的 --strict（audit C round 4）：run_v17_*.sh 不帶 --strict；rl.sft_short_conversations 在有單訊息對話時是 WARN。若日後加上 --strict，含單訊息對話的 fold 會被判失敗。
