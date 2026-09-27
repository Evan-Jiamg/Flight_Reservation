G=/tmp2/mzjiang_usersim/grpo_planner; S=$G/code_snapshots/pend_v13; M=/tmp2/MingZhi_HcWang/MingZhi_Code
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
echo "live trainer before: $(pgrep -u mzjiang -f 'train_planner_rl.py --fold 2' | head -1)"
cp -p $S/train_planner_rl.py $S/rl_controllers.py $S/verify_pipeline.py $S/test_intervention.py $S/local_sha_v13.txt $M/code/
cp -p $G/run_v13_continue.sh $G/run_v13_guard.sh $G/intervention_w_dist_v13.json $M/ops/
chmod -x $M/ops/run_v13_continue.sh $M/ops/run_v13_guard.sh
cd $M/code && sha256sum -c --quiet local_sha_v13.txt && echo "MingZhi_Code/code = pend_v13: sha OK ($(wc -l < local_sha_v13.txt) files)"
diff -rq $S $M/code 2>&1 | grep -v "__pycache__\|Only in $M/code" | head -5
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=$M/_pycache E1R_TREE=$G/trees/e1r_cf19400 V2FIX_TREE=/home/mzjiang/Sep-Simulator
nice -n 19 $PY -m pytest -q -p no:cacheprovider --ignore=test_batching_server.py --ignore=test_rl_algos_server.py 2>&1 | tail -1
cat >> $M/README.md <<'MDEOF'

---

## v13 更新（2026-09-27，已同步）

`code/` 現在等於正式實驗使用的快照 `pend_v13`（39 個檔，`sha256sum -c local_sha_v13.txt`）。相對 v12 的變更：

- `train_planner_rl.py`：新增 `--intervention FILE`（使用者核准的控制器介入）。權重值只在下一次 update 前設定一次；控制器上下限在每次啟動時都會套用；介入紀錄寫進 run_meta 與每個 checkpoint；之後每次 resume 都必須傳同一個檔案（檔案內容改變或漏傳都會拒絕執行）。
- `rl_controllers.py`：LLM 控制器新增 `set_bounds` / `intervene`；rollback 會被夾在目前的上下限內，不會把人工決定倒回去。
- `verify_pipeline.py`：新增 `rl.intervention` 檢查（介入後第一個 update 使用設定值、之後權重都在新上下限內）。
- `test_intervention.py`：上述行為的測試（含 dry-run resume）。
- `ops/intervention_w_dist_v13.json`：本次介入 B（w_dist 設回 1.0，控制器下限 1.0）。
- `ops/run_v13_continue.sh`、`ops/run_v13_guard.sh`：正式實驗的 v13 續跑與 OOM 防護（路徑寫死為正式實驗，請勿直接執行，改變數方式同上）。
MDEOF
echo "live trainer after: $(pgrep -u mzjiang -f 'train_planner_rl.py --fold 2' | head -1)"
ls $M/code | wc -l; tail -3 $M/README.md
