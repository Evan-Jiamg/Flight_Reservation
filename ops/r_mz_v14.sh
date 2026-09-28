G=/tmp2/mzjiang_usersim/grpo_planner; S=$G/code_snapshots/pend_v14; M=/tmp2/MingZhi_HcWang/MingZhi_Code
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
chmod u+w $M/code/train_planner_rl.py $M/code/rl_algos.py $M/code/verify_pipeline.py 2>/dev/null
cp -p $S/train_planner_rl.py $S/rl_algos.py $S/verify_pipeline.py $S/test_reselect.py $S/local_sha_v14.txt $M/code/
cp -p $G/run_v14_reselect.sh $G/reselect_boot.py $M/ops/ && chmod -x $M/ops/run_v14_reselect.sh
cd $M/code && sha256sum -c --quiet local_sha_v14.txt && echo "MingZhi_Code/code = pend_v14: sha OK ($(wc -l < local_sha_v14.txt) files)"
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=$M/_pycache E1R_TREE=$G/trees/e1r_cf19400 V2FIX_TREE=/home/mzjiang/Sep-Simulator
nice -n 19 $PY -m pytest -q -p no:cacheprovider --ignore=test_batching_server.py --ignore=test_rl_algos_server.py 2>&1 | tail -1
cat >> $M/README.md <<'MDEOF'

## v14 更新（2026-09-27，已同步）：用 8 seeds 重新選 checkpoint

`code/` 現在等於 `pend_v14`（40 個檔，`sha256sum -c local_sha_v14.txt`）。

- `train_planner_rl.py --reselect-seeds 0 1 2 3 4 5 6 7`：把訓練中驗證過的每個 checkpoint（u0、u5、…）用相同流程（同驗證對話、溫度、環境、unclean 重跑規則、選擇分數公式）以 8 個 seeds 重新驗證；結果寫進 `reselect.jsonl`，最佳者寫進 `reselect_best.json`（同分取較早的 update）。不會寫入 validation.jsonl、best.json 或任何 checkpoint。
- `rl_algos.py`：`TorchLearner.load_policy` 只載入 checkpoint 的 adapter（舊 checkpoint 的 optimizer 已刪）；載入後 policy sha 必須與 checkpoint 紀錄相同。
- `verify_pipeline.py`：重選資料只能在驗證 id、只能是訓練中驗證過的 checkpoint、只能用宣告的 seeds，reselect_best 必須與分數一致；並做與 validation 相同的結構 / 截斷 / pend / few-shot / R0 歸屬檢查。
- `test_reselect.py`：上述行為的測試。
- `ops/run_v14_reselect.sh`：正式實驗的重選流程（等正式訓練自然結束後才執行；不會啟動最終 test）。`ops/reselect_boot.py`：以驗證對話為單位的配對 bootstrap（會先自我檢查與 summary 分數一致）。
MDEOF
echo "live trainer: $(pgrep -u mzjiang -f 'train_planner_rl.py --fold 2' | head -1)"
