G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v16; OUT=$G/runs/v16_prelim
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
curl -s -m 5 http://127.0.0.1:8029/v1/models | grep -q gpt-oss-120b || { echo "gpt-oss (formal run) not up"; exit 1; }
pgrep -u mzjiang -f step0_coverage > /dev/null && { echo "step0 already running"; exit 1; }
cd $C
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
# no GPU of its own: the judge requests go to the formal run's gpt-oss (a few dozen calls, cached)
setsid nohup nice -n 10 timeout 2h $PY step0_coverage.py --fold 2 --splits $G/splits_v1.json --out $OUT/step0_f2.jsonl --workers 2 \
    > $OUT/step0.log 2>&1 < /dev/null &
sleep 20
tail -5 $OUT/step0.log | cut -c1-200
bash $G/watch_v16.sh | head -1
