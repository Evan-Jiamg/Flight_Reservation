G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/stage_c_code
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
cmp stop_prompt.py $G/stop_prompt.py && echo "stop_prompt identical to Stage B copy"
sha256sum *.py *.md > SHA256SUMS; date >> SHA256SUMS
PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python
PYTHONNOUSERSITE=1 $PY test_task2_episode.py | tail -1
PYTHONNOUSERSITE=1 $PY test_analyzers.py | cut -c1-60
cd /home/mzjiang/Sep-Simulator && PYTHONNOUSERSITE=1 $PY -c 'import inspect; from sepsim import planner_prompt as PP; print(inspect.signature(PP.user_prompt))' 2>&1 | tail -1
ls -la /tmp2/hchsu/trec2026-usersim-benchmark/.env /tmp2/hchsu/trec2026-usersim-benchmark/tools/.env /home/mzjiang/Sep-Simulator/.env 2>&1 | cut -c1-120
grep -n "load_dotenv\|\.env" /tmp2/hchsu/trec2026-usersim-benchmark/tools/metrics/judge.py | head -8
for f in 0 1 2; do echo "fold$f:"; grep -v "^Loading\|Warning\|warn" $G/trec_inner_fold${f}_7b_v1.log | tail -3 | cut -c1-250; done
tail -2 $G/stageA_eval_v1.log
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
