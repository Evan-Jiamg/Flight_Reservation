G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1; C=$G/code_snapshots/analysis_v4
n=0; for k in 0 1 2 3; do c=$(wc -l < $R/rep0_shard$k/nogate.jsonl); echo "rep0 shard$k: $c"; n=$((n+c)); done; echo "total $n/68"
cat $G/run_stageC_queue.log
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
if [ $n -eq 68 ] && [ ! -f $R/rep0/stageC_table.txt ]; then
  cd $C
  PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared setsid nohup /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python stageC_analysis.py --replicate 0 > $G/stageC_analysis_rep0.log 2>&1 < /dev/null &
  sleep 30; tail -3 $G/stageC_analysis_rep0.log
fi
