G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1; C=$G/code_snapshots/stageC_v1
ps -o user,pid,etime,cmd -p 724883 | cut -c1-120
wc -l $R/rep0_shard1/nogate.jsonl $R/rep1_shard1/nogate.jsonl 2>&1 | head -3
mv $R/rep0_shard1.log $R/rep0_shard1.oom_attempt1.log; mv $R/rep1_shard1.log $R/rep1_shard1.oom_attempt1.log
cat > $R/queue_shard1.sh <<'EOF'
#!/bin/bash
# Re-queue of shard 1 after OOM at load (GPU0 shared with another user's process).
# Waits until a GPU has >= 42000 MiB free, then runs rep0 and rep1 there. Same frozen code.
G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1; C=$G/code_snapshots/stageC_v1
PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /home/mzjiang/.secrets/openai.env; set +a
cd $C
for rep in 0 1; do
  while true; do
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=42000{print $1; exit}')
    [ -n "$GPU" ] && break; sleep 120
  done
  echo "shard1 rep$rep start on GPU$GPU $(date)"
  $PY rollout_stop_sft.py --scenarios $R/scenarios_shard1.json --fold -1 --side inner_union \
    --gpu $GPU --log-prompts --replicate $rep --out-dir $R/rep${rep}_shard1 --arm nogate=nogate \
    > $R/rep${rep}_shard1.log 2>&1
  echo "shard1 rep$rep rc=$? $(date)"
done
EOF
setsid nohup bash $R/queue_shard1.sh > $R/queue_shard1.out 2>&1 < /dev/null &
sleep 3; cat $R/queue_shard1.out; echo "--- smoke"
grep -E "^\s+\[|DONE" $R/smoke_crn.log | cut -c1-200
