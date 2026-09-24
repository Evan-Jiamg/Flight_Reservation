#!/bin/bash
# Ditto replicate 1 (same env/code as Ditto replicate 0: consistent-test + stageC_v2), 4 shards.
G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1; C2=$G/code_snapshots/stageC_v2
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /home/mzjiang/.secrets/openai.env; set +a
MAXJ=4
running() { pgrep -fc "rollout_stop_sft.py" ; }
for K in 0 1 2 3; do
  OUT=$R/rep1_ditto_shard$K
  while true; do
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=42000{print $1; exit}')
    [ -n "$GPU" ] && [ "$(running)" -lt $MAXJ ] && break
    sleep 120
  done
  echo "start ditto shard$K rep1 on GPU$GPU $(date)"
  ( cd $C2 && $PY rollout_stop_sft.py --scenarios $R/scenarios_shard$K.json --fold -1 \
      --side inner_union --gpu $GPU --log-prompts --replicate 1 --out-dir $OUT \
      --arm nogate=nogate --speaker ditto > $OUT.log 2>&1; echo "done ditto shard$K rep1 rc=$? $(date)" ) &
  sleep 240
done
wait
echo "DITTO REP1 DONE $(date)"
