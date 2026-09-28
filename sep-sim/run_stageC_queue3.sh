#!/bin/bash
# Stack control for the Speaker comparison: UserLM under the consistent-test env (the env Ditto
# needed), replicate 0, same scenarios/seeds, same stageC_v2 runner. Isolates Speaker vs stack:
#   UserLM@blackwell vs UserLM@consistent-test = stack effect
#   UserLM@consistent-test vs Ditto@consistent-test = Speaker effect
G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1; C2=$G/code_snapshots/stageC_v2
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /home/mzjiang/.secrets/openai.env; set +a
MAXJ=5
running() { pgrep -fc "rollout_stop_sft.py" ; }
for K in 0 1 2 3; do
  OUT=$R/rep0_userlm_ct_shard$K
  while true; do
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=42000{print $1; exit}')
    [ -n "$GPU" ] && [ "$(running)" -lt $MAXJ ] && break
    sleep 120
  done
  echo "start userlm@consistent-test shard$K rep0 on GPU$GPU $(date)"
  ( cd $C2 && $PY rollout_stop_sft.py --scenarios $R/scenarios_shard$K.json --fold -1 \
      --side inner_union --gpu $GPU --log-prompts --replicate 0 --out-dir $OUT \
      --arm nogate=nogate --speaker userlm > $OUT.log 2>&1; echo "done userlm_ct shard$K rc=$? $(date)" ) &
  sleep 240
done
wait
echo "QUEUE3 DONE $(date)"
