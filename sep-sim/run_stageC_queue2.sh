#!/bin/bash
# Stage C job queue v2. Ditto runs under the consistent-test env (transformers 5.7.0 has
# qwen3_vl; blackwell-kto-test 4.56.1 does not). UserLM replicate 1 keeps the replicate-0
# stack (blackwell-kto-test, stageC_v1 code). Launch when a GPU has >= 42000 MiB free and
# fewer than MAXJ of our rollouts run. Jobs resume from their own output files.
G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1
C1=$G/code_snapshots/stageC_v1; C2=$G/code_snapshots/stageC_v2
PY_U=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python
PY_D=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /home/mzjiang/.secrets/openai.env; set +a
MAXJ=4
JOBS=()
for k in 0 1 2 3; do JOBS+=("$C2 $PY_D ditto $k 0 $R/rep0_ditto_shard$k"); done
for k in 0 1 2 3; do JOBS+=("$C1 $PY_U userlm $k 1 $R/rep1_shard$k"); done
running() { pgrep -fc "rollout_stop_sft.py" ; }
for job in "${JOBS[@]}"; do
  set -- $job; CODE=$1; PY=$2; SPK=$3; K=$4; REP=$5; OUT=$6
  while true; do
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2>=42000{print $1; exit}')
    [ -n "$GPU" ] && [ "$(running)" -lt $MAXJ ] && break
    sleep 120
  done
  EXTRA=""; [ "$SPK" = ditto ] && EXTRA="--speaker ditto"
  [ -f $OUT.log ] && mv $OUT.log $OUT.log.prev_$(date +%H%M%S)
  echo "start $SPK shard$K rep$REP on GPU$GPU $(date)"
  ( cd $CODE && $PY rollout_stop_sft.py --scenarios $R/scenarios_shard$K.json --fold -1 \
      --side inner_union --gpu $GPU --log-prompts --replicate $REP --out-dir $OUT \
      --arm nogate=nogate $EXTRA > $OUT.log 2>&1; echo "done $SPK shard$K rep$REP rc=$? $(date)" ) &
  sleep 240
done
wait
echo "QUEUE2 DONE $(date)"
