#!/bin/bash
# Same-stack control for Stage A' on host 244 (V100, fp16, consistent env): no-offset retrain.
# Starts after the model download finishes and a GPU has >= 13000 MiB free.
M=/tmp2/mzjiang_usersim; G=$M/grpo_planner; C=$M/code_snapshots/stageA_cox_v1
PY=/home/mzjiang/miniconda3/envs/consistent/bin/python
BASE=$M/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
export PYTHONNOUSERSITE=1 HF_HOME=$M/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "^DONE" $M/dl_qwen7b.log; do sleep 60; done
while true; do
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | awk -F, '$2>=13000{print $1; exit}')
  [ -n "$GPU" ] && break; sleep 120
done
echo "start no-offset control on GPU$GPU $(date)"
cd $C
$PY train_stop_sft_cox.py --train $G/prism_pretrain/train.jsonl --validation $G/prism_pretrain/validation.jsonl \
  --base-model $BASE --gpu $GPU --compute-dtype float16 --micro-batch 2 --grad-accum 8 --max-length 1536 \
  --learning-rate 2e-5 --seed 20260923 --epochs 2 --no-offset --out $G/stageA_nooffset_244_v1 \
  > $G/stageA_nooffset_244_v1.log 2>&1
echo "no-offset rc=$? $(date)"
