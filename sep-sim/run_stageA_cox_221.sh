#!/bin/bash
# Stage A' on host 221 (P40, fp16, consistent env). Sequential jobs, frozen code snapshot.
# 1) same-stack baseline: original Stage A ep2 (bf16-trained on 245) evaluated here, no offsets
# 2) Cox position-offset SFT (main)          3) no-offset retrain (same-stack control)
set -uo pipefail
M=/tmp2/mzjiang_usersim; G=$M/grpo_planner; C=$M/code_snapshots/stageA_cox_v1
PY=/home/mzjiang/miniconda3/envs/consistent/bin/python
BASE=$M/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
export PYTHONNOUSERSITE=1 HF_HOME=$M/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $C
common="--train $G/prism_pretrain/train.jsonl --validation $G/prism_pretrain/validation.jsonl --base-model $BASE --gpu 0 --compute-dtype float16 --micro-batch 2 --grad-accum 8 --max-length 1536 --learning-rate 2e-5 --seed 20260923"
$PY train_stop_sft_cox.py $common --epochs 0 --no-offset --init-adapter $G/prism_sft_7b_v1/best --out $G/stageA_orig_ep2_on221 > $G/stageA_orig_ep2_on221.log 2>&1; echo "baseline rc=$? $(date)"
$PY train_stop_sft_cox.py $common --epochs 2 --out $G/stageA_cox_v1 > $G/stageA_cox_v1.log 2>&1; echo "cox rc=$? $(date)"
$PY train_stop_sft_cox.py $common --epochs 2 --no-offset --out $G/stageA_nooffset_221_v1 > $G/stageA_nooffset_221_v1.log 2>&1; echo "nooffset rc=$? $(date)"
echo "STAGE A PRIME DONE $(date)"
