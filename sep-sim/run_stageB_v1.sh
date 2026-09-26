#!/bin/bash
# Stage B inner TREC adaptation, three independent folds in parallel, from a frozen code snapshot.
set -euo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner
PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python
BASE=/tmp2/hf_shared/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
PRISM=$G/prism_sft_7b_v1
SNAP=$G/code_snapshots/stageB_v1
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
grep -q '"pass": true' $G/audits/stop_prompt_parity_audit.json
mkdir -p $SNAP
cp $G/stop_prompt.py $G/train_stop_sft.py $G/eval_stop_sft.py $G/compare_stop_sft.py $SNAP/
( cd $SNAP && sha256sum *.py > SHA256SUMS; sha256sum $PRISM/best/adapter_model.safetensors $PRISM/epoch1/adapter_model.safetensors $PRISM/epoch2/adapter_model.safetensors >> SHA256SUMS )
echo "snapshot $SNAP"; cat $SNAP/SHA256SUMS
cmp $PRISM/best/adapter_model.safetensors $PRISM/epoch2/adapter_model.safetensors && echo "best==epoch2 confirmed"
gpu_of=(0 1 0)
for fold in 0 1 2; do
  ( "$PY" $SNAP/train_stop_sft.py \
      --train $G/nested/fold${fold}_inner_train.jsonl \
      --validation $G/nested/fold${fold}_inner_validation.jsonl \
      --base-model $BASE --init-adapter $PRISM/best \
      --out $G/trec_inner_fold${fold}_7b_v1 --gpu ${gpu_of[$fold]} \
      --compute-dtype bfloat16 --epochs 2 --micro-batch 2 \
      --grad-accum 4 --max-length 2048 --learning-rate 5e-6 \
      > $G/trec_inner_fold${fold}_7b_v1.log 2>&1
    O=$G/trec_inner_fold${fold}_7b_v1; E=$G/stageB_eval_v1; mkdir -p $E
    for ep in 0 1 2; do
      if [ $ep = 0 ]; then A=$PRISM/best; else A=$O/epoch$ep; fi
      "$PY" $SNAP/eval_stop_sft.py --prompts $G/nested/fold${fold}_inner_validation.jsonl \
        --base-model $BASE --adapter $A --out $E/fold${fold}_ep${ep}_innerval.jsonl \
        --gpu ${gpu_of[$fold]} --max-length 2048 --purpose trec_inner_validation
    done >> $G/trec_inner_fold${fold}_7b_v1.log 2>&1
    echo "FOLD $fold DONE" >> $G/trec_inner_fold${fold}_7b_v1.log ) &
  sleep 30
done
# PRISM validation predictions for Stage A epoch1 vs epoch2 first-stop timing (GPU1)
( E=$G/stageA_eval_v1; mkdir -p $E
  for ep in 1 2; do
    "$PY" $SNAP/eval_stop_sft.py --prompts $G/prism_pretrain/validation.jsonl \
      --base-model $BASE --adapter $PRISM/epoch$ep --out $E/prism_val_ep${ep}.jsonl \
      --gpu 1 --max-length 1536 --purpose prism_validation
  done > $G/stageA_eval_v1.log 2>&1; echo "STAGEA EVAL DONE" >> $G/stageA_eval_v1.log ) &
wait
echo "ALL STAGE B DONE $(date)"
