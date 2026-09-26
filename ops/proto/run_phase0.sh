#!/bin/bash
# Phase 0 + vLLM speed prototype (read-only measurements; no training, no gpt-oss needed).
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; P=$G/phase0
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
pickgpu() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \
            awk -F', ' '{f=int(($2-$3)/1024); if (f>=40) print f" "$1}' | sort -nr | head -1 | cut -d" " -f2; }
cd $P
echo "=== 1 splits $(date)"; $PY p2_splits.py 2>&1 | tail -8
GPU=""; while [ -z "$GPU" ]; do GPU=$(pickgpu); [ -z "$GPU" ] && sleep 60; done
echo "=== 2 vLLM prototype on GPU $GPU (memory share ${VLLM_UTIL:-0.15}) $(date)"
CUDA_VISIBLE_DEVICES=$GPU VLLM_UTIL=${VLLM_UTIL:-0.15} $PY p1_vllm.py > $P/p1.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "vLLM at 0.15 failed (rc=$rc): $(grep -iE 'error|memory' $P/p1.log | tail -2 | cut -c1-200); retrying at 0.20"
  CUDA_VISIBLE_DEVICES=$GPU VLLM_UTIL=0.20 $PY p1_vllm.py > $P/p1.log 2>&1; rc=$?
fi
echo "vllm rc=$rc"; cat $P/vllm_summary.json 2>/dev/null
[ $rc -ne 0 ] && tail -20 $P/p1.log | cut -c1-300
echo "=== 3 HF: KL u0->u1, HF speed, mismatch $(date)"
CUDA_VISIBLE_DEVICES=$GPU $PY p0_hf.py > $P/p0.log 2>&1
echo "hf rc=$?"; cat $P/hf_summary.json 2>/dev/null || tail -20 $P/p0.log | cut -c1-300
echo "PHASE0 DONE $(date)"
