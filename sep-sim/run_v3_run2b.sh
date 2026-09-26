#!/bin/bash
# Smaller-Planner probe, reduced to TWO candidates (user, 2026-09-25): Qwen2.5-7B-Instruct and
# Llama-3.1-8B-Instruct, v3 architecture, 245 GPU1, in parallel. Waits for the probe labels.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/v3_run2; O=$G/v3_run1; R=$G/stageC_v1; P=$O/probe_v3
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
snap() { ls -d $1/snapshots/*/ 2>/dev/null | head -1; }
LLAMA8=$(snap /tmp2/hf_shared/hub/models--meta-llama--Llama-3.1-8B-Instruct)
Q7=$(snap /tmp2/hf_shared/hub/models--Qwen--Qwen2.5-7B-Instruct)
until grep -q "^DONE" $P/labels_probe.log 2>/dev/null; do sleep 60; done
cd $C
cat $O/judge/labels_llama70b.jsonl $P/labels_probe_llama70b.jsonl > $P/labels_merged.jsonl
$PY planner_probe_v3.py build --episodes "$R/rep0_ditto_shard*/nogate.jsonl" --labels $P/labels_merged.jsonl --out $P/probe_v3.jsonl
run() { $PY planner_probe_v3.py run --probe $P/probe_v3.jsonl --model $2 $3 --gpu 1 --out $P/$1.jsonl > $P/$1.log 2>&1; echo "probe_v3 $1 rc=$? $(date)"; }
run qwen25_7b $Q7 "" & sleep 60
run llama31_8b $LLAMA8 "--context 131072" & wait
$PY planner_probe_v3.py summary $P/qwen25_7b.jsonl $P/llama31_8b.jsonl > $P/summary.txt 2>&1
cat $P/summary.txt
echo "RUN2B DONE $(date)"
