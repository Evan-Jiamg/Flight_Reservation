#!/bin/bash
# v3 run 1 (approved 2026-09-25): judge labels from existing Ditto trajectories + smaller-Planner probe.
# Both lines on 245 GPU1 only (GPU0 reserved for a teammate). Frozen code snapshot.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/v3_run1; O=$G/v3_run1; R=$G/stageC_v1
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
snap() { ls -d $1/snapshots/*/ 2>/dev/null | head -1; }
LLAMA70=$(snap /tmp2/hf_shared/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct)
LLAMA8=$(snap /tmp2/hf_shared/hub/models--meta-llama--Llama-3.1-8B-Instruct)
Q35=$(snap /tmp2/hf_shared/hub/models--Qwen--Qwen3.5-9B)
GOSS=$(snap /tmp2/hf_shared/hub/models--openai--gpt-oss-20b)
P32=/tmp2/TREC_UserSim_MingZhi/UserLM/00-models-v4GRPO-deps/Qwen2.5-32B-Instruct
mkdir -p $O/judge $O/probe; cd $C
echo "models: $LLAMA70 | $LLAMA8 | $Q35 | $GOSS"
$PY build_judge_samples.py --episodes "ditto_rep0=$R/rep0_ditto_shard*/nogate.jsonl" \
  --episodes "ditto_rep1=$R/rep1_ditto_shard*/nogate.jsonl" --union $R/scenarios_union.json --out $O/judge/samples.jsonl
$PY planner_probe.py build --episodes "$R/rep0_ditto_shard*/nogate.jsonl" --out $O/probe/probe_set.jsonl
# line A: labels
( $PY label_goal_status.py --samples $O/judge/samples.jsonl --model $LLAMA70 --load-4bit --dtype bfloat16 \
    --gpu 1 --out $O/judge/labels_llama70b.jsonl > $O/judge/labels_llama70b.log 2>&1
  echo "LABELS rc=$? $(date)" ) &
sleep 180
# line B: probes (sequential, each failure logged and skipped)
(
  $PY planner_probe.py run --probe $O/probe/probe_set.jsonl --model $P32 --nf4 --gpu 1 --out $O/probe/qwen25_32b_nf4.jsonl > $O/probe/qwen25_32b.log 2>&1; echo "probe 32B rc=$?"
  $PY planner_probe.py run --probe $O/probe/probe_set.jsonl --model $Q35 --gpu 1 --out $O/probe/qwen35_9b.jsonl > $O/probe/qwen35_9b.log 2>&1; echo "probe qwen3.5-9b rc=$?"
  $PY planner_probe.py run --probe $O/probe/probe_set.jsonl --model $LLAMA8 --gpu 1 --context 131072 --out $O/probe/llama31_8b.jsonl > $O/probe/llama31_8b.log 2>&1; echo "probe llama8b rc=$?"
  $PY planner_probe.py run --probe $O/probe/probe_set.jsonl --model $GOSS --dtype auto --gpu 1 --context 131072 --out $O/probe/gptoss_20b.jsonl > $O/probe/gptoss_20b.log 2>&1; echo "probe gpt-oss-20b rc=$?"
  echo "PROBES DONE $(date)"
) &
wait
echo "RUN1 DONE $(date)"
