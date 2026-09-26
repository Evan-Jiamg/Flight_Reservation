#!/bin/bash
# Smaller-Planner probes (re-launch after the CUDA-init fix), 245 GPU1, sequential.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/v3_run1_probe; O=$G/v3_run1
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
snap() { ls -d $1/snapshots/*/ 2>/dev/null | head -1; }
LLAMA8=$(snap /tmp2/hf_shared/hub/models--meta-llama--Llama-3.1-8B-Instruct)
Q35=$(snap /tmp2/hf_shared/hub/models--Qwen--Qwen3.5-9B)
GOSS=$(snap /tmp2/hf_shared/hub/models--openai--gpt-oss-20b)
P32=/tmp2/TREC_UserSim_MingZhi/UserLM/00-models-v4GRPO-deps/Qwen2.5-32B-Instruct
cd $C
for spec in "qwen25_32b_nf4|$P32|--nf4" "qwen35_9b|$Q35|" "llama31_8b|$LLAMA8|--context 131072" "gptoss_20b|$GOSS|--dtype auto --context 131072"; do
  IFS='|' read -r name path extra <<< "$spec"
  rm -f $O/probe/$name.jsonl
  $PY planner_probe.py run --probe $O/probe/probe_set.jsonl --model $path $extra --gpu 1 \
    --out $O/probe/$name.jsonl > $O/probe/$name.log 2>&1
  echo "probe $name rc=$? $(date)"
done
echo "PROBES DONE $(date)"
