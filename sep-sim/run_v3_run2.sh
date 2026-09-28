#!/bin/bash
# v3 run 2: smaller-Planner probe under the v3 Planner architecture (245 GPU1 only).
# 32B is NOT re-run: its logged (kept) data is the reference, as requested.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/v3_run2; O=$G/v3_run1; R=$G/stageC_v1
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
snap() { ls -d $1/snapshots/*/ 2>/dev/null | head -1; }
LLAMA70=$(snap /tmp2/hf_shared/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct)
LLAMA8=$(snap /tmp2/hf_shared/hub/models--meta-llama--Llama-3.1-8B-Instruct)
Q35=$(snap /tmp2/hf_shared/hub/models--Qwen--Qwen3.5-9B)
Q7=$(snap /tmp2/hf_shared/hub/models--Qwen--Qwen2.5-7B-Instruct)
GOSS=$(snap /tmp2/hf_shared/hub/models--openai--gpt-oss-20b)
P=$O/probe_v3; mkdir -p $P; cd $C
# 1. label the probe episodes' samples first (separate process; greedy, identical to the main labeller)
$PY - <<EOF
import json, glob, hashlib
eps=[json.loads(l) for p in sorted(glob.glob("$R/rep0_ditto_shard*/nogate.jsonl")) for l in open(p)]
eps=[e for e in eps if e["seed"]==0]; eps.sort(key=lambda e: hashlib.sha256(e["conversation_id"].encode()).hexdigest())
ids=["ditto_rep0|%s|s0|r0|t%d"%(e["conversation_id"],s["t"]) for e in eps[:20] for s in e["trace"] if s["decision"]=="continue"]
open("$P/probe_label_ids.txt","w").write("\n".join(ids)); print("probe label ids", len(ids))
EOF
$PY label_goal_status.py --samples $O/judge/samples.jsonl --model $LLAMA70 --load-4bit --dtype bfloat16 --gpu 1 \
  --ids $P/probe_label_ids.txt --out $P/labels_probe_llama70b.jsonl > $P/labels_probe.log 2>&1
echo "probe labels rc=$? $(date)"
cat $O/judge/labels_llama70b.jsonl $P/labels_probe_llama70b.jsonl > $P/labels_merged.jsonl
$PY planner_probe_v3.py build --episodes "$R/rep0_ditto_shard*/nogate.jsonl" --labels $P/labels_merged.jsonl --out $P/probe_v3.jsonl
run() { $PY planner_probe_v3.py run --probe $P/probe_v3.jsonl --model $2 $3 --gpu 1 --out $P/$1.jsonl > $P/$1.log 2>&1; echo "probe_v3 $1 rc=$? $(date)"; }
run qwen35_9b $Q35 "" & sleep 60
run llama31_8b $LLAMA8 "--context 131072" & wait
run qwen25_7b $Q7 ""
run gptoss_20b $GOSS "--dtype auto --context 131072"
$PY planner_probe_v3.py summary $P/qwen35_9b.jsonl $P/llama31_8b.jsonl $P/qwen25_7b.jsonl $P/gptoss_20b.jsonl > $P/summary.txt 2>&1
cat $P/summary.txt
echo "RUN2 DONE $(date)"
