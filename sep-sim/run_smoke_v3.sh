#!/bin/bash
# End-to-end smoke of the audited v3 pipeline on GPU1 only (GPU0 reserved for a teammate).
# SMOKE labels come from the UNTRAINED Qwen3-4B and exist only to exercise the code path.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; D=$G/dev_v3; S=$D/smoke; R=$G/stageC_v1
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /home/mzjiang/.secrets/openai.env; set +a
Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
mkdir -p $S; cd $D
echo "== 1 samples"; $PY build_judge_samples.py --episodes "ditto_rep0=$R/rep0_ditto_shard*/nogate.jsonl" \
  --episodes "ditto_rep1=$R/rep1_ditto_shard*/nogate.jsonl" --union $R/scenarios_union.json --out $S/samples.jsonl
echo "== 2 smoke labels (untrained 4B)"
$PY - <<EOF
import json
cf=json.load(open("$G/crossfit_manifest_v1.json")); f=[x for x in cf["folds"] if x["fold"]==0][0]
held=set(f["groups"][0]); train=set(f["scenarios"])-held
ids=[]; nt=nh=0
for l in open("$S/samples.jsonl"):
    s=json.loads(l)
    if s["conversation_id"] in train and nt<12 and s["t"]>=2: ids.append(s["id"]); nt+=1
    elif s["conversation_id"] in held and nh<6 and s["t"]>=2: ids.append(s["id"]); nh+=1
open("$S/smoke_ids.txt","w").write("\n".join(ids)); print("smoke ids", nt, nh)
EOF
$PY label_goal_status.py --samples $S/samples.jsonl --model $Q4 --out $S/labels_smoke.jsonl --ids $S/smoke_ids.txt --gpu 1
echo "== 3 judge train smoke (fold0 group0, 2 steps)"
rm -rf $S/judge_f0g0
$PY train_goal_judge.py --samples $S/samples.jsonl --labels $S/labels_smoke.jsonl --crossfit-manifest $G/crossfit_manifest_v1.json \
  --fold 0 --group 0 --base $Q4 --out $S/judge_f0g0 --gpu 1 --accum 2 --max-steps 2
echo "== 4a rollout a0 (1 episode)"
$PY rollout_ditto_v3.py --arm a0 --scenarios $R/scenarios_union.json --out-dir $S/a0 --gpu 1 --limit 1 --replicate 9
echo "== 4b rollout a2 (1 episode, fold0 group0, smoke judge)"
$PY rollout_ditto_v3.py --arm a2 --fold 0 --group 0 --judge-adapter $S/judge_f0g0 --judge-base $Q4 \
  --out-dir $S/a2 --gpu 1 --limit 1 --replicate 9
echo "== 4c leak gate must refuse a judge evaluated on its own training group"
$PY rollout_ditto_v3.py --arm a2 --fold 0 --group 1 --judge-adapter $S/judge_f0g0 --judge-base $Q4 \
  --out-dir $S/a2_bad --gpu 1 --limit 1 --replicate 9 2>&1 | grep -E "LEAK GATE" || echo "!! leak gate did NOT fire"
echo "SMOKE DONE $(date)"
