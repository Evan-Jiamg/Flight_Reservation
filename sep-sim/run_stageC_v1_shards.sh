#!/bin/bash
# Stage C v1, sharded: replaces the two-half schedule of run_stageC_v1.sh (which never
# started its halves). Same frozen code (code_snapshots/stageC_v1), same union scenario
# list; 4 interleaved shards, 2 per GPU; each shard runs replicate 0 then replicate 1.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner
C=$G/code_snapshots/stageC_v1
R=$G/stageC_v1
PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /home/mzjiang/.secrets/openai.env; set +a
cd $C
$PY - <<'EOF'
import json
R = "/tmp2/mzjiang_usersim/grpo_planner/stageC_v1"
u = json.load(open(f"{R}/scenarios_union.json"))
for k in range(4):
    part = u["scenarios"][k::4]
    json.dump({"t_max": 10, "seeds": [0, 1], "scenarios": [dict(s, order=i) for i, s in enumerate(part)]},
              open(f"{R}/scenarios_shard{k}.json", "w"), indent=1)
    print("shard", k, len(part), "scenarios")
EOF
shard() {  # $1 shard, $2 gpu
  for rep in 0 1; do
    $PY rollout_stop_sft.py --scenarios $R/scenarios_shard$1.json --fold -1 --side inner_union \
      --gpu $2 --log-prompts --replicate $rep --out-dir $R/rep${rep}_shard$1 --arm nogate=nogate \
      > $R/rep${rep}_shard$1.log 2>&1
    echo "shard $1 rep $rep rc=$? $(date)"
  done
}
shard 0 0 & sleep 120
shard 1 0 & sleep 120
shard 2 1 &
# shard 3 waits for the CRN smoke to release its share of GPU1
while pgrep -f "smoke_crn" > /dev/null; do sleep 60; done
echo "smoke finished $(date)"
shard 3 1 &
wait
echo "STAGE C ALL SHARDS DONE $(date)"
