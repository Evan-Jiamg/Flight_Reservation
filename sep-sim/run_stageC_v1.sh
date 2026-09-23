#!/bin/bash
# Stage C v1: corrected-runner no-gate Task 2 with gate-prompt logging on the union of
# inner-train/inner-validation scenarios (replicate 0), preceded by an online CRN smoke
# that checks the truncation counterfactual against a real gated run.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner
C=$G/code_snapshots/stageC_v1          # frozen code; never edited while runs are live
R=$G/stageC_v1
PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /home/mzjiang/.secrets/openai.env; set +a
mkdir -p $R
cd $C
# ---- union scenario list (IDs only), then split into two halves --------------------
$PY - <<'EOF'
import json, hashlib
G = "/tmp2/mzjiang_usersim/grpo_planner"; R = G + "/stageC_v1"
seen, sc = set(), []
for side, d in (("inner_train", "task2_train_scenarios"), ("inner_validation", "task2_val_scenarios")):
    for f in (0, 1, 2):
        for s in json.load(open(f"{G}/{d}/fold{f}_{side}_scenarios.json"))["scenarios"]:
            if s["conversation_id"] not in seen:
                seen.add(s["conversation_id"]); sc.append(dict(s, order=len(sc)))
for name, part in (("union", sc), ("half_a", sc[0::2]), ("half_b", sc[1::2])):
    json.dump({"t_max": 10, "seeds": [0, 1], "scenarios": [dict(s, order=i) for i, s in enumerate(part)]},
              open(f"{R}/scenarios_{name}.json", "w"), indent=1)
print("union scenarios", len(sc), "halves", len(sc[0::2]), len(sc[1::2]))
EOF
# ---- smoke: fold1 inner-train first scenario, seed 0, nogate vs gate under CRN --------
S=$R/smoke_crn
if [ ! -f $S/DONE ]; then
  $PY rollout_stop_sft.py --scenarios $G/task2_train_scenarios/fold1_inner_train_scenarios.json \
    --fold 1 --side inner_train --limit 1 --gpu 1 --r0-crn --log-prompts --out-dir $S \
    --arm nogate=nogate --arm f1ep2_t02=$G/trec_inner_fold1_7b_v1/epoch2@0.2 > $S.log 2>&1 && touch $S/DONE
fi
echo "smoke rc=$? $(date)"
# ---- main no-gate logging runs, replicate 0 ----------------------------------------
( $PY rollout_stop_sft.py --scenarios $R/scenarios_half_a.json --fold -1 --side inner_union \
    --gpu 1 --log-prompts --replicate 0 --out-dir $R/rep0_half_a --arm nogate=nogate \
    > $R/rep0_half_a.log 2>&1; echo "half_a rc=$? $(date)" ) &
# GPU0 waits for Stage B to release it
until grep -q "ALL STAGE B DONE" $G/run_stageB_v1.log 2>/dev/null; do sleep 60; done
( $PY rollout_stop_sft.py --scenarios $R/scenarios_half_b.json --fold -1 --side inner_union \
    --gpu 0 --log-prompts --replicate 0 --out-dir $R/rep0_half_b --arm nogate=nogate \
    > $R/rep0_half_b.log 2>&1; echo "half_b rc=$? $(date)" ) &
wait
echo "STAGE C REP0 DONE $(date)"
