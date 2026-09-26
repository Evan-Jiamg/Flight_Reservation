G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/analysis_v2; A=$G/analysis_v1; E=$G/stageB_eval_v1
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.py *.md > SHA256SUMS
cp PREREG_STAGE_BC_20260924.md $G/PREREG_STAGE_BC_20260924.md; date > $G/PREREG_STAGE_BC_20260924.md.timestamp; sha256sum $G/PREREG_STAGE_BC_20260924.md >> $G/PREREG_STAGE_BC_20260924.md.timestamp
PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python; export PYTHONNOUSERSITE=1
$PY test_derive_gate_arms.py | tail -1
for ep in 1 2; do $PY stop_timing.py $G/stageA_eval_v1/prism_val_ep${ep}.jsonl --bootstrap 1000 > $A/stageA_prism_val_ep${ep}_timing_v2.json; done
for ep in 0 1 2; do $PY stop_timing.py $E/fold0_ep${ep}_innerval.jsonl $E/fold1_ep${ep}_innerval.jsonl $E/fold2_ep${ep}_innerval.jsonl --bootstrap 1000 > $A/stageB_pooled_ep${ep}_timing_v2.json; done
$PY - <<'EOF'
import json
A="/tmp2/mzjiang_usersim/grpo_planner/analysis_v1"
for name in ["stageA_prism_val_ep1","stageA_prism_val_ep2","stageB_pooled_ep0","stageB_pooled_ep1","stageB_pooled_ep2"]:
    d=json.load(open(f"{A}/{name}_timing_v2.json")); h=d["hazard"]
    print(f"{name:22s} S={d['sessions']:3d} hazard early {h['point']['early']:.3f} {[round(x,3) for x in h['session_bootstrap_95']['early']]} exact {h['point']['exact']:.3f} {[round(x,3) for x in h['session_bootstrap_95']['exact']]} censored {h['point']['not_by_k1_censored']:.3f}")
EOF
echo "--- corpus structure"
$PY - <<'EOF'
import json
r=json.loads(open("/home/mzjiang/v5-latency/data.jsonl").readline())
print(sorted(r.keys()))
for k,v in r.items():
    if isinstance(v,list) and v and isinstance(v[0],dict): print(k, len(v), sorted(v[0].keys())[:8])
EOF
echo "--- smoke"; grep -E "leak|META|\[1\]|\[2\]|Traceback|Error" $G/stageC_v1/smoke_crn.log | cut -c1-200 | tail -5
tail -2 $G/run_stageC_v1.log
