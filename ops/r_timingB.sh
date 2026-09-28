G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/stageC_v1; E=$G/stageB_eval_v1; A=$G/analysis_v1
PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python; export PYTHONNOUSERSITE=1
cd $C
for ep in 0 1 2; do
  for f in 0 1 2; do $PY stop_timing.py $E/fold${f}_ep${ep}_innerval.jsonl > $A/stageB_fold${f}_ep${ep}_timing.json; done
  $PY stop_timing.py $E/fold0_ep${ep}_innerval.jsonl $E/fold1_ep${ep}_innerval.jsonl $E/fold2_ep${ep}_innerval.jsonl > $A/stageB_pooled_ep${ep}_timing.json
done
$PY - <<'EOF'
import json
A="/tmp2/mzjiang_usersim/grpo_planner/analysis_v1"
for ep in (0,1,2):
    for f in ("fold0","fold1","fold2","pooled"):
        d=json.load(open(f"{A}/stageB_{f}_ep{ep}_timing.json"))
        line=f"ep{ep} {f:6s} S={d['sessions']:2d} cont={d['n_cont']:2d} AUC={d['auc']:.3f} NLL={d['nll']:.4f} |"
        for t in ("0.1","0.2","0.3","0.5"):
            p=d["by_threshold"][t]["point"]
            line+=f" t{t}: FS {p['false_stop']:.2f} K1 {p['k1_end']:.2f} E/X/C {p['early']:.2f}/{p['exact']:.2f}/{p['not_by_k1_censored']:.2f} |"
        print(line)
# p_stop distributions on inner val (pooled ep2)
import statistics
rows=[json.loads(l) for f in (0,1,2) for l in open(f"/tmp2/mzjiang_usersim/grpo_planner/stageB_eval_v1/fold{f}_ep2_innerval.jsonl")]
for lab in (False,True):
    ps=sorted(r["p_stop"] for r in rows if r["target_stop"]==lab)
    print("ep2 target",lab,"n",len(ps),"p_stop min/med/max %.3f/%.3f/%.3f"%(ps[0],statistics.median(ps),ps[-1]))
EOF
echo; tail -2 $G/stageC_v1/smoke_crn.log | cut -c1-250
