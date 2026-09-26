G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/stage_c_code
cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python
export PYTHONNOUSERSITE=1
[ -d $G/task2_val_scenarios ] || $PY make_task2_val_scenarios.py --nested-manifest $G/nested/nested_manifest.json \
  --corpus /home/mzjiang/v5-latency/data.jsonl --shards /tmp2/hchsu/trec2026-usersim-benchmark/data/req_shards_v1.json \
  --out $G/task2_val_scenarios | grep -E '"fold"|scenarios"|sessions'
cat $G/task2_train_scenarios/audit.json | grep -E '"fold"|"scenarios"'
mkdir -p $G/analysis_v1
for ep in 1 2; do
  $PY stop_timing.py $G/stageA_eval_v1/prism_val_ep${ep}.jsonl > $G/analysis_v1/stageA_prism_val_ep${ep}_timing.json
  $PY - <<EOF
import json; d=json.load(open("$G/analysis_v1/stageA_prism_val_ep${ep}_timing.json"))
print("ep${ep}", "sessions",d["sessions"],"n_cont",d["n_cont"],"AUC %.4f NLL %.5f"%(d["auc"],d["nll"]))
for t in ("0.3","0.5","0.7","0.9"):
    p=d["by_threshold"][t]["point"]; c=d["by_threshold"][t]["session_bootstrap_95"]
    print(" thr",t,"FS %.3f K1 %.3f | early %.3f exact %.3f censored %.3f lead %s | early95 %s"%(p["false_stop"],p["k1_end"],p["early"],p["exact"],p["not_by_k1_censored"],p["mean_turns_early"] and round(p["mean_turns_early"],2),[round(x,3) for x in c["early"]]))
EOF
done
for f in 0 1 2; do echo "fold$f:"; grep -E "^epoch [0-9] (validation|train_nll)|complete|DONE" $G/trec_inner_fold${f}_7b_v1.log | cut -c1-200; done
