G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1; C=$G/code_snapshots/planner_audit_v1
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.py > SHA256SUMS
PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python planner_stop_audit.py \
  --glob "ditto_rep0=$R/rep0_ditto_shard*/nogate.jsonl" --glob "userlm_rep0=$R/rep0_shard*/nogate.jsonl" \
  --out $R/planner_stop_audit 2>&1 | tail -20
echo "--- ditto analysis log"; tail -3 $G/ditto_analysis_rep0.log | cut -c1-200
