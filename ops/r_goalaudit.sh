G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1; C=$G/code_snapshots/planner_audit_v2
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
cd /home/mzjiang/Sep-Simulator && PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python $C/planner_goal_audit.py "$R/rep0_ditto_shard*/nogate.jsonl" 2>&1 | grep -v -i warn | tail -25
echo "--- ditto analysis"; tail -4 $G/ditto_analysis_rep0.log | cut -c1-200
