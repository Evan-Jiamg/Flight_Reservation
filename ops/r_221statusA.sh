G=/tmp2/mzjiang_usersim/grpo_planner
date; cat $G/run_stageA_cox_221.log
for r in stageA_orig_ep2_on221 stageA_cox_v1 stageA_nooffset_221_v1; do
  echo "--- $r"; grep -E "^epoch|optimizer_step|complete|Traceback|Error" $G/$r.log 2>/dev/null | tail -3 | cut -c1-400
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
