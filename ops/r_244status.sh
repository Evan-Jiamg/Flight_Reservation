G=/tmp2/mzjiang_usersim/grpo_planner
date; cat $G/run_stageA_nooffset_244.log
grep -E "^epoch|optimizer_step|complete|Traceback|Error|OutOfMemory" $G/stageA_nooffset_244_v1.log 2>/dev/null | tail -3 | cut -c1-300
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
