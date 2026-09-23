G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1
date; cat $G/run_stageC_v1_shards.log
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for k in 0 1 2 3; do echo "--- shard$k"; grep -E "leak gate|^\s+\[|Traceback|Error|OutOfMemory" $R/rep0_shard$k.log 2>/dev/null | cut -c1-160 | tail -2; done
