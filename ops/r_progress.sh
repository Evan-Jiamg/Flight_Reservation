G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1
date; tail -3 $G/run_stageC_v1_shards.log; cat $R/queue_shard1.out
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for r in 0 1; do for k in 0 1 2 3; do f=$R/rep${r}_shard$k/nogate.jsonl; [ -f $f ] && echo "rep$r shard$k: $(wc -l < $f) episodes"; done; done
grep -hE "Traceback|OutOfMemory|Error" $R/rep*_shard*.log 2>/dev/null | cut -c1-160 | sort | uniq -c | head
grep -hE "^\s+\[" $R/rep0_shard*.log | tail -4 | cut -c1-170
