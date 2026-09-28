G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1
date; cat $G/run_stageC_queue2.log
for f in $R/rep0_ditto_shard*/nogate.jsonl $R/rep1_shard*/nogate.jsonl; do [ -f $f ] && echo "$(basename $(dirname $f)): $(wc -l < $f)"; done
grep -hE "Traceback|OutOfMemory|Error" $R/rep0_ditto_shard*.log | sort | uniq -c | head -3
grep -hE "^\s+\[" $R/rep0_ditto_shard*.log | tail -4 | cut -c1-160
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
