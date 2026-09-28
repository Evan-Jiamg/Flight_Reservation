G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1
date
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do ps -o pid,user,etime,args -p $p --no-headers 2>/dev/null | cut -c1-200; done
echo "--- ditto rep1 progress"
cat $G/run_stageC_ditto_rep1.log
for k in 0 1 2 3; do f=$R/rep1_ditto_shard$k/nogate.jsonl; [ -f $f ] && echo "shard$k: $(wc -l < $f) episodes (shard size: $(grep -c conversation_id $R/scenarios_shard$k.json) scenarios x2)"; done
grep -hE "^\s+\[" $R/rep1_ditto_shard1.log | tail -2 | cut -c1-150
echo "--- ditto analysis (scoring on GPU1)"
ls -la $R/ditto_rep0/ 2>/dev/null; wc -l $R/ditto_rep0/scores.jsonl 2>/dev/null
