R=/tmp2/mzjiang_usersim/grpo_planner/stageC_v1
P=$(pgrep -f "replicate 1 --out-dir $R/rep1_shard3"); echo "shard3 rep1 pid: $P"; [ -n "$P" ] && kill $P
sleep 3
for f in rep0_shard0 rep0_shard1 rep0_shard2 rep0_shard3 rep1_shard0 rep1_shard3 rep0_ditto_shard0; do echo "$f: $(wc -l < $R/$f/nogate.jsonl 2>/dev/null)"; done
ps -u mzjiang -o pid,etime,cmd | grep rollout_stop_sft | grep -v grep | sed 's/.*--gpu/--gpu/' | cut -c1-140
grep -E "leak|Traceback|Error" $R/rep0_ditto_shard0.log | cut -c1-160 | tail -3
cat /tmp2/mzjiang_usersim/grpo_planner/run_stageC_queue.log
