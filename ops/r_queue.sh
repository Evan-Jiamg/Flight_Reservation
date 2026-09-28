G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1; C2=$G/code_snapshots/stageC_v2
ls -d /tmp2/mzjiang_usersim/models/Ditto-8B && ls /tmp2/mzjiang_usersim/models/Ditto-8B | head -5
grep -c "class DittoSpeaker" /home/mzjiang/Sep-Simulator/sepsim/models.py
[ -d $C2 ] && { echo "stageC_v2 exists, refusing"; exit 1; }
mkdir -p $C2 && cd $C2 && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
cmp stop_prompt.py $G/code_snapshots/stageC_v1/stop_prompt.py && cmp task2_episode.py $G/code_snapshots/stageC_v1/task2_episode.py && echo "shared modules identical to stageC_v1"
diff <(grep -v -E "speaker|ditto|Ditto" $G/code_snapshots/stageC_v1/rollout_stop_sft.py) <(grep -v -E "speaker|ditto|Ditto" rollout_stop_sft.py) | head -20
sha256sum *.py *.sh > SHA256SUMS; chmod -w *.py *.sh
echo "--- stopping schedulers (not the rep0 rollouts)"
ps -u mzjiang -o pid,etime,cmd | grep -E "run_stageC_v1_shards|queue_shard1|rollout_stop_sft" | grep -v grep | cut -c1-170
pkill -f run_stageC_v1_shards.sh; pkill -f queue_shard1.sh
P=$(pgrep -f "replicate 1 --out-dir $R/rep1_shard0"); echo "shard0 rep1 pid: $P"; [ -n "$P" ] && kill $P
sleep 3; wc -l $R/rep1_shard0/nogate.jsonl 2>/dev/null
echo "scheduler replaced by $C2/run_stageC_queue.sh $(date)" >> $G/run_stageC_v1_shards.log
setsid nohup bash $C2/run_stageC_queue.sh > $G/run_stageC_queue.log 2>&1 < /dev/null &
sleep 20; cat $G/run_stageC_queue.log
ps -u mzjiang -o pid,etime,cmd | grep rollout_stop_sft | grep -v grep | sed 's/.*--scenarios //' | cut -c1-150
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
