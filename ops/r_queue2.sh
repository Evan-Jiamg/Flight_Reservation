G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1; S=$G/code_snapshots/stageC_queue2
cat $G/run_stageC_queue.log
pkill -f "code_snapshots/stageC_v2/run_stageC_queue.sh" && echo "old queue stopped"
# stop any UserLM rep1 rollout it may have started (resumable; queue2 restarts it later)
for P in $(pgrep -f "replicate 1 --out-dir $R/rep1_shard"); do echo "stop rep1 pid $P"; kill $P; done
sleep 3
for f in $R/rep1_shard*/nogate.jsonl; do echo "$f $(wc -l < $f)"; done
ls $R/rep0_ditto_shard*/nogate.jsonl 2>/dev/null && echo "WARNING ditto output exists"
mkdir -p $S && cd $S && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.sh > SHA256SUMS && chmod -w *.sh
setsid nohup bash $S/run_stageC_queue2.sh > $G/run_stageC_queue2.log 2>&1 < /dev/null &
sleep 100; cat $G/run_stageC_queue2.log
grep -v -E "Warning|warn|pynvml|Loading" $R/rep0_ditto_shard0.log | tail -4 | cut -c1-250
ps -u mzjiang -o pid,etime,cmd | grep -E "rollout_stop_sft|stageC_analysis" | grep -v grep | sed 's/.*--out-dir//' | cut -c1-120
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
