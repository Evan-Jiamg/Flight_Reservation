G=/tmp2/mzjiang_usersim/grpo_planner; S=$G/code_snapshots/stageC_v1_shards
kill 713434 && echo "orchestrator 713434 stopped (smoke python keeps running)"
sleep 2; ps -p 713438 -o pid,etime,cmd | cut -c1-120
ls $G/stageC_v1/ | grep -E "rep0_half" && echo "WARNING: half dirs exist"
mkdir -p $S && cd $S && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum run_stageC_v1_shards.sh > SHA256SUMS && chmod -w run_stageC_v1_shards.sh
echo "orchestrator replaced by $S/run_stageC_v1_shards.sh at $(date)" >> $G/run_stageC_v1.log
setsid nohup bash $S/run_stageC_v1_shards.sh > $G/run_stageC_v1_shards.log 2>&1 < /dev/null &
sleep 90; cat $G/run_stageC_v1_shards.log
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
