G=/tmp2/mzjiang_usersim/grpo_planner; S=$G/code_snapshots/stageC_queue3
mkdir -p $S && cd $S && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.sh > SHA256SUMS && chmod -w *.sh
setsid nohup bash $S/run_stageC_queue3.sh > $G/run_stageC_queue3.log 2>&1 < /dev/null &
sleep 30; cat $G/run_stageC_queue3.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
