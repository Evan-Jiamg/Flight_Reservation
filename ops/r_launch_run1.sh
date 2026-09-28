G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/v3_run1
[ -d $C ] && { echo "snapshot exists, refusing"; exit 1; }
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.py *.sh > SHA256SUMS && chmod -w *.py *.sh
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
setsid nohup bash $C/run_v3_parallel1.sh > $G/v3_run1.log 2>&1 < /dev/null &
sleep 240; cat $G/v3_run1.log; tail -2 $G/v3_run1/judge/labels_llama70b.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
