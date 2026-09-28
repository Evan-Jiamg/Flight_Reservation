G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/v3_run2
[ -d $C ] && { echo "snapshot exists, refusing"; exit 1; }
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.py *.sh > SHA256SUMS && chmod -w *.py *.sh
setsid nohup bash $C/run_v3_run2.sh > $G/v3_run2.log 2>&1 < /dev/null &
sleep 60; cat $G/v3_run2.log | grep -v -i warn | tail -5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
