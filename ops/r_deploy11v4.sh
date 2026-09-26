G=/tmp2/mzjiang_usersim/grpo_planner
cd $G && echo "$PAYLOAD_B64" | base64 -d | tar xzf - || exit 1
echo "--- stopping OUR waiting v3 chain (nothing started yet)"
for pat in run_v11_launch3.sh run_v11_formal3.sh run_v11_continue3.sh start_servers3.sh; do
  p=$(pgrep -u mzjiang -f "$pat" | tr '
' ' '); [ -n "$p" ] && { echo "TERM $pat: $p"; kill -TERM $p; }
done
sleep 3
pgrep -u mzjiang -f "vllm serve" > /dev/null && echo "WARNING: a vllm serve of ours is running" || echo "no vllm server of ours"
setsid nohup bash $G/run_v11_launch4.sh > $G/run_v11_launch4.log 2>&1 < /dev/null &
echo "launcher4 queued"
sleep 45
cat $G/run_v11_launch4.log; tail -3 $G/gpu_holder.log
bash $G/watch.sh
