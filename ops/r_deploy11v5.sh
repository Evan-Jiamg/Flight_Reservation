G=/tmp2/mzjiang_usersim/grpo_planner
cd $G && echo "$PAYLOAD_B64" | base64 -d | tar xzf - || exit 1
pgrep -u mzjiang -f "run_v11_launch4.sh|run_v11_formal4.sh|start_servers4.sh|gpu_holder.py" && echo "WARNING: v4 pieces still alive"
setsid nohup bash $G/run_v11_launch5.sh > $G/run_v11_launch5.log 2>&1 < /dev/null &
echo "launcher5 queued"
sleep 25
cat $G/run_v11_launch5.log; tail -3 $G/gpu_holder2.log
bash $G/watch.sh
