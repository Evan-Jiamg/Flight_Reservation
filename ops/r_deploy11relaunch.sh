G=/tmp2/mzjiang_usersim/grpo_planner
if pgrep -u mzjiang -f run_v11_launch.sh > /dev/null; then echo "launcher already queued"; exit 1; fi
cd $G && echo "$PAYLOAD_B64" | base64 -d | tar xzf - || exit 1
ls -la $G/start_servers2.sh $G/run_v11_formal2.sh $G/run_v11_continue2.sh $G/run_v11_launch.sh $G/watch.sh | awk '{print $NF}'
setsid nohup bash $G/run_v11_launch.sh > $G/run_v11_launch.log 2>&1 < /dev/null &
echo "launcher queued"
bash $G/watch.sh
