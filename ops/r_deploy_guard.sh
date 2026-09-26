G=/tmp2/mzjiang_usersim/grpo_planner
cd $G && echo "$PAYLOAD_B64" | base64 -d | tar xzf - || exit 1
pgrep -u mzjiang -f run_v11_guard.sh > /dev/null && { echo "guard already running"; exit 0; }
setsid nohup bash $G/run_v11_guard.sh > $G/run_v11_guard.log 2>&1 < /dev/null &
echo "guard armed"
bash $G/watch.sh
