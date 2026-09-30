G=/tmp2/mzjiang_usersim/grpo_planner
cp -p $G/start_servers5.sh $G/start_servers5.sh.bak_20260930
echo "$PAYLOAD_B64" | base64 -d > $G/start_servers5.sh.new && bash -n $G/start_servers5.sh.new && mv $G/start_servers5.sh.new $G/start_servers5.sh
sha256sum $G/start_servers5.sh $G/start_servers5.sh.bak_20260930 | cut -c1-16
grep -n "grace of 3 checks" $G/start_servers5.sh
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
