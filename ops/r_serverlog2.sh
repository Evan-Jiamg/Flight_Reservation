G=/tmp2/mzjiang_usersim/grpo_planner
grep -nE "^R=|^Q=| R=| Q=" $G/start_servers5.sh | head
R=$(grep -oE "R=[^ ;]+" $G/start_servers5.sh | head -1 | cut -d= -f2)
R=$(eval echo $R)
echo "R=$R"; ls -la $R | head
echo "=== serve.log tail"; tail -60 $R/serve.log | grep -vE "^\s*$" | cut -c1-260 | tail -45
echo "=== serve.sh"; cat $R/serve.sh | cut -c1-250
