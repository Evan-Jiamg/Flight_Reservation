G=/tmp2/mzjiang_usersim/grpo_planner
echo "=== servers_check_smoke_v17.log"; cat $G/servers_check_smoke_v17.log | tail -20
echo "=== start_servers5.sh (log paths)"; grep -nE "log|LOG|serve.sh|gpu-memory|8029" $G/start_servers5.sh | head -20
for f in $(ls -t $G/*oss*.log $G/logs/*oss*.log $G/vllm*.log 2>/dev/null | head -3); do echo "=== $f"; tail -40 $f | cut -c1-250; done
echo "=== GPU users"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do echo "pid $p user $(ps -o user= -p $p 2>/dev/null)"; done
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
