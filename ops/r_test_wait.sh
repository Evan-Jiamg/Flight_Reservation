G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold
echo "=== launcher"; tail -5 $G/run_v16_test.log | cut -c1-200
echo "=== servers_check_test.log"; tail -6 $G/servers_check_test.log 2>/dev/null | cut -c1-200
echo "=== holder"; for f in role_server role_train status_0 status_1 target_0 target_1; do printf "%s=%s " $f "$(cat $H/$f 2>/dev/null)"; done; echo
tail -4 $G/gpu_holder2.log | cut -c1-200
echo "=== GPU processes (owner only, no args)"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do echo "$p $(ps -o user= -p $p 2>/dev/null)"; done
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | cut -c1-80
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
pgrep -u mzjiang -af "vllm serve|eval_test_rl|start_servers5" | grep -v pgrep | cut -c1-100
