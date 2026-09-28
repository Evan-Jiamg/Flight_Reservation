echo "pid ppid sid tty stat etime cmd  (tty '?' = no terminal; own sid = survives ssh/VPN drops)"
for pat in run_v16_test gpu_holder2 eval_test_rl "vllm serve"; do
  for p in $(pgrep -u mzjiang -f "$pat"); do ps -o pid=,ppid=,sid=,tty=,stat=,etime=,args= -p $p | cut -c1-110; done
done
echo "=== log tail"; tail -3 /tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/test.log | cut -c1-160
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
