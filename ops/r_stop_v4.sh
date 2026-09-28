G=/tmp2/mzjiang_usersim/grpo_planner
touch $G/hold/stop
for pat in run_v11_launch4.sh run_v11_formal4.sh run_v11_continue4.sh start_servers4.sh; do
  p=$(pgrep -u mzjiang -f "$pat" | tr '\n' ' '); [ -n "$p" ] && { echo "TERM $pat: $p"; kill -TERM $p; }
done
for i in $(seq 1 15); do pgrep -u mzjiang -f gpu_holder.py > /dev/null || break; sleep 1; done
p=$(pgrep -u mzjiang -f gpu_holder.py | tr '\n' ' '); [ -n "$p" ] && { echo "holder still up, TERM $p"; kill -TERM $p; sleep 3; }
tail -2 $G/gpu_holder.log
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo "--- gpt-oss style endpoints answering on this host (API only, read-only)"
for port in 8000 8001 8002 8010 8019 8020 8029 8080 8888; do
  r=$(curl -s -m 3 http://127.0.0.1:$port/v1/models 2>/dev/null)
  [ -n "$r" ] && echo "port $port: $(echo "$r" | python3 -c 'import json,sys; d=json.load(sys.stdin); print([(m.get("id"), m.get("max_model_len")) for m in d.get("data", [])])' 2>/dev/null)"
done
