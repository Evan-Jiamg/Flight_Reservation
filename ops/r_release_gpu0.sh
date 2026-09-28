G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold
echo "before:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "our GPU processes:"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  [ "$(ps -o user= -p $p 2>/dev/null)" = "mzjiang" ] && echo "  $p $(ps -o cmd= -p $p | cut -c1-110)"
done
# placeholder: stop file + terminate (it holds 0 GiB now)
touch $H/stop; sleep 3; pkill -u mzjiang -f gpu_holder2.py
# our servers (gpt-oss on 8029, Planner vLLM on 8031) and their workers
pkill -u mzjiang -f "vllm serve" ; pkill -u mzjiang -f "vllm.entrypoints"
for i in $(seq 1 30); do
  left=$(pgrep -u mzjiang -f "vllm serve|vllm.entrypoints|gpu_holder2" | wc -l); [ $left -eq 0 ] && break; sleep 2
done
left=$(pgrep -u mzjiang -f "vllm serve|vllm.entrypoints|gpu_holder2" | wc -l)
echo "our server/holder processes left: $left"
sleep 5
echo "our GPU processes after:"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  [ "$(ps -o user= -p $p 2>/dev/null)" = "mzjiang" ] && echo "  STILL $p $(ps -o cmd= -p $p | cut -c1-110)"
done
echo "after:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
curl -s -m 3 http://127.0.0.1:8029/v1/models > /dev/null && echo "8029 still up" || echo "8029 down"
curl -s -m 3 http://127.0.0.1:8031/v1/models > /dev/null && echo "8031 still up" || echo "8031 down"
