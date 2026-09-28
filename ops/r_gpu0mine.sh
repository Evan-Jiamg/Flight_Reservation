tot=0; mine=0
while IFS=', ' read -r pid mem; do
  [ -z "$pid" ] && continue
  tot=$((tot + mem))
  [ "$(ps -o user= -p $pid 2>/dev/null)" = "mzjiang" ] && mine=$((mine + mem))
done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i 0)
echo "GPU0 compute memory: ours ${mine} MiB, others $((tot - mine)) MiB"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
tail -2 /tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/train.log | cut -c1-200
