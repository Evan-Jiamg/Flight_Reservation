hostname; date
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | while IFS=, read u p m; do
  idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$(echo $u | xargs)" | cut -d, -f1)
  echo "GPU$idx pid$p $m $(ps -o user=,args= -p $p 2>/dev/null | sed 's/.*scenarios_\(shard[0-9]\).*replicate \([0-9]\).*--speaker \([a-z]*\).*/\1 rep\2 \3/' | cut -c1-80)"
done
