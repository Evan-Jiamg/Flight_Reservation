G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1
date; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do ps -o user=,pid=,args= -p $p 2>/dev/null | cut -c1-110; done
tail -4 $G/run_stageC_ditto_rep1.log
for k in 0 1 2 3; do echo "rep1_ditto_shard$k $(wc -l < $R/rep1_ditto_shard$k/nogate.jsonl 2>/dev/null)"; done
ps -u mzjiang -o pid,etime,args | grep -E "rollout|resume_shard2|train_stop" | grep -v grep | cut -c1-120
