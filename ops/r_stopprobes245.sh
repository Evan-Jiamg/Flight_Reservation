G=/tmp2/mzjiang_usersim/grpo_planner
pkill -f "v3_run1_probe/run_v3_probes.sh" && echo "probe queue stopped"
for P in $(pgrep -f "planner_probe.py run"); do echo "stop probe pid $P"; kill $P; done
sleep 5
echo "old-prompt probes stopped by request: re-run with the v3 Planner architecture, no 32B re-run $(date)" >> $G/v3_run1_probes.log
wc -l $G/v3_run1/probe/*.jsonl 2>/dev/null
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do ps -o user=,pid=,args= -p $p 2>/dev/null | cut -c1-90; done
tail -1 $G/v3_run1/judge/labels_llama70b.log | cut -c1-100; wc -l $G/v3_run1/judge/labels_llama70b.jsonl
