# Release every GPU allocation of OUR processes (user mzjiang) on this server -- user request 2026-09-26.
# Only our own processes are touched (pgrep -u mzjiang); nothing of other users is read or signalled.
G=/tmp2/mzjiang_usersim/grpo_planner; RUN=$G/runs/pend_f2_v10
echo "=== before $(date)"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo "=== v10 gate state before stopping"
if pgrep -u mzjiang -f run_v10_formal.sh > /dev/null; then echo "gate STILL RUNNING (will be stopped; checkpoints kept)"; else echo "gate not running"; fi
tail -40 $G/run_v10_formal.log 2>/dev/null | grep -v "Loading weights" | cut -c1-300
ls $RUN/ckpt 2>/dev/null
cp -f $G/run_v10_formal.log $RUN/run_v10_formal.log.at_release 2>/dev/null
echo "=== stopping our jobs"
for pat in run_v10_formal.sh run_v9_smoke.sh run_v8_smoke.sh train_planner_rl.py task1_v4.py rollout_v4.py test_rl_algos_server.py test_batching_server.py; do
  pids=$(pgrep -u mzjiang -f "$pat" | tr '\n' ' ')
  [ -n "$pids" ] && { echo "TERM $pat: $pids"; kill -TERM $pids 2>/dev/null; }
done
echo "=== stopping our vLLM gpt-oss-120b server (port 8029)"
pids=$(pgrep -u mzjiang -f "vllm" | tr '\n' ' ')
[ -n "$pids" ] && { echo "TERM vllm: $pids"; kill -TERM $pids 2>/dev/null; }
sleep 30
left=$( (pgrep -u mzjiang -f vllm; for pat in run_v10_formal.sh train_planner_rl.py task1_v4.py rollout_v4.py test_rl_algos_server.py test_batching_server.py; do pgrep -u mzjiang -f "$pat"; done) | sort -u | tr '\n' ' ')
[ -n "$left" ] && { echo "KILL still alive: $left"; kill -KILL $left 2>/dev/null; sleep 5; }
echo "=== our processes still holding GPU memory (should be none)"
ours=" $(pgrep -u mzjiang | tr '\n' ' ') "
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read pid mem; do
  case "$ours" in *" $pid "*) echo "OURS STILL ON GPU: $pid $mem";; esac
done
echo "=== after $(date)"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
