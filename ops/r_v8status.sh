G=/tmp2/mzjiang_usersim/grpo_planner; O=$G/v8_smoke
if pgrep -u mzjiang -f run_v8_smoke.sh > /dev/null; then echo "STATE_RUNNING"; else
  if grep -q "V8 SMOKE DONE" $G/run_v8_smoke.log 2>/dev/null; then echo "STATE_DONE"; else echo "STATE_DEAD"; fi; fi
echo "--- main log (last 60 lines)"; tail -60 $G/run_v8_smoke.log 2>/dev/null | cut -c1-400
for f in t1.log t2.log rl.log; do [ -f $O/$f ] && { echo "--- $f (last 8)"; tail -8 $O/$f | cut -c1-300; }; done
for f in t1.jsonl.errors.jsonl t2_smoke/pend.jsonl.errors.jsonl; do [ -f $O/$f ] && { echo "--- ERRORS $f"; tail -3 $O/$f | cut -c1-1500; }; done
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
