G=/tmp2/mzjiang_usersim/grpo_planner; P=$G/phase0
if pgrep -u mzjiang -f run_phase0.sh > /dev/null; then echo "STATE_RUNNING"; else
  if grep -q "PHASE0 DONE" $P/run_phase0.log 2>/dev/null; then echo "STATE_DONE"; else echo "STATE_DEAD"; fi; fi
echo "--- main log"; tail -80 $P/run_phase0.log 2>/dev/null | cut -c1-400
for f in p1.log p0.log; do [ -f $P/$f ] && { echo "--- $f (last 6)"; grep -v "it/s\]" $P/$f | tail -6 | cut -c1-300; }; done
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
