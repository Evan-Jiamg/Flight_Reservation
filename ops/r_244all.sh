G=/tmp2/mzjiang_usersim/grpo_planner
date
echo "--- probe extraction"; tail -2 $G/prism_probe_v1.log | cut -c1-150
echo "--- probe results"; cat $G/prism_probe_v1_probe.log 2>/dev/null | cut -c1-300
echo "--- no-offset control"; grep -E "^epoch [0-9] train|complete|Traceback" $G/stageA_nooffset_244_v1.log | tail -2 | cut -c1-420
grep -E "optimizer_step" $G/stageA_nooffset_244_v1.log | tail -1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
