G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1
date
echo "--- orchestrator"; cat $G/run_stageC_v1.log | grep -v Warning | tail -4
echo "--- smoke"; grep -E "^\s+\[|Traceback|Error|DONE" $R/smoke_crn.log | cut -c1-220 | tail -6
for h in rep0_half_a rep0_half_b; do echo "--- $h"; grep -E "^\s+\[|Traceback|Error|DONE" $R/$h.log 2>/dev/null | cut -c1-200 | tail -3; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
