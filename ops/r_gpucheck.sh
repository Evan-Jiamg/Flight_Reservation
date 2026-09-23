G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1
date
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
echo "--- my python procs"; ps -u mzjiang -o pid,etime,cmd | grep -E "rollout_stop_sft|run_stageC" | grep -v grep | cut -c1-200
echo "--- orchestrator"; grep -v Warning $G/run_stageC_v1.log | tail -3
echo "--- smoke"; grep -E "^\s+\[|Traceback|Error" $R/smoke_crn.log | cut -c1-200 | tail -4; ls $R/smoke_crn 2>/dev/null
for h in rep0_half_a rep0_half_b; do echo "--- $h"; grep -E "^\s+\[|Traceback|Error" $R/$h.log 2>/dev/null | cut -c1-200 | tail -3; done
free -g | head -2
