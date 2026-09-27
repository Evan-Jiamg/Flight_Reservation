G=/tmp2/mzjiang_usersim/grpo_planner
echo "=== smoke check"; cat $G/run_v16_smoke_check.txt
echo "=== prelim log (step 0 part)"; grep -nE "step 0|gpt-oss|STOP|died" $G/run_v16_prelim.log | tail -6 | cut -c1-200
echo "=== gpt-oss log tail"; grep -E "Error|error|CUDA|memory|OOM|Traceback" $G/runs/v16_prelim/gptoss.log | tail -8 | cut -c1-250
echo "=== night log"; cat $G/run_v16_night.log | cut -c1-200
