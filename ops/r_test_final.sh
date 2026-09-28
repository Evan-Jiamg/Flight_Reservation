G=/tmp2/mzjiang_usersim/grpo_planner; RUN=$G/runs/pend_f2_v16
echo "=== launcher log"; grep -vE "^\s*[\{\}\"]|^   " $G/run_v16_test.log | tail -12 | cut -c1-200
echo "=== processes"; pgrep -u mzjiang -af "run_v16_test|eval_test_rl|gpu_holder2|vllm serve" | grep -v pgrep | cut -c1-80; echo "(none above = finished)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== test_boot.txt"; cat $RUN/test_boot.txt 2>/dev/null | cut -c1-220
echo "=== verify_test"; grep -E "PIPELINE VERIFICATION|FAIL" $RUN/verify_test.txt 2>/dev/null | head -8 | cut -c1-200
echo "=== test.log errors"; grep -cE "Traceback|Error" $RUN/test.log
