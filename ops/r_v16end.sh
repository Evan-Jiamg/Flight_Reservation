G=/tmp2/mzjiang_usersim/grpo_planner; RUN=$G/runs/pend_f2_v16
echo "=== formal log tail"; grep -vE "^U[0-9]|^VAL" $G/run_v16_formal.log | tail -12 | cut -c1-250
echo "=== launcher"; tail -4 $G/run_v16_launch.log | cut -c1-200
echo "=== guard"; tail -4 $G/run_v16_guard.log | cut -c1-200
echo "=== verify u10 FAIL lines"; grep -E "FAIL" $RUN/verify_u10.txt 2>/dev/null | head -12 | cut -c1-250
echo "=== train.log tail"; grep -v "Loading weights" $RUN/train.log | tail -5 | cut -c1-250
cat $RUN/best.json 2>/dev/null | head -12
