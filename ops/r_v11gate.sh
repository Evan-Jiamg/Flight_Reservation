G=/tmp2/mzjiang_usersim/grpo_planner
sed -n '/=== 1 probe/,/V11 GATE DONE/p' $G/run_v11_formal5.log | grep -v "Loading weights" | cut -c1-330 | head -60
echo "--- continuation"
tail -5 $G/run_v11_continue5.log | cut -c1-200
