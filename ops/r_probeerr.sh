O=/tmp2/mzjiang_usersim/grpo_planner/v3_run1
grep -v -i -E "warn|Loading" $O/probe/qwen25_32b.log | tail -12 | cut -c1-300
tail -3 $O/judge/labels_llama70b.log | cut -c1-200
