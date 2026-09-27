G=/tmp2/mzjiang_usersim/grpo_planner; OUT=$G/runs/v16_prelim
grep -v "^\s*$" $OUT/step0.log | tail -25 | cut -c1-220
ls -la $OUT/step0_f2.jsonl* 2>&1 | cut -c1-150
wc -l $OUT/step0_f2.jsonl 2>&1
