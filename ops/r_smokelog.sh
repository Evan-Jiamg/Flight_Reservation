G=/tmp2/mzjiang_usersim/grpo_planner
grep -v -i -E "warn|Loading weights|it/s\]" $G/smoke_v3.log | tail -25 | cut -c1-400
