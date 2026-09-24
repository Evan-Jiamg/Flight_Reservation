G=/tmp2/mzjiang_usersim/grpo_planner
grep -v -E "Warning|warn|pynvml|Loading" $G/stageC_analysis_rep0.log | tail -60 | cut -c1-260
