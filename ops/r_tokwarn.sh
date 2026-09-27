RUN=/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11
echo "train.log: $(grep -c 'longer than the specified maximum sequence length' $RUN/train.log)"
grep -n -B2 'longer than the specified maximum sequence length' $RUN/train.log | head -8 | cut -c1-200
echo "reselect.log: $(grep -c 'longer than the specified maximum sequence length' $RUN/reselect.log)"
