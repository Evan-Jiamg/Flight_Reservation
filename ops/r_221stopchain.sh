pkill -f run_stageA_cox_221.sh && echo "221 launcher stopped (cox python keeps running; no-offset moved to 244)"
echo "launcher stopped $(date); no-offset control moved to host 244" >> /tmp2/mzjiang_usersim/grpo_planner/run_stageA_cox_221.log
ps -u mzjiang -o pid,etime,cmd | grep train_stop_sft_cox | grep -v grep | cut -c1-120
grep -E "optimizer_step|^epoch" /tmp2/mzjiang_usersim/grpo_planner/stageA_cox_v1.log | tail -2 | cut -c1-120
