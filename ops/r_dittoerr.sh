G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1
grep -v -E "Warning|warn|pynvml|Loading checkpoint" $R/rep0_ditto_shard0.log | tail -15 | cut -c1-300
echo "--- shard1"; grep -v -E "Warning|warn|pynvml|Loading checkpoint" $R/rep0_ditto_shard1.log | tail -4 | cut -c1-300
cat $G/run_stageC_queue.log
ps -u mzjiang -o pid,etime,cmd | grep -E "rollout_stop_sft|stageC_analysis|run_stageC_queue" | grep -v grep | cut -c1-150
echo "--- analysis log"; tail -5 $G/stageC_analysis_rep0.log | cut -c1-300
