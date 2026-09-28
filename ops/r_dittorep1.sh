G=/tmp2/mzjiang_usersim/grpo_planner; S=$G/code_snapshots/stageC_ditto_rep1
mkdir -p $S && cd $S && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.sh > SHA256SUMS && chmod -w *.sh
setsid nohup bash $S/run_stageC_ditto_rep1.sh > $G/run_stageC_ditto_rep1.log 2>&1 < /dev/null &
sleep 20; cat $G/run_stageC_ditto_rep1.log
