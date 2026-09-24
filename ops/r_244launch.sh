M=/tmp2/mzjiang_usersim; C=$M/code_snapshots/stageA_cox_v1
cd $M && sha256sum xfer_prism.tgz && tar xzf xfer_prism.tgz -C grpo_planner && wc -l grpo_planner/prism_pretrain/*.jsonl
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.py *.sh > SHA256SUMS
setsid nohup bash $C/run_stageA_nooffset_244.sh > $M/grpo_planner/run_stageA_nooffset_244.log 2>&1 < /dev/null &
sleep 5; tail -c 200 $M/dl_qwen7b.log; echo; cat $M/grpo_planner/run_stageA_nooffset_244.log
