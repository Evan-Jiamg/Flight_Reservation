G=/tmp2/mzjiang_usersim/grpo_planner
du -sh $G/prism_pretrain/*.jsonl
cd $G && tar czf /tmp2/mzjiang_usersim/xfer_prism.tgz prism_pretrain && ls -la /tmp2/mzjiang_usersim/xfer_prism.tgz && sha256sum /tmp2/mzjiang_usersim/xfer_prism.tgz
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
cat $G/run_stageC_queue2.log
for f in $G/stageC_v1/rep0_ditto_shard*/nogate.jsonl; do echo "$f $(wc -l < $f)"; done
grep -hE "Traceback|Error" $G/stageC_v1/rep0_ditto_shard*.log | head -3
head -c 400 $G/prism_pretrain/validation.jsonl | tr '\n' ' '; echo
