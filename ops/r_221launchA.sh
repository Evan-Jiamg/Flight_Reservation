M=/tmp2/mzjiang_usersim; C=$M/code_snapshots/stageA_cox_v1
cd $M && sha256sum xfer_prism.tgz && tar xzf xfer_prism.tgz -C grpo_planner && wc -l grpo_planner/prism_pretrain/*.jsonl
[ -d $C ] && { echo "snapshot exists"; exit 1; }
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/consistent/bin/python test_stop_prompt_v2.py
sha256sum *.py *.sh > SHA256SUMS; chmod -w *.py *.sh
setsid nohup bash $C/run_stageA_cox_221.sh > $M/grpo_planner/run_stageA_cox_221.log 2>&1 < /dev/null &
sleep 100; tail -5 $M/grpo_planner/stageA_orig_ep2_on221.log | cut -c1-300; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
