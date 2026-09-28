G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/v3_run1_probe
[ -d $C ] && { echo "snapshot exists, refusing"; exit 1; }
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.py *.sh > SHA256SUMS && chmod -w *.py *.sh
setsid nohup bash $C/run_v3_probes.sh > $G/v3_run1_probes.log 2>&1 < /dev/null &
sleep 150; cat $G/v3_run1_probes.log; grep -v -i -E "warn|Loading" $G/v3_run1/probe/qwen25_32b_nf4.log | tail -3 | cut -c1-200
wc -l $G/v3_run1/probe/qwen25_32b_nf4.jsonl 2>/dev/null; tail -1 $G/v3_run1/judge/labels_llama70b.log | cut -c1-120
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
