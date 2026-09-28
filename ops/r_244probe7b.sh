M=/tmp2/mzjiang_usersim; C=$M/code_snapshots/v3_run1_probe_244b; O=$M/v3_run1/probe
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.py > SHA256SUMS
Q7=$M/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
rm -f $O/qwen25_7b_fp16_v100.jsonl
cd $C && PYTHONNOUSERSITE=1 HF_HOME=$M/hf setsid nohup /home/mzjiang/miniconda3/envs/consistent/bin/python planner_probe.py run \
  --probe $O/probe_set.jsonl --model $Q7 --dtype float16 --gpu 2 --out $O/qwen25_7b_fp16_v100.jsonl > $O/qwen25_7b.log 2>&1 < /dev/null &
sleep 75; grep -E "Traceback|Error" $O/qwen25_7b.log | tail -3; wc -l $O/qwen25_7b_fp16_v100.jsonl 2>/dev/null
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
