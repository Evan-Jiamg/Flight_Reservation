O=/tmp2/mzjiang_usersim/v3_run1/probe
ps -u mzjiang -o pid,etime,args | grep planner_probe | grep -v grep | cut -c1-100
wc -l $O/qwen25_7b_fp16_v100.jsonl 2>/dev/null; grep -E "Traceback|Error" $O/qwen25_7b.log | tail -3
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
