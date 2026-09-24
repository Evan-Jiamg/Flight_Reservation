G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1
pkill -f "stageC_queue2/run_stageC_queue2.sh" && echo "queue2 stopped"
pkill -f "stageC_queue3/run_stageC_queue3.sh" && echo "queue3 stopped"
for P in $(pgrep -f "rollout_stop_sft.py.*(rep1_shard|userlm_ct_shard)"); do echo "stop rollout pid $P"; kill $P; done
sleep 5
echo "userlm runs stopped by user request (focus on Ditto) $(date)" | tee -a $G/run_stageC_queue2.log >> $G/run_stageC_queue3.log
for f in $R/rep1_shard*/nogate.jsonl $R/rep0_userlm_ct_shard*/nogate.jsonl; do [ -f $f ] && echo "$(basename $(dirname $f)): $(wc -l < $f)"; done
ps -u mzjiang -o pid,cmd | grep rollout_stop_sft | grep -v grep | wc -l
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
B=/tmp2/hchsu/trec2026-usersim-benchmark
echo "--- termination probe (ditto)"; head -c 1500 $B/instruments/termination_probe_v1/ditto_8b.json; echo
ls $B/instruments/termination_probe_v1/
grep -n -i "ditto" $B/protocols/stop_judgement.md | head -10
