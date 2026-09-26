G=/tmp2/mzjiang_usersim/grpo_planner
pkill -f "dev_v3/run_smoke_v3.sh" && echo "smoke orchestrator stopped"
for P in $(pgrep -f "dev_v3.*rollout_ditto_v3.py|rollout_ditto_v3.py.*dev_v3"); do echo "stop pid $P"; kill $P; done
sleep 5
ps -u mzjiang -o pid,args | grep -E "rollout_ditto_v3|run_smoke" | grep -v grep | cut -c1-100
echo "smoke stopped by request (architecture under revision; outputs not used) $(date)" >> $G/smoke_v3.log
grep -v -i -E "warn|Loading weights|it/s\]" $G/smoke_v3.log | grep -E "^\s+\[|==|leak|DONE|Traceback|Error" | tail -8 | cut -c1-200
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
