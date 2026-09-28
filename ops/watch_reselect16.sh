G=/tmp2/mzjiang_usersim/grpo_planner; RUN=$G/runs/pend_f2_v16
st=""; for s in run_v16_reselect gpu_holder2 train_planner_rl; do pgrep -u mzjiang -f "$s" > /dev/null && st="$st $s"; done
stage=$(grep -E "^(===|WAITING_GPU|reselect rc|verify|STOP|V16 RESELECT|RESELECT)" $G/run_v16_reselect.log 2>/dev/null | tail -1 | cut -c1-100)
gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '{printf "g%s=%dG ", $1, $2/1024}')
prog=""
if [ -f $RUN/reselect.jsonl ]; then
  prog=$(python3 -c "
import json
rows=[json.loads(l) for l in open('$RUN/reselect.jsonl') if l.strip()]
print(' '.join('u%d:%d/32%s' % (u, sum(1 for r in rows if r.get('kind')=='episode' and r['update']==u), '(done)' if any(r.get('kind')=='summary' and r['update']==u for r in rows) else '') for u in (0,5,10)))" 2>/dev/null)
fi
echo "$(date +%H:%M) running:[${st# }] | ${stage} | ${gpu}| ${prog}"
grep -nE "STOP|Traceback|OutOfMemory|CUDA out of memory|FAIL" $G/run_v16_reselect.log 2>/dev/null | tail -2 | sed "s|^|ALERT reselect: |" | cut -c1-220
[ -f $RUN/reselect.log ] && grep -nE "Traceback|Error|OutOfMemory|SystemExit" $RUN/reselect.log | grep -v UserWarning | tail -2 | sed "s|^|ALERT reselect.log: |" | cut -c1-220
