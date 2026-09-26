# One-line supervision snapshot of the v11 formal pipeline (+ ALERT lines). Read-only.
G=/tmp2/mzjiang_usersim/grpo_planner; RUN=$G/runs/pend_f2_v11
st=""
for s in run_v11_launch5 run_v11_formal5 run_v11_continue5 gpu_holder2 run_v11_guard; do
  pgrep -u mzjiang -f "$s" > /dev/null && st="$st $s"
done
LAST=$(ls -t $G/run_v11_continue5_retry*.log 2>/dev/null | head -1)
stage=$(cat $G/run_v11_launch5.log $G/run_v11_formal5.log $G/run_v11_continue5.log $LAST $G/run_v11_guard.log 2>/dev/null | grep -E "^(===|WAITING_GPU|server GPU|gpt-oss|planner vLLM|train rc|verify|stop rule|early stop|LAUNCHER|V11|GUARD)" | tail -1 | cut -c1-110)
gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '{printf "g%s=%dG ", $1, $2/1024}')
srv=""
curl -s -m 5 http://127.0.0.1:8029/v1/models | grep -q gpt-oss-120b && srv="${srv}oss:up " || srv="${srv}oss:down "
curl -s -m 5 http://127.0.0.1:8031/v1/models | grep -q planner-base && srv="${srv}planner:up" || srv="${srv}planner:down"
upd=""
if [ -f $RUN/updates.jsonl ]; then
  upd=$(tail -1 $RUN/updates.jsonl | python3 -c "import json,sys; u=json.loads(sys.stdin.read()); a=u['train_aggregate']; st=u['learner_stats']; print('U%d rew=%.3f turns=%.2f mism=%s kl=%s %.0fs' % (u['update'], a['reward_mean'], a['components_mean'].get('turns',0), st.get('behav_mismatch_mean'), st.get('kl'), u['update_s']))" 2>/dev/null)
fi
nro=0; [ -f $RUN/rollouts.jsonl ] && nro=$(wc -l < $RUN/rollouts.jsonl)
hold="$(for g in 0 1; do printf "g%s=%s/%s " $g "$(cat $G/hold/status_$g 2>/dev/null)" "$(cat $G/hold/target_$g 2>/dev/null)"; done)srv=$(cat $G/hold/role_server 2>/dev/null) trn=$(cat $G/hold/role_train 2>/dev/null) "
echo "$(date +%H:%M) running:[${st# }] | ${stage} | ${gpu}| held:${hold}| ${srv} | rollouts=${nro} ${upd}"
# alerts
for f in $G/run_v11_launch5.log $G/run_v11_formal5.log $([ -n "$LAST" ] && echo $LAST || echo $G/run_v11_continue5.log) $G/gpu_holder2.log $G/run_v11_guard.log; do
  [ -f $f ] && grep -nE "ABORT|STOP:|Traceback|FAILED|Killed|OutOfMemory|CUDA out of memory|MismatchAbort|did not pass|needs attention|giving up" $f | tail -2 | sed "s|^|ALERT $(basename $f): |" | cut -c1-240
done
if [ -f $RUN/train.log ]; then
  grep -nE "Traceback|Error|OutOfMemory|Killed|AssertionError|SystemExit" $RUN/train.log | grep -v "UserWarning" | tail -2 | sed "s|^|ALERT train.log: |" | cut -c1-240
fi
if [ -n "$st" ] && [ -f $RUN/rollouts.jsonl ] && pgrep -u mzjiang -f train_planner_rl.py > /dev/null; then
  age=$(( $(date +%s) - $(stat -c %Y $RUN/rollouts.jsonl) ))
  [ $age -gt 2700 ] && echo "ALERT stale: no new rollout row for $((age / 60)) min while training runs"
fi
ls $RUN/ABORTED_u*.json 2>/dev/null | sed 's|^|ALERT mismatch abort marker: |'
