# One-line supervision snapshot of the v16 night: night runner, prelim, formal run (+ ALERT lines). Read-only.
G=/tmp2/mzjiang_usersim/grpo_planner; RUN=$G/runs/pend_f2_v16
st=""
for s in run_v16_night run_v16_prelim run_v16_launch run_v16_formal run_v16_guard gpu_holder2 train_planner_rl smoke_v16 step0_coverage; do
  pgrep -u mzjiang -f "$s" > /dev/null && st="$st $s"
done
LAST=$(ls -t $G/run_v16_formal_retry*.log 2>/dev/null | head -1)
stage=$(cat $G/run_v16_night.log $G/run_v16_prelim.log $G/run_v16_launch.log $G/run_v16_formal.log $LAST $G/run_v16_guard.log 2>/dev/null | \
  grep -E "^(NIGHT|===|WAITING_GPU|server GPU|gpt-oss|planner vLLM|smoke rc|step0 rc|SMOKE VERDICT|train rc|verify|stop rule|early stop|LAUNCHER|V16|GUARD)" | tail -1 | cut -c1-120)
gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '{printf "g%s=%dG ", $1, $2/1024}')
srv=""
curl -s -m 5 http://127.0.0.1:8029/v1/models | grep -q gpt-oss-120b && srv="${srv}oss:up " || srv="${srv}oss:down "
curl -s -m 5 http://127.0.0.1:8031/v1/models | grep -q planner-base && srv="${srv}planner:up" || srv="${srv}planner:down"
upd=""
if [ -f $RUN/updates.jsonl ]; then
  upd=$(tail -1 $RUN/updates.jsonl | python3 -c "import json,sys; u=json.loads(sys.stdin.read()); a=u['train_aggregate']; st=u['learner_stats']; t=a.get('task1_train') or {}; print('U%d rew=%.3f turns=%.2f cov=%.3f mism=%s kl=%s rl_gn=%s aux_gn=%s aux_w=%s t1 fin/early=%s/%s %.0fs' % (u['update'], a['reward_mean'], a['components_mean'].get('turns',0), a['components_mean'].get('coverage',0), st.get('behav_mismatch_mean'), st.get('kl'), st.get('rl_grad_norm'), st.get('aux_grad_norm'), a.get('aux_weight'), t.get('end_at_final'), t.get('end_at_nonfinal'), u['update_s']))" 2>/dev/null)
fi
nro=0; [ -f $RUN/rollouts.jsonl ] && nro=$(wc -l < $RUN/rollouts.jsonl)
hold="$(for g in 0 1; do printf "g%s=%s/%s " $g "$(cat $G/hold/status_$g 2>/dev/null)" "$(cat $G/hold/target_$g 2>/dev/null)"; done)srv=$(cat $G/hold/role_server 2>/dev/null) trn=$(cat $G/hold/role_train 2>/dev/null) "
echo "$(date +%H:%M) running:[${st# }] | ${stage} | ${gpu}| held:${hold}| ${srv} | rollouts=${nro} ${upd}"
for f in $G/run_v16_night.log $G/run_v16_prelim.log $G/run_v16_launch.log $([ -n "$LAST" ] && echo $LAST || echo $G/run_v16_formal.log) $G/gpu_holder2.log $G/run_v16_guard.log; do
  [ -f $f ] && grep -nE "ABORT|STOP|Traceback|FAILED|FAIL |Killed|OutOfMemory|CUDA out of memory|MismatchAbort|did not pass|needs attention|giving up|WARNING" $f | tail -2 | sed "s|^|ALERT $(basename $f): |" | cut -c1-240
done
for f in $RUN/train.log $G/runs/v16_prelim/smoke.log $G/runs/v16_prelim/step0.log; do
  [ -f $f ] && grep -nE "Traceback|Error|OutOfMemory|Killed|AssertionError|SystemExit" $f | grep -v "UserWarning" | tail -2 | sed "s|^|ALERT $(basename $f): |" | cut -c1-240
done
if pgrep -u mzjiang -f train_planner_rl.py > /dev/null && [ -f $RUN/rollouts.jsonl ]; then
  age=$(( $(date +%s) - $(stat -c %Y $RUN/rollouts.jsonl $RUN/validation.jsonl 2>/dev/null | sort -n | tail -1) ))
  [ $age -gt 2700 ] && echo "ALERT stale: no new rollout / validation row for $((age / 60)) min while training runs"
fi
