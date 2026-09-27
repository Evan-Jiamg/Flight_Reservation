G=/tmp2/mzjiang_usersim/grpo_planner; RUN=$G/runs/pend_f2_v11
echo "--- v12 continuation tail"; tail -6 $G/run_v12_continue.log | cut -c1-220
echo "--- v12 guard"; tail -2 $G/run_v12_guard.log | cut -c1-200
echo "--- v13 continuation"; cat $G/run_v13_continue.log | cut -c1-220
echo "--- train.log intervention"; grep -n "INTERVENTION" $RUN/train.log | tail -2 | cut -c1-220
tail -1 $RUN/run_meta.jsonl | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print('meta', r['kind'], 'intervention', r.get('intervention') and {k: r['intervention'][k] for k in ('at_update','set_cfg','controller_bounds')})"
tail -2 $G/runs/pend_f2_v11/llm_controller.jsonl | python3 -c "
import json,sys
for l in sys.stdin:
    r=json.loads(l); print('ctrl u%s' % r.get('update'), 'human' if r.get('human_intervention') else '', r.get('applied') or r.get('human_intervention'), 'rollback', r.get('rollback'), 'after', r.get('cfg_after'))"
