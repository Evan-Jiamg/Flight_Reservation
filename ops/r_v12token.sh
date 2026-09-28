python3 - <<'PYEOF'
import json
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
rows = [json.loads(l) for l in open(O + "rollouts.jsonl")]
for r in rows[158:]:
    e = r["episode"]
    print("u%d %s token=%s total=%s own=%s clean=%s" % (r["update"], e["conversation_id"][:8], e.get("process_token"),
          e.get("r0_len_truncated_total"), (e.get("episode_counters") or {}).get("r0_len_truncated"), e.get("clean")))
PYEOF
tail -1 /tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/run_meta.jsonl | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print('meta', r.get('kind'), 'allow_code_change', (r.get('args') or {}).get('allow_code_change'), 'task2_env', (r.get('code_sha256') or {}).get('task2_env.py','')[:12])"
sha256sum /tmp2/mzjiang_usersim/grpo_planner/code_snapshots/pend_v12/task2_env.py | cut -c1-12
