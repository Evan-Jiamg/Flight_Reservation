python3 - <<'PYEOF'
import json, time
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
for i, l in enumerate(open(O + "rollouts.jsonl")):
    r = json.loads(l)
    if r["update"] in (8, 9):
        e = r["episode"]
        print("%3d u%d slot%d g%d %s t=%s rollout_s=%d total_trunc=%s counters=%s clean=%s turns=%d" % (i, r["update"], r["slot"],
              r["replicate"], e["conversation_id"][:8], time.strftime("%H:%M:%S", time.localtime(r["time"])), r["rollout_s"],
              e.get("r0_len_truncated_total"), e.get("episode_counters"), e.get("clean"), e["emitted_user_turns"]))
PYEOF
grep -n "Traceback\|Error\|r0-empty\|Warning" /tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/train.log | grep -v "Loading weights" | tail -8 | cut -c1-250
