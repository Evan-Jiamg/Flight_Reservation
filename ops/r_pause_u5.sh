G=/tmp2/mzjiang_usersim/grpo_planner
echo "$PAYLOAD_B64" | base64 -d > $G/pause_after_u5.sh && chmod +x $G/pause_after_u5.sh
pgrep -u mzjiang -f pause_after_u5.sh > /dev/null && { echo "already running"; exit 1; }
setsid nohup bash $G/pause_after_u5.sh > $G/pause_after_u5.log 2>&1 < /dev/null &
sleep 5; cat $G/pause_after_u5.log
python3 - <<'PYEOF'
import json
rows = [json.loads(l) for l in open("/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/reselect.jsonl") if l.strip()]
print("u5 episodes so far (this run):", sum(1 for r in rows if r.get("kind") == "episode" and r["update"] == 5 and r["time"] > 1790489000),
      "task1:", sum(1 for r in rows if r.get("kind") == "task1" and r["update"] == 5))
PYEOF
