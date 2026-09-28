echo "pid ppid sid tty stat etime cmd  (tty '?' = no terminal; sid = own session => survives ssh/VPN drops)"
for pat in run_v16_reselect gpu_holder2 "train_planner_rl" "vllm serve"; do
  for p in $(pgrep -u mzjiang -f "$pat"); do ps -o pid=,ppid=,sid=,tty=,stat=,etime=,args= -p $p | cut -c1-110; done
done
bash /tmp2/mzjiang_usersim/grpo_planner/../grpo_planner/watch_v16.sh >/dev/null 2>&1
python3 - <<'PYEOF'
import json
rows = [json.loads(l) for l in open("/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/reselect.jsonl") if l.strip()]
for u in (0, 5, 10):
    e = sum(1 for r in rows if r.get("kind") == "episode" and r["update"] == u)
    d = any(r.get("kind") == "summary" and r["update"] == u for r in rows)
    print("u%d: %d/32 episodes%s" % (u, e, " (summary done)" if d else ""))
PYEOF
