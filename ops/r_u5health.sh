G=/tmp2/mzjiang_usersim/grpo_planner; RUN=$G/runs/pend_f2_v11
ps -o pid,etime,stat,pcpu,rss,cmd -u mzjiang | grep -E "train_planner_rl|pause_after|run_v15" | grep -v grep | cut -c1-120
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
python3 - <<'PYEOF'
import json, time
rows = [json.loads(l) for l in open("/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/reselect.jsonl") if l.strip()]
t0 = time.mktime(time.strptime("2026-09-27 14:11:32", "%Y-%m-%d %H:%M:%S"))
eps = [r for r in rows if r.get("kind") == "episode" and r["update"] == 5 and r["time"] >= t0]
t1 = [r for r in rows if r.get("kind") == "task1" and r["update"] == 5 and r["time"] >= t0]
print("now %s | u5 episodes %d/32 clean %d, attempts>0: %d | task1 %d/4" % (time.strftime("%H:%M:%S"), len(eps),
      sum(1 for r in eps if r["episode"]["clean"]), sum(1 for r in eps if r.get("attempt", 0) > 0), len(t1)))
last = max(r["time"] for r in rows)
print("last row written %s (%d s ago)" % (time.strftime("%H:%M:%S", time.localtime(last)), time.time() - last))
done = sorted((r["conversation_id"][:8], r["seed"]) for r in eps)
print("done (conv, seed):", done)
print("policy sha of u5 rows:", {r["policy_sha"][:12] for r in eps})
st = json.load(open("/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/ckpt/u00005/state.json"))
print("ckpt u5 policy sha:", st["policy_sha"][:12])
for r in eps[-3:]:
    e = r["episode"]
    print("  %s s%d turns %d human %d cov %.2f end %s counters %s" % (r["conversation_id"][:8], r["seed"], e["emitted_user_turns"],
          e["human_turns"], e["coverage"], e["end_kind"], {k: v for k, v in (e.get("episode_counters") or {}).items() if v}))
PYEOF
grep -v "Loading weights" $RUN/reselect.log | tail -4 | cut -c1-200
grep -nE "Traceback|Error" $RUN/reselect.log | tail -3 | cut -c1-200
tail -1 $G/pause_after_u5.log
