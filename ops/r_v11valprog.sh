RUN=/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11
python3 - <<'PYEOF'
import json, os, time
p = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/validation.jsonl"
if not os.path.exists(p):
    print("no validation rows yet")
else:
    rows = [json.loads(l) for l in open(p) if l.strip()]
    for r in rows:
        if r["kind"] == "episode":
            e = r["episode"]
            print("VAL-EP %s s%d att%d emitted=%d human=%d end=%s cov=%.3f clean=%s at %s" % (e["conversation_id"][:8], r["seed"],
                  r.get("attempt", 0), e["emitted_user_turns"], e["human_turns"], e["end_kind"], e["coverage"], e["clean"],
                  time.strftime("%H:%M", time.localtime(r["time"]))))
        elif r["kind"] == "task1":
            t = r["task1"]
            print("VAL-T1 %s k1_blank=%s ends=%s at %s" % (r["conversation_id"][:8], t.get("k1_speaker_blank"),
                  [x["t"] for x in t["turns"] if x["ended_planner"]], time.strftime("%H:%M", time.localtime(r["time"]))))
PYEOF
grep -v "Loading weights" $RUN/train.log | tail -3 | cut -c1-200
