G=/tmp2/mzjiang_usersim/grpo_planner; RUN=$G/runs/pend_f2_v16
L=$(grep -vE "^\s*[\{\}\"]|^   " $G/run_v16_test.log | tail -3 | tr '\n' ' ' | cut -c1-260)
P=$(pgrep -u mzjiang -f "run_v16_test.sh" > /dev/null && echo running || echo stopped)
C=$(python3 - <<'PYEOF'
import json, os
p = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/test.jsonl"
rows = [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
out = []
for u in (0, 5):
    e = len({(r["conversation_id"], r["seed"]) for r in rows if r.get("kind") == "episode" and r["update"] == u})
    t = len({r["conversation_id"] for r in rows if r.get("kind") == "task1" and r["update"] == u})
    d = any(r.get("kind") == "summary" and r["update"] == u for r in rows)
    out.append("u%d %d/40 ep %d/9 t1%s" % (u, e, t, " DONE" if d else ""))
print("; ".join(out))
PYEOF
)
E=$(tail -200 $RUN/test.log 2>/dev/null | grep -cE "Traceback|Error|error:" )
M=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')
echo "[$(date +%H:%M)] $P | $C | errlines(last200)=$E | GPU $M | $L"
