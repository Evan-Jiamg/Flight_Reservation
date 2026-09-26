PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python - <<'EOF'
import glob, json, re
from collections import Counter
R = "/tmp2/mzjiang_usersim/grpo_planner/stageC_v1"
eps = [json.loads(l) for p in sorted(glob.glob(R + "/rep0_ditto_shard*/nogate.jsonl")) for l in open(p)]
c = Counter(); n = 0
for e in eps:
    s = next((x for x in e["trace"] if x.get("ended_planner")), None)
    if not s: continue
    n += 1
    u = (s.get("user") or "").strip()
    q = "?" in u
    closing = bool(re.search(r"\b(thank|thanks|perfect|exactly what i needed|that's all|bye|appreciate|satisfied|all set)\b", u.lower()))
    c[("question" if q else "no-question") + "/" + ("closing-words" if closing else "no-closing-words")] += 1
print("first planner-end utterances:", n, dict(c))
# also: across ALL steps with ended_planner, share of questions
allq = [("?" in (s.get("user") or "")) for e in eps for s in e["trace"] if s.get("ended_planner")]
print("all end-act steps:", len(allq), "with a question mark:", sum(allq))
EOF
