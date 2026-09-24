G=/tmp2/mzjiang_usersim/grpo_planner
head -32 $G/stageC_v1/rep0/stageC_table.txt | cut -c1-200
grep -o '"max_abs_p_stop_diff": [0-9.e-]*' $G/stageC_v1/smoke_crn/equivalence.json
PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python - <<'EOF'
import json, collections
R="/tmp2/mzjiang_usersim/grpo_planner/stageC_v1/rep0/nogate.jsonl"
eps=[json.loads(l) for l in open(R)]
print("nogate episodes", len(eps), "end kinds", collections.Counter(e["end_kind"] for e in eps))
print("mean emitted %.2f coverage %.3f complete %d/%d" % (sum(e["emitted_user_turns"] for e in eps)/len(eps), sum(e["coverage"] for e in eps)/len(eps), sum(bool(e["complete"]) for e in eps), len(eps)))
# coverage gained after human K turns
hum={}
for l in open("/home/mzjiang/v5-latency/data.jsonl"):
    r=json.loads(l); hum[r["conversation_id"]]=sum(1 for m in r["chat_messages"] if m["participant_name"].lower()=="user")
gain=[]
for e in eps:
    k=hum[e["conversation_id"]]; tr=[s for s in e["trace"] if s["decision"]=="continue"]
    at_k=max([s["coverage_after"] for s in tr if s["t"]<=k] or [0.0])
    gain.append(e["coverage"]-at_k)
print("coverage gained after human stop turn K: mean %.3f, episodes with any gain %d/%d" % (sum(gain)/len(gain), sum(g>0 for g in gain), len(gain)))
EOF
