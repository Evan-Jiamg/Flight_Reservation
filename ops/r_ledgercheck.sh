R=/tmp2/mzjiang_usersim/grpo_planner/stageC_v1
PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python
PYTHONNOUSERSITE=1 $PY - <<'EOF'
import json, glob
R="/tmp2/mzjiang_usersim/grpo_planner/stageC_v1"
eps=[json.loads(l) for f in sorted(glob.glob(R+"/rep0_shard*/nogate.jsonl")) for l in open(f)]
for e in eps:
    print(e["conversation_id"][:8], e["seed"], e["end_kind"], e["emitted_user_turns"], "cov", e["coverage"], "n_req", e["n_req"])
e=[x for x in eps if x["coverage"]==0][0]
print("LEDGER keys:", list((e["ledger"] or {}).keys()))
print(json.dumps(e["ledger"], ensure_ascii=False)[:1500])
print("USER t1:", e["trace"][0]["user"][:300])
print("USER t2:", e["trace"][1]["user"][:300])
print("AGENT t1:", (e["trace"][0]["agent"] or "")[:300])
EOF
grep -iE "judge|parse|error|warn" $R/rep0_shard2.log | grep -v -iE "pynvml|FutureWarning|import" | head -5
grep -c . /tmp2/hchsu/trec2026-usersim-benchmark/data/req_shards_v1.json
