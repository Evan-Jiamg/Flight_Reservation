PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python
PYTHONNOUSERSITE=1 $PY - <<'EOF'
import json, glob
R="/tmp2/mzjiang_usersim/grpo_planner/stageC_v1"; T="/tmp2/mzjiang_usersim/task2"
new={(e["conversation_id"],e["seed"]):e for f in glob.glob(R+"/rep0_shard*/nogate.jsonl") for l in open(f) for e in [json.loads(l)]}
for old_name in ("episodes_stop_gate_base_4bit.jsonl","episodes_full.jsonl"):
    old={}
    for l in open(f"{T}/{old_name}"):
        e=json.loads(l); old[(e["conversation_id"],e["seed"])]=e
    same=tot=0
    for k,e in new.items():
        if k in old:
            ou=[s.get("user","") for s in old[k]["trace"] if (s.get("user") or "").strip()]
            tot+=1; same+= bool(ou) and ou[0]==e["trace"][0]["user"]
            if k[0].startswith(("3192ee7b","780b64d1")):
                print(old_name[:28], k[0][:8], k[1], "old cov", old[k]["coverage"], "| old t1:", (ou[0] if ou else "")[:90])
    print(old_name, "first-turn identical", same, "/", tot)
EOF
