G=/tmp2/mzjiang_usersim/grpo_planner; D=$G/dev_v3; B=/tmp2/hchsu/trec2026-usersim-benchmark
mkdir -p $D && cd $D && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
: <<"SKIP"
import json
o=json.load(open("/tmp2/hchsu/trec2026-usersim-benchmark/domains/main_dataset_search/folds3_goal_persona_v1.json"))
print(type(o).__name__, list(o.keys())[:10] if isinstance(o,dict) else len(o))
f=(o.get("folds") if isinstance(o,dict) else o)[0]; print({k:(v if not isinstance(v,list) else "list[%d]"%len(v)) for k,v in f.items()})
SKIP
PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python make_crossfit_groups.py \
  --nested-manifest $G/nested/nested_manifest.json --outer-folds $B/domains/main_dataset_search/folds3_goal_persona_v1.json \
  --shards $B/data/req_shards_v1.json --union $G/stageC_v1/scenarios_union.json --out $G/crossfit_manifest_v1.json
