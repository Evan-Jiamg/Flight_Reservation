G=/tmp2/mzjiang_usersim/grpo_planner; D=$G/dev_v3; B=/tmp2/hchsu/trec2026-usersim-benchmark
mkdir -p $D && cd $D && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python make_splits_v1.py \
  --nested-manifest $G/nested/nested_manifest.json --outer-folds $B/domains/main_dataset_search/folds3_goal_persona_v1.json \
  --shards $B/data/req_shards_v1.json --out $G/splits_v1.json && sha256sum $G/splits_v1.json
echo "--- run2"; grep -v -i warn $G/v3_run2.log | tail -4; tail -1 $G/v3_run1/probe_v3/labels_probe.log 2>/dev/null | cut -c1-100
tail -1 $G/v3_run1/judge/labels_llama70b.log | cut -c1-100
echo "--- multiwoz data"; ls $B/domains | head; ls $B/domains/multiwoz* 2>/dev/null | head; find $B -maxdepth 3 -iname "*multiwoz*" | head
