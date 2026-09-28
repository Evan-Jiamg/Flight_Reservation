G=/tmp2/mzjiang_usersim/grpo_planner
echo "--- ditto analysis"; tail -3 $G/ditto_analysis_rep0.log | cut -c1-200; [ -f $G/stageC_v1/ditto_rep0/table.txt ] && grep -E "hazard|nogate" $G/stageC_v1/ditto_rep0/table.txt | cut -c1-200
echo "--- ditto rep1"; cat $G/run_stageC_ditto_rep1.log; for f in $G/stageC_v1/rep1_ditto_shard*/nogate.jsonl; do [ -f $f ] && echo "$(basename $(dirname $f)) $(wc -l < $f)"; done
echo "--- candidate local judge models"
ls /tmp2/hf_shared/hub | grep -i -E "gpt-oss|qwen3|qwen2.5-(32|72)|llama|mistral" | head -20
ls -d /tmp2/*/models/*gpt-oss* /tmp2/*/*gpt-oss* 2>/dev/null | head
echo "--- qwen3-4b snapshot"
ls /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head; du -shL /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507 2>/dev/null
