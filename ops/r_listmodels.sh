ls /tmp2/hf_shared/hub | sed 's/models--//' | grep -v "^\." | head -80
grep -n "max_length=12000\|truncation" /home/mzjiang/Sep-Simulator/sepsim/models.py | head
PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python -c "import json;c=json.load(open('/tmp2/TREC_UserSim_MingZhi/UserLM/00-models-v4GRPO-deps/Qwen2.5-32B-Instruct/config.json'));print('32B max_position_embeddings',c.get('max_position_embeddings'),'rope_scaling',c.get('rope_scaling'))"
