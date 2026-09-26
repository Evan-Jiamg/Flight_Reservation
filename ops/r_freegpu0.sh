G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1; C2=$G/code_snapshots/stageC_v2
# the rep1 queue has already launched all 4 jobs; make sure no scheduler of ours can start anything new
pkill -f "stageC_ditto_rep1/run_stageC_ditto_rep1.sh" && echo "rep1 queue script stopped (its jobs keep running)"
P=$(pgrep -f "scenarios_shard2.json.*--replicate 1 .*rep1_ditto_shard2"); echo "shard2 pid: $P"; [ -n "$P" ] && kill $P
sleep 10
echo "shard2 kept episodes: $(wc -l < $R/rep1_ditto_shard2/nogate.jsonl)"
mv $R/rep1_ditto_shard2.log $R/rep1_ditto_shard2.log.part1_gpu0
echo "shard2 stopped on GPU0 to free it for a teammate $(date); resumes on GPU1 only" >> $G/run_stageC_ditto_rep1.log
cat > $G/resume_shard2_gpu1.sh <<'EOF'
#!/bin/bash
# Resume Ditto rep1 shard2 on GPU1 ONLY, after shard1 (GPU1) finishes. Never touches GPU0.
G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/stageC_v1; C2=$G/code_snapshots/stageC_v2
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /home/mzjiang/.secrets/openai.env; set +a
while pgrep -f "scenarios_shard1.json.*rep1_ditto_shard1" > /dev/null; do sleep 60; done
until [ "$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 42000 ]; do sleep 120; done
echo "resume shard2 on GPU1 $(date)" >> $G/run_stageC_ditto_rep1.log
cd $C2 && /home/mzjiang/miniconda3/envs/consistent-test/bin/python rollout_stop_sft.py \
  --scenarios $R/scenarios_shard2.json --fold -1 --side inner_union --gpu 1 --log-prompts \
  --replicate 1 --out-dir $R/rep1_ditto_shard2 --arm nogate=nogate --speaker ditto > $R/rep1_ditto_shard2.log 2>&1
echo "done ditto shard2 rep1 (resumed) rc=$? $(date)" >> $G/run_stageC_ditto_rep1.log
EOF
setsid nohup bash $G/resume_shard2_gpu1.sh > /dev/null 2>&1 < /dev/null &
sleep 3
echo "--- GPU state"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
ps -u mzjiang -o pid,args | grep -E "rollout_stop_sft|train_stop|score_logged|stageD|resume_shard2" | grep -v grep | cut -c1-150
