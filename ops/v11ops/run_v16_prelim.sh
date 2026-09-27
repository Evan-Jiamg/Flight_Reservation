#!/bin/bash
# v16 preliminary experiments on GPU0 only (user 2026-09-28: GPU smoke with the real model + Step 0), code pend_v16.
#  1. GPU0 must be free (< 2 GiB used): never take a GPU someone else is using.
#  2. Planner vLLM on GPU0 (same flags as the formal run) -> smoke_v16.py (validation Task 1 end probabilities,
#     Task 1 groups with refill, one learner update: step-size statistics) -> the Planner server is stopped.
#  3. gpt-oss-120b on GPU0 (same flags as the formal run) -> step0_coverage.py on the fold-2 TRAIN conversations ->
#     gpt-oss is stopped; GPU0 is left free.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v16
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
R=/tmp2/mzjiang_usersim/r0_vllm; Q=/tmp2/mzjiang_usersim/planner_vllm; OUT=$G/runs/v16_prelim
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export R0_BASE_URL=http://127.0.0.1:8029/v1 R0_MODEL=gpt-oss-120b R0_REASONING_EFFORT=low
export JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
export CONTROLLER_BASE_URL=http://127.0.0.1:8029/v1 OPENAI_API_KEY=local-vllm-unused
M=$(ls -d /tmp2/hf_shared/hub/models--openai--gpt-oss-120b/snapshots/*/ | head -1)
Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
up() { curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q "$2"; }
wait_up() {
  for i in $(seq 1 120); do
    up $1 $2 && return 0
    pgrep -u mzjiang -f "port $1" > /dev/null || { echo "server on $1 died:"; grep -E "Error|error" $3 | tail -3 | cut -c1-250; return 1; }
    sleep 10
  done
  return 1
}
stop_port() {   # our own server on this port only
  pkill -u mzjiang -f "vllm serve .*--port $1"
  for i in $(seq 1 60); do pgrep -u mzjiang -f "vllm serve .*--port $1" > /dev/null || break; sleep 2; done
  sleep 10
}
cd $C
sha256sum -c --quiet local_sha_v16.txt || { echo "STOP: pend_v16 snapshot sha mismatch"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$used" -gt 2048 ]; then echo "STOP: GPU0 is in use ($used MiB) - not taking it"; exit 1; fi
if pgrep -u mzjiang -f "vllm serve|train_planner_rl.py|gpu_holder2" > /dev/null; then echo "STOP: our servers / training already running"; exit 1; fi
mkdir -p $OUT $Q/cache $R/cache
# ---- 1. smoke
echo "=== smoke: planner vLLM on GPU0 $(date +%H:%M)"
cat > $Q/serve_prelim.sh <<EOF
#!/bin/bash
export PATH=$PYDIR:\$PATH CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared VLLM_CACHE_ROOT=$Q/cache
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec $PYDIR/vllm serve $Q4 --served-model-name planner-base --dtype bfloat16 \\
  --enable-lora --max-lora-rank 16 --max-loras 2 --enable-prefix-caching \\
  --max-model-len 20480 --host 127.0.0.1 --port 8031 --gpu-memory-utilization 0.15
EOF
setsid nohup bash $Q/serve_prelim.sh > $OUT/planner_vllm.log 2>&1 < /dev/null &
wait_up 8031 planner-base $OUT/planner_vllm.log || { echo "STOP: planner vLLM did not come up"; stop_port 8031; exit 1; }
echo "planner vLLM up $(date +%H:%M)"
rm -rf $OUT/smoke_run
CUDA_VISIBLE_DEVICES=0 $PY smoke_v16.py --fold 2 --planner-path $Q4 --gpu 0 --G 4 --scenarios-per-update 4 \
    --rollout-workers 4 --out $OUT/smoke_run --val-convs 2 > $OUT/smoke.log 2>&1
rc=$?; echo "smoke rc=$rc $(date +%H:%M)"
stop_port 8031
[ $rc -ne 0 ] && { echo "STOP: smoke failed"; grep -v "Loading weights" $OUT/smoke.log | tail -15 | cut -c1-300; exit 1; }
grep -v "Loading weights" $OUT/smoke.log | tail -60 | cut -c1-300
# ---- 2. Step 0
echo "=== step 0: gpt-oss on GPU0 $(date +%H:%M)"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$used" -gt 2048 ]; then echo "STOP: GPU0 not free after the smoke ($used MiB)"; exit 1; fi
cat > $R/serve_prelim.sh <<EOF
#!/bin/bash
export PATH=$PYDIR:\$PATH CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared VLLM_CACHE_ROOT=$R/cache
exec $PYDIR/vllm serve $M --served-model-name gpt-oss-120b \\
  --max-model-len 12288 --host 127.0.0.1 --port 8029 --gpu-memory-utilization 0.78
EOF
setsid nohup bash $R/serve_prelim.sh > $OUT/gptoss.log 2>&1 < /dev/null &
wait_up 8029 gpt-oss-120b $OUT/gptoss.log || { echo "STOP: gpt-oss did not come up"; stop_port 8029; exit 1; }
echo "gpt-oss up $(date +%H:%M)"
$PY step0_coverage.py --fold 2 --splits $G/splits_v1.json --out $OUT/step0_f2.jsonl --workers 4 > $OUT/step0.log 2>&1
rc=$?; echo "step0 rc=$rc $(date +%H:%M)"
stop_port 8029
[ $rc -ne 0 ] && { echo "STOP: step 0 failed"; tail -15 $OUT/step0.log | cut -c1-300; exit 1; }
cat $OUT/step0.log | cut -c1-300
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "V16 PRELIM DONE $(date)"
