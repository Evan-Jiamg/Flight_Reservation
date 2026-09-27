#!/bin/bash
# v16 preliminary experiments (user 2026-09-28: GPU smoke with the real model + Step 0), code pend_v16. GPU: whichever
# GPU is completely free (< 2 GiB used) at the moment of each step (user: GPU0 and GPU1 both usable, dynamic); a GPU
# someone else is using is never taken.
#  1. Planner vLLM on a free GPU (same flags as the formal run) -> smoke_v16.py (validation Task 1 end probabilities,
#     Task 1 groups with refill, one learner update: step-size statistics) -> the Planner server is stopped.
#  2. gpt-oss-120b on a free GPU (same flags as the formal run) -> step0_coverage.py on the fold-2 TRAIN
#     conversations -> gpt-oss is stopped; our processes leave every GPU.
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
  sleep 5                        # let the launcher exec into vllm before checking that it lives
  for i in $(seq 1 120); do
    up $1 $2 && return 0
    pgrep -u mzjiang -f "vllm serve .*--port $1" > /dev/null || { echo "server on $1 died:"; grep -E "Error|error" $3 | tail -3 | cut -c1-250; return 1; }
    sleep 10
  done
  return 1
}
stop_port() {   # our own server on this port only; escalates to SIGKILL, then waits for its GPU workers to go
  pkill -u mzjiang -f "vllm serve .*--port $1"
  for i in $(seq 1 60); do pgrep -u mzjiang -f "vllm serve .*--port $1" > /dev/null || break; sleep 2; done
  pkill -9 -u mzjiang -f "vllm serve .*--port $1"
  sleep 10
}
our_gpu() {     # our compute processes on any GPU
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "mzjiang" ] && echo $p
  done
}
pick_free() {   # the first GPU with < 2 GiB used; nothing when none (or when nvidia-smi cannot be read)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $2 < 2048 {print $1; exit}'
}
cleanup() {     # on any exit: our two servers down; our leftover GPU processes killed
  stop_port 8031; stop_port 8029
  for p in $(our_gpu); do kill -9 $p 2>/dev/null; done
}
cd $C
sha256sum -c --quiet local_sha_v16.txt || { echo "STOP: pend_v16 snapshot sha mismatch"; exit 1; }
if pgrep -u mzjiang -f "vllm serve|train_planner_rl.py|gpu_holder2" > /dev/null; then echo "STOP: our servers / training already running"; exit 1; fi
PG=$(pick_free)
[ -n "$PG" ] || { echo "STOP: no completely free GPU - not taking a GPU in use"; exit 1; }
trap cleanup EXIT
mkdir -p $OUT $Q/cache $R/cache
# ---- 1. smoke
echo "=== smoke: planner vLLM on GPU $PG $(date +%H:%M)"
cat > $Q/serve_prelim.sh <<EOF
#!/bin/bash
export PATH=$PYDIR:\$PATH CUDA_VISIBLE_DEVICES=$PG PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared VLLM_CACHE_ROOT=$Q/cache
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec $PYDIR/vllm serve $Q4 --served-model-name planner-base --dtype bfloat16 \\
  --enable-lora --max-lora-rank 16 --max-loras 2 --enable-prefix-caching \\
  --max-model-len 20480 --host 127.0.0.1 --port 8031 --gpu-memory-utilization 0.15
EOF
setsid nohup bash $Q/serve_prelim.sh > $OUT/planner_vllm.log 2>&1 < /dev/null &
wait_up 8031 planner-base $OUT/planner_vllm.log || { echo "STOP: planner vLLM did not come up"; exit 1; }
echo "planner vLLM up $(date +%H:%M)"
rm -rf $OUT/smoke_run
CUDA_VISIBLE_DEVICES=$PG timeout 3h $PY smoke_v16.py --fold 2 --planner-path $Q4 --gpu 0 --G 4 --scenarios-per-update 4 \
    --rollout-workers 4 --out $OUT/smoke_run --val-convs 2 > $OUT/smoke.log 2>&1
rc=$?; echo "smoke rc=$rc on GPU $PG $(date +%H:%M)"
stop_port 8031
[ $rc -ne 0 ] && { echo "STOP: smoke failed"; grep -v "Loading weights" $OUT/smoke.log | tail -15 | cut -c1-300; exit 1; }
grep -v "Loading weights" $OUT/smoke.log | tail -60 | cut -c1-300
for p in $(our_gpu); do kill -9 $p 2>/dev/null; done
sleep 10
# ---- 2. Step 0
PG=$(pick_free)
[ -n "$PG" ] || { echo "STOP: no completely free GPU for Step 0"; exit 1; }
echo "=== step 0: gpt-oss on GPU $PG $(date +%H:%M)"
cat > $R/serve_prelim.sh <<EOF
#!/bin/bash
export PATH=$PYDIR:\$PATH CUDA_VISIBLE_DEVICES=$PG PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared VLLM_CACHE_ROOT=$R/cache
exec $PYDIR/vllm serve $M --served-model-name gpt-oss-120b \\
  --max-model-len 12288 --host 127.0.0.1 --port 8029 --gpu-memory-utilization 0.78
EOF
setsid nohup bash $R/serve_prelim.sh > $OUT/gptoss.log 2>&1 < /dev/null &
wait_up 8029 gpt-oss-120b $OUT/gptoss.log || { echo "STOP: gpt-oss did not come up"; exit 1; }
echo "gpt-oss up $(date +%H:%M)"
timeout 2h $PY step0_coverage.py --fold 2 --splits $G/splits_v1.json --out $OUT/step0_f2.jsonl --workers 4 > $OUT/step0.log 2>&1
rc=$?; echo "step0 rc=$rc $(date +%H:%M)"
stop_port 8029
[ $rc -ne 0 ] && { echo "STOP: step 0 failed"; tail -15 $OUT/step0.log | cut -c1-300; exit 1; }
cat $OUT/step0.log | cut -c1-300
cleanup
trap - EXIT
left=$(our_gpu | wc -l)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
[ "$left" -eq 0 ] || { echo "WARNING: $left of our processes still on a GPU"; exit 1; }
echo "V16 PRELIM DONE $(date)"
