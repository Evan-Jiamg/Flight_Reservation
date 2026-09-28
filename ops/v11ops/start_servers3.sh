# Start OUR two vLLM servers on WHICHEVER cfda5 GPU first has room for both (user 2026-09-26: watch every GPU):
#   gpt-oss-120b (R0 + ledger judge + LLM controller)  port 8029, memory share 0.78
#   Qwen3-4B Planner generation (runtime LoRA)          port 8031, memory share 0.15
# The chosen GPU is written to $G/server_gpu.txt (the trainer takes another GPU). Other users' processes are never
# touched: we only wait until a GPU has room. Both servers bound to 127.0.0.1.
G=/tmp2/mzjiang_usersim/grpo_planner
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin
R=/tmp2/mzjiang_usersim/r0_vllm; Q=/tmp2/mzjiang_usersim/planner_vllm
mkdir -p $R $Q
M=$(ls -d /tmp2/hf_shared/hub/models--openai--gpt-oss-120b/snapshots/*/ | head -1)
Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
up() { curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q "$2"; }
freeof() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $1 | awk -F', ' '{print int(($1-$2)/1024)}'; }
wait_up() {
  for i in $(seq 1 120); do
    up $1 $2 && return 0
    if ! pgrep -u mzjiang -f "$3" > /dev/null; then echo "server process died:"; grep -E "Error|error" $4 | tail -3 | cut -c1-250; return 1; fi
    sleep 10
  done
  return 1
}
if up 8029 gpt-oss-120b && up 8031 planner-base; then
  echo "servers already up on GPU $(cat $G/server_gpu.txt 2>/dev/null)"; exit 0; fi
if up 8029 gpt-oss-120b || up 8031 planner-base; then
  # one of the two is up (e.g. the planner died): restart the missing one on the SAME GPU
  SG=$(cat $G/server_gpu.txt); need=0
  up 8029 gpt-oss-120b || need=76
  up 8031 planner-base || need=15
  n=0; until [ "$(freeof $SG)" -ge $need ]; do
    [ $((n % 10)) -eq 0 ] && echo "WAITING_GPU: need $need GiB on server GPU $SG, have $(freeof $SG) ($(date +%H:%M))"; n=$((n + 1)); sleep 60; done
else
  n=0; SG=""
  while [ -z "$SG" ]; do
    SG=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \
         awk -F', ' '{f=int(($2-$3)/1024); if (f>=91) print f" "$1}' | sort -nr | head -1 | cut -d" " -f2)
    if [ -z "$SG" ]; then
      [ $((n % 10)) -eq 0 ] && echo "WAITING_GPU: need 91 GiB free on one GPU; free now: $(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{printf "g%s=%dG ", $1, ($2-$3)/1024}')($(date +%H:%M))"
      n=$((n + 1)); sleep 60
    fi
  done
  echo "server GPU $SG has room: $(freeof $SG) GiB free"
  echo $SG > $G/server_gpu.txt
fi
if ! up 8029 gpt-oss-120b; then
cat > $R/serve.sh <<EOF
#!/bin/bash
export PATH=$PYDIR:\$PATH CUDA_VISIBLE_DEVICES=$SG PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared VLLM_CACHE_ROOT=$R/cache
exec $PYDIR/vllm serve $M --served-model-name gpt-oss-120b \\
  --max-model-len 12288 --host 127.0.0.1 --port 8029 --gpu-memory-utilization 0.78
EOF
chmod +x $R/serve.sh
setsid nohup bash $R/serve.sh > $R/serve.log 2>&1 < /dev/null &
echo "gpt-oss launched on GPU $SG (0.78) $(date +%H:%M)"
wait_up 8029 gpt-oss-120b "port 8029" $R/serve.log || { echo "ABORT: gpt-oss did not come up"; exit 1; }
fi
echo "gpt-oss up"
if ! up 8031 planner-base; then
cat > $Q/serve.sh <<EOF
#!/bin/bash
export PATH=$PYDIR:\$PATH CUDA_VISIBLE_DEVICES=$SG PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared VLLM_CACHE_ROOT=$Q/cache
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec $PYDIR/vllm serve $Q4 --served-model-name planner-base --dtype bfloat16 \\
  --enable-lora --max-lora-rank 16 --max-loras 2 --enable-prefix-caching \\
  --max-model-len 20480 --host 127.0.0.1 --port 8031 --gpu-memory-utilization 0.15
EOF
chmod +x $Q/serve.sh
setsid nohup bash $Q/serve.sh > $Q/serve.log 2>&1 < /dev/null &
echo "planner vLLM launched on GPU $SG (0.15) $(date +%H:%M)"
wait_up 8031 planner-base "port 8031" $Q/serve.log || { echo "ABORT: planner vLLM did not come up"; exit 1; }
fi
echo "planner vLLM up"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
