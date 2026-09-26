# Start OUR two vLLM servers on GPU0 (user-approved layout 2026-09-26), WAITING for the memory they need:
#   gpt-oss-120b (R0 + ledger judge + LLM controller)  port 8029, memory share 0.78
#   Qwen3-4B Planner generation (runtime LoRA)          port 8031, memory share 0.15
# Other users' processes are never touched: we only wait until GPU0 has room. Both bound to 127.0.0.1.
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin
R=/tmp2/mzjiang_usersim/r0_vllm; Q=/tmp2/mzjiang_usersim/planner_vllm
mkdir -p $R $Q
M=$(ls -d /tmp2/hf_shared/hub/models--openai--gpt-oss-120b/snapshots/*/ | head -1)
Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
up() { curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q "$2"; }
free0() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 | awk -F', ' '{print int(($1-$2)/1024)}'; }
# wait_up PORT NAME PATTERN LOG: up within 20 min, and fail FAST when the server process has died
wait_up() {
  for i in $(seq 1 120); do
    up $1 $2 && return 0
    if ! pgrep -u mzjiang -f "$3" > /dev/null; then echo "server process died:"; grep -E "Error|error" $4 | tail -3 | cut -c1-250; return 1; fi
    sleep 10
  done
  return 1
}
need=0
up 8029 gpt-oss-120b || need=$((need + 76))
up 8031 planner-base || need=$((need + 15))
if [ $need -gt 0 ]; then
  n=0
  until [ "$(free0)" -ge $need ]; do
    [ $((n % 10)) -eq 0 ] && echo "WAITING_GPU0: need ${need} GiB free on GPU0, have $(free0) GiB ($(date +%H:%M))"
    n=$((n + 1)); sleep 60
  done
  echo "GPU0 has room: $(free0) GiB free"
fi
if up 8029 gpt-oss-120b; then echo "gpt-oss already up"; else
cat > $R/serve.sh <<EOF
#!/bin/bash
export PATH=$PYDIR:\$PATH CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared VLLM_CACHE_ROOT=$R/cache
exec $PYDIR/vllm serve $M --served-model-name gpt-oss-120b \\
  --max-model-len 12288 --host 127.0.0.1 --port 8029 --gpu-memory-utilization 0.78
EOF
chmod +x $R/serve.sh
setsid nohup bash $R/serve.sh > $R/serve.log 2>&1 < /dev/null &
echo "gpt-oss launched (0.78) $(date +%H:%M)"
wait_up 8029 gpt-oss-120b "port 8029" $R/serve.log || { echo "ABORT: gpt-oss did not come up"; exit 1; }
fi
echo "gpt-oss up"
if up 8031 planner-base; then echo "planner vLLM already up"; else
cat > $Q/serve.sh <<EOF
#!/bin/bash
export PATH=$PYDIR:\$PATH CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared VLLM_CACHE_ROOT=$Q/cache
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec $PYDIR/vllm serve $Q4 --served-model-name planner-base --dtype bfloat16 \\
  --enable-lora --max-lora-rank 16 --max-loras 2 --enable-prefix-caching \\
  --max-model-len 20480 --host 127.0.0.1 --port 8031 --gpu-memory-utilization 0.15
EOF
chmod +x $Q/serve.sh
setsid nohup bash $Q/serve.sh > $Q/serve.log 2>&1 < /dev/null &
echo "planner vLLM launched (0.15) $(date +%H:%M)"
wait_up 8031 planner-base "port 8031" $Q/serve.log || { echo "ABORT: planner vLLM did not come up"; exit 1; }
fi
echo "planner vLLM up"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
