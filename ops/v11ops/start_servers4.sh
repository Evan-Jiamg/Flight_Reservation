# Start OUR two vLLM servers on cfda5 with the GPU placeholder (gpu_holder.py) holding the memory until the hand-over:
#   gpt-oss-120b (R0 + ledger judge + LLM controller)  port 8029, memory share 0.78
#   Qwen3-4B Planner generation (runtime LoRA)          port 8031, memory share 0.15
# Server GPU = the first GPU where held+free reaches 91 GiB; the other GPU becomes the training GPU (hold 45 GiB).
# Hand-over: release down to 16 GiB -> start gpt-oss -> release to 0 -> start the Planner vLLM (no gap for others).
G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin
R=/tmp2/mzjiang_usersim/r0_vllm; Q=/tmp2/mzjiang_usersim/planner_vllm
mkdir -p $R $Q $H
M=$(ls -d /tmp2/hf_shared/hub/models--openai--gpt-oss-120b/snapshots/*/ | head -1)
Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
up() { curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q "$2"; }
freeof() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $1 | awk -F', ' '{print int(($1-$2)/1024)}'; }
held() { cat $H/status_$1 2>/dev/null || echo 0; }
settarget() { echo $2 > $H/target_$1.tmp && mv -f $H/target_$1.tmp $H/target_$1; }
waitheld_le() { until [ "$(held $1)" -le $2 ]; do sleep 1; done; }
waitheld_ge() { n=0; until [ "$(held $1)" -ge $2 ]; do [ $((n % 300)) -eq 0 ] && echo "WAITING_GPU: g$1 holds $(held $1) GiB, need $2 ($(date +%H:%M))"; n=$((n + 1)); sleep 2; done; }
wait_up() {
  for i in $(seq 1 120); do
    up $1 $2 && return 0
    if ! pgrep -u mzjiang -f "$3" > /dev/null; then echo "server process died:"; grep -E "Error|error" $4 | tail -3 | cut -c1-250; return 1; fi
    sleep 10
  done
  return 1
}
pgrep -u mzjiang -f gpu_holder.py > /dev/null || { echo "ABORT: the GPU holder is not running"; exit 1; }
if up 8029 gpt-oss-120b && up 8031 planner-base; then echo "servers already up on GPU $(cat $G/server_gpu.txt)"; exit 0; fi
if [ -f $G/server_gpu.txt ] && (up 8029 gpt-oss-120b || up 8031 planner-base); then
  SG=$(cat $G/server_gpu.txt)
else
  # decide the server GPU: the first where our hold + remaining free reaches 91 GiB
  n=0; SG=""
  while [ -z "$SG" ]; do
    for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
      [ $(( $(held $g) + $(freeof $g) )) -ge 91 ] && { SG=$g; break; }
    done
    if [ -z "$SG" ]; then
      [ $((n % 300)) -eq 0 ] && echo "WAITING_GPU: need 91 GiB on one GPU; held/free now: $(for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do printf 'g%s=%s+%sG ' $g $(held $g) $(freeof $g); done)($(date +%H:%M))"
      n=$((n + 1)); sleep 2
    fi
  done
  echo $SG > $G/server_gpu.txt
  TG=$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -vx "$SG" | head -1)
  echo $TG > $H/train_gpu
  settarget $TG 45
  echo "server GPU $SG, training GPU $TG ($(date +%H:%M))"
fi
if ! up 8029 gpt-oss-120b; then
  need=76; up 8031 planner-base || need=91
  settarget $SG $need; waitheld_ge $SG $need
  if up 8031 planner-base; then settarget $SG 0; waitheld_le $SG 0; else settarget $SG 16; waitheld_le $SG 16; fi
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
  [ "$(held $SG)" -lt 16 ] && { settarget $SG 16; waitheld_ge $SG 16; }
  settarget $SG 0; waitheld_le $SG 0
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
settarget $SG 0
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
