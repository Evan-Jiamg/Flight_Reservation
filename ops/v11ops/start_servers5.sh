# Start OUR two vLLM servers on cfda5, taking the memory from the all-at-once GPU placeholder (gpu_holder2.py):
#   gpt-oss-120b (R0 + ledger judge + LLM controller)  port 8029, memory share 0.78
#   Qwen3-4B Planner generation (runtime LoRA)          port 8031, memory share 0.15
# The holder picks the server GPU (role_server) the moment one GPU has room for both and takes 91 GiB at once.
# Hand-over: release down to 16 GiB -> start gpt-oss -> release to 0 -> start the Planner vLLM.
G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin
R=/tmp2/mzjiang_usersim/r0_vllm; Q=/tmp2/mzjiang_usersim/planner_vllm
mkdir -p $R $Q $H
M=$(ls -d /tmp2/hf_shared/hub/models--openai--gpt-oss-120b/snapshots/*/ | head -1)
Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
up() { curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q "$2"; }
held() { cat $H/status_$1 2>/dev/null || echo 0; }
settarget() { echo $2 > $H/target_$1.tmp && mv -f $H/target_$1.tmp $H/target_$1; }
waitheld_le() { until [ "$(held $1)" -le $2 ]; do sleep 1; done; }
waitheld_ge() { n=0; until [ "$(held $1)" -ge $2 ]; do [ $((n % 600)) -eq 0 ] && echo "WAITING_GPU: g$1 holds $(held $1) GiB, need $2 ($(date +%H:%M))"; n=$((n + 1)); sleep 1; done; }
wait_up() {
  for i in $(seq 1 120); do
    up $1 $2 && return 0
    if ! pgrep -u mzjiang -f "$3" > /dev/null; then echo "server process died:"; grep -E "Error|error" $4 | tail -3 | cut -c1-250; return 1; fi
    sleep 10
  done
  return 1
}
pgrep -u mzjiang -f gpu_holder2.py > /dev/null || { echo "ABORT: the GPU holder is not running"; exit 1; }
if up 8029 gpt-oss-120b && up 8031 planner-base; then echo "servers already up on GPU $(cat $H/role_server)"; exit 0; fi
n=0
until [ -f $H/role_server ]; do
  [ $((n % 600)) -eq 0 ] && echo "WAITING_GPU: no GPU with room for the servers yet; used: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '{printf "g%s=%dG ", $1, $2/1024}')($(date +%H:%M))"
  n=$((n + 1)); sleep 1
done
SG=$(cat $H/role_server)
echo "server GPU $SG (holder role) $(date +%H:%M)"
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
