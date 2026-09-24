#!/bin/bash
# Survey lab hosts ONE AT A TIME, 20 s apart (pacing rule), single batched command each.
for h in 140.109.21.221 140.109.21.244 140.109.21.238 140.109.21.243; do
  sleep 20
  echo "================ $h"
  timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes $h 'hostname; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1 | head -8; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -8; df -h /tmp2 2>/dev/null | tail -1; nproc; free -g | sed -n 2p; ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen2.5-7B-Instruct /tmp2/mzjiang_usersim 2>&1 | head -2' 2>&1 | tail -22
done
