#!/bin/bash
# Supervision loop for the v11 formal pipeline: every ~10 minutes one snapshot of cfda5 (pipeline + GPUs, ALERT lines)
# and one of cfda4 (GPUs only). Several connections are allowed since the 2026-09-26 server upgrade (kept to two).
cd /c/Users/User/.claude/worktrees/stop-sft-stageB/ops || exit 1
for i in 1 2 3; do
  out=$(timeout 150 bash ssh_run.sh 140.109.21.245 v11ops/watch.sh 2>&1 | grep -E "running:|ALERT|ssh:|timed out")
  [ -z "$out" ] && out="$(date +%H:%M) NO ANSWER from cfda5 (ssh)"
  c4=$(timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes 140.109.21.244 \
       "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits" 2>/dev/null | \
       awk -F', ' '{printf "g%s=%d/%dG ", $1, $2/1024, $3/1024}')
  echo "$out" | tr '\n' ' '
  echo "| cfda4: ${c4:-no answer}"
  case "$out" in
    *"timed out"*|*"NO ANSWER"*) sleep 1200 ;;
    *) sleep 560 ;;
  esac
done
