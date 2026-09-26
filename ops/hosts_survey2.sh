#!/bin/bash
# Parallel read-only survey of the lab hosts (the servers were upgraded: several connections are allowed).
cd /c/Users/User/.claude/worktrees/stop-sft-stageB/ops || exit 1
for h in 140.109.21.244 140.109.21.245 140.109.21.221 140.109.21.238 140.109.21.243; do
  ( out=$(timeout 90 ssh -o ConnectTimeout=20 -o BatchMode=yes $h "bash -s" < r_hostinfo.sh 2>&1 | tail -1); echo "$h: $out" ) &
done
wait
