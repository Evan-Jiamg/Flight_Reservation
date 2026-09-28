#!/bin/bash
# local poller for the v10 gate: one ssh check every 10 min (server pacing rule), exits on done/dead
cd /c/Users/User/.claude/worktrees/stop-sft-stageB/ops || exit 1
for i in $(seq 1 30); do
  sleep 600
  out=$(bash ssh_run.sh 140.109.21.245 r_v10status.sh 2>&1)
  echo "[$i $(date +%H:%M)] $(echo "$out" | head -1) | $(echo "$out" | grep -E '^=== ' | tail -1)"
  case "$out" in *STATE_DONE*|*STATE_DEAD*) echo "$out" | tail -90; exit 0;; esac
  case "$out" in *STATE_*) ;; *) sleep 600;; esac      # ssh hung / timed out: wait 10 more minutes
done
