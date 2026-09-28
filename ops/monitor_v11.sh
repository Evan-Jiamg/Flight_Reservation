#!/bin/bash
# Supervision loop for the v11 formal pipeline: one ssh every ~10 minutes (server pacing rule), one event line per check
# (plus ALERT lines). After an ssh failure it waits 20 minutes before the next attempt.
cd /c/Users/User/.claude/worktrees/stop-sft-stageB/ops || exit 1
for i in 1 2 3; do
  out=$(bash ssh_run.sh 140.109.21.245 v11ops/watch.sh 2>&1 | grep -E "running:|ALERT|ssh:|timed out")
  [ -z "$out" ] && out="$(date +%H:%M) NO ANSWER from the server (ssh)"
  echo "$out" | tr '\n' ' '
  echo
  case "$out" in
    *"timed out"*|*"NO ANSWER"*) sleep 1200 ;;
    *) sleep 570 ;;
  esac
done
