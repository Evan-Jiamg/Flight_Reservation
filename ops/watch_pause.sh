#!/bin/bash
# local: poll the server every ~5 min until the u5 pause is done (one line per notable event)
cd /c/Users/User/.claude/worktrees/stop-sft-stageB/ops
for i in $(seq 1 12); do
  r=$(bash ssh_run.sh 140.109.21.245 r_pausecheck.sh 2>&1 | tail -1)
  case "$r" in
    DONE*) echo "u5 done, u25 paused"; exit 0 ;;
    *"re-selection ended"*|*timed*|*refused*) echo "check: $r" ;;
  esac
  sleep 285
done
echo "still waiting after 60 min"
