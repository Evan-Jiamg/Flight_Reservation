G=/tmp2/mzjiang_usersim/grpo_planner
L=$(grep -vE "^\s*$" $G/run_v17_smoke.log | tail -2 | tr '\n' ' ' | cut -c1-220)
O=$(ls -td $G/runs/smoke_v17_f2_* 2>/dev/null | head -1)
S=""; [ -n "$O" ] && S=$(grep -vE "Loading weights|it/s\]|^\s*$" $O/smoke.log 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-250)
P=$(pgrep -u mzjiang -f "run_v17_smoke.sh" > /dev/null && echo running || echo stopped)
M=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')
echo "[$(date +%H:%M)] $P | GPU $M | launcher: $L | smoke: $S"
