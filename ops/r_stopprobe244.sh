for P in $(pgrep -f "planner_probe.py run"); do echo "stop probe pid $P"; kill $P; done
sleep 3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
