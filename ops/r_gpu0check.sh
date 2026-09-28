echo "=== gpt-oss serve.log (last 25, filtered)"
grep -iE "error|memory|cuda|started|Uvicorn|Application startup|KV cache|ValueError|RuntimeError|Available" /tmp2/mzjiang_usersim/r0_vllm/serve.log | tail -25 | cut -c1-260
echo "=== GPU apps (pid, MiB) and whether the pid is ours"
ours=" $(pgrep -u mzjiang | tr '\n' ' ') "
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader | while IFS=, read pid mem uuid; do
  case "$ours" in *" $pid "*) o=OURS;; *) o=other;; esac; echo "$pid $mem $o"; done
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
curl -s -m 5 http://127.0.0.1:8029/v1/models | head -c 120; echo
