echo "--- Qwen3-4B candidates"
ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B* /tmp2/*/hub/models--Qwen--Qwen3-4B* /tmp2/*/models/Qwen3-4B* /tmp2/*/*/Qwen3-4B* /tmp2/*/*/models--Qwen--Qwen3-4B* 2>/dev/null | head
find /tmp2 -maxdepth 4 -iname "*qwen3-4b*" 2>/dev/null | head
echo "--- listening model servers"
ss -ltnp 2>/dev/null | grep -E ":80[0-9][0-9]|:90[0-9][0-9]|:30000" | head
for p in 8000 8001 8021 8022 8030 30000; do r=$(timeout 5 curl -s localhost:$p/v1/models 2>/dev/null | head -c 300); [ -n "$r" ] && echo "port $p: $r"; done
echo "--- judge config in benchmark"
grep -rn "8021\|gpt-oss" /tmp2/hchsu/trec2026-usersim-benchmark/tools/metrics/judge.py | head -5
