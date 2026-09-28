for d in /home/mzjiang/miniconda3/envs/*/lib/python3*/site-packages/transformers /tmp2/mzjiang*/*/lib/python3*/site-packages/transformers /tmp2/mzjiang_usersim/*/*/lib/python3*/site-packages/transformers; do
  [ -d "$d" ] || continue
  v=$(grep -m1 "__version__" $d/__init__.py | cut -d'"' -f2)
  q=$([ -d $d/models/qwen3_vl ] && echo qwen3_vl || echo -)
  echo "$v $q $d"
done
echo "--- how was ditto_swap_full produced"
ls -la /tmp2/mzjiang_usersim/grpo_planner/ditto_swap_full.jsonl
grep -rl "Ditto\|ditto" /tmp2/mzjiang_usersim/grpo_planner/*.sh /tmp2/mzjiang_usersim/*.sh /tmp2/mzjiang_usersim/logs/*.log 2>/dev/null | head
ls /tmp2/mzjiang_usersim/logs 2>/dev/null | grep -i ditto
head -c 600 /tmp2/mzjiang_usersim/logs/*ditto*.log 2>/dev/null
ps aux | grep -i -E "vllm|ditto" | grep -v grep | cut -c1-200 | head -5
