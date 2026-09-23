ls -la /tmp2/hchsu/.env 2>&1 | cut -c1-80
T=/tmp2/mzjiang_usersim/task2
grep -n "OPENAI\|env\|source" $T/run_task2_pair.sh /tmp2/mzjiang_usersim/grpo_planner/run_stop_evaluation_when_ready.sh 2>/dev/null | sed 's/sk-[A-Za-z0-9_-]*/sk-REDACTED/g' | head
for f in ~/.bashrc ~/.profile ~/.bash_profile; do grep -l "OPENAI_API_KEY" $f 2>/dev/null; done
[ -n "${OPENAI_API_KEY:-}" ] && echo "OPENAI_API_KEY set in ssh env" || echo "OPENAI_API_KEY not in ssh env"
bash -ic '[ -n "${OPENAI_API_KEY:-}" ] && echo "set in interactive bash" || echo "not in interactive bash"' 2>/dev/null
cat $T/runstats_stop_gate_base_4bit.json | head -c 400
