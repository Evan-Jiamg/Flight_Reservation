B=/tmp2/hchsu/trec2026-usersim-benchmark/tools/r0_client.py
grep -n "def __init__\|cache\|def reply\|def stats\|temperature\|seed" $B | head -40
echo ---; grep -n "def update\|def as_dict\|def coverage\|def complete" $B
echo ---; tail -c 600 /tmp2/mzjiang_usersim/task2/full.log
