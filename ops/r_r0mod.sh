B=/tmp2/hchsu/trec2026-usersim-benchmark
ls -la --time-style=full-iso $B/tools/r0_client.py $B/tools/metrics/judge.py 2>&1 | cut -c1-200
grep -nE "^class |^def |^[A-Z_]+ *=" $B/tools/r0_client.py | head -30
cd $B && git log --oneline -5 -- tools/r0_client.py 2>&1 | head -5
cd $B && git status --short tools/r0_client.py 2>&1 | head -3
ls -la --time-style=full-iso $B/tools/ | head -30 | cut -c1-150
