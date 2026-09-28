B=/tmp2/hchsu/trec2026-usersim-benchmark
GIT="git -c safe.directory=$B -C $B"
OLD=7cf8ccd
$GIT log --format="%h %ad %s" --date=iso $OLD -1
$GIT log --format="%h %ad %s" --date=iso $OLD..HEAD -- tools/r0_client.py tools/metrics/judge.py | cut -c1-150
echo "=== diff stat"
$GIT diff --stat $OLD HEAD -- tools/r0_client.py tools/metrics/judge.py
echo "=== r0_client diff (code lines only, comments/docstrings dropped)"
$GIT diff -U0 $OLD HEAD -- tools/r0_client.py | grep -E "^[+-]" | grep -vE "^[+-]\s*#|^(\+\+\+|---)" | head -120 | cut -c1-170
echo "=== judge.py diff"
$GIT diff -U0 $OLD HEAD -- tools/metrics/judge.py | grep -E "^[+-]" | grep -vE "^[+-]\s*#|^(\+\+\+|---)" | head -60 | cut -c1-170
