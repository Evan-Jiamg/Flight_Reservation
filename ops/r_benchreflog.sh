B=/tmp2/hchsu/trec2026-usersim-benchmark
GIT="git -c safe.directory=$B -C $B"
echo "=== reflog (HEAD moves of the shared tree)"
$GIT reflog -n 12 --date=iso 2>&1 | cut -c1-150
echo "=== .git/logs/HEAD tail"
tail -5 $B/.git/logs/HEAD 2>&1 | awk '{print $1, $2, $5, $6, substr($0, index($0,$7))}' | cut -c1-170
