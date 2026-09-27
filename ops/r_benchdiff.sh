B=/tmp2/hchsu/trec2026-usersim-benchmark
GIT="git -c safe.directory=$B -C $B"
$GIT log --format="%h %ad %s" --date=iso -8 2>&1 | cut -c1-160
echo "--- files changed in the last commits (stat)"
$GIT log --format="=== %h %s" --stat -3 2>&1 | grep -E "^===|data/|tools/r0_client|tools/metrics/judge|folds3|req_shards|Ledger" | head -40 | cut -c1-160
echo "--- data files"
ls -la --time-style=full-iso $B/data/req_shards_v1.json $B/domains/main_dataset_search/folds3_goal_persona_v1.json 2>&1 | cut -c1-200
sha256sum $B/data/req_shards_v1.json $B/domains/main_dataset_search/folds3_goal_persona_v1.json 2>&1
echo "--- was r0_client.R0Client / Ledger in an earlier commit?"
for c in $($GIT log --format=%h -6 -- tools/r0_client.py 2>/dev/null); do
  echo "$c: $($GIT show $c:tools/r0_client.py 2>/dev/null | grep -cE '^class (R0Client|Ledger)')"
done
