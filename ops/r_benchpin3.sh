B=/tmp2/hchsu/trec2026-usersim-benchmark
GIT="git -c safe.directory=$B -C $B"
X=391f05b
$GIT log --format="%H %ad %s" --date=iso -1 $X | cut -c1-160
echo "tools diff $X..ca13b33 (files):"; $GIT diff --stat $X ca13b33 -- tools/ | tail -5
echo "r0_client at $X: $($GIT show $X:tools/r0_client.py | grep -cE '^class (R0Client|Ledger)') classes"
echo "data at $X vs live:"
for f in data/req_shards_v1.json domains/main_dataset_search/folds3_goal_persona_v1.json; do
  a=$($GIT show $X:$f | sha256sum | cut -c1-16); b=$(sha256sum $B/$f | cut -c1-16); echo "  $f  $X=$a live=$b"
done
