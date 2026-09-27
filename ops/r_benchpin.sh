B=/tmp2/hchsu/trec2026-usersim-benchmark
GIT="git -c safe.directory=$B -C $B"
PIN=""
for c in $($GIT log --format=%h -- tools/r0_client.py tools/metrics 2>/dev/null); do
  src=$($GIT show $c:tools/r0_client.py 2>/dev/null)
  if echo "$src" | grep -q "^class R0Client" && echo "$src" | grep -q "^class Ledger"; then PIN=$c; break; fi
done
echo "PIN=$PIN $($GIT log --format='%ad %s' --date=iso -1 $PIN | cut -c1-140)"
echo "=== commits after the pin touching tools/r0_client.py or tools/metrics"
$GIT log --format="%h %ad %s" --date=iso $PIN..HEAD -- tools/r0_client.py tools/metrics | cut -c1-150
echo "=== code diff pin..HEAD tools/r0_client.py + tools/metrics (comments/docstring-only lines dropped)"
$GIT diff -U0 $PIN HEAD -- tools/r0_client.py tools/metrics | grep -E "^[+-]" | grep -vE "^[+-]\s*#|^(\+\+\+|---)|^[+-]\s*$" | cut -c1-170
echo "=== metrics files at the pin"
$GIT ls-tree -r --name-only $PIN tools/metrics tools/r0_client.py
