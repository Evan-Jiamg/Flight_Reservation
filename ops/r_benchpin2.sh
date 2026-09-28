B=/tmp2/hchsu/trec2026-usersim-benchmark
GIT="git -c safe.directory=$B -C $B"
L=$($GIT log --format=%h --reverse 91b81bf..HEAD -- tools/r0_client.py | head -1)
P=$($GIT rev-parse --short $L^)
echo "Ledger removed by: $($GIT log --format='%h %ad %s' --date=iso -1 $L | cut -c1-150)"
echo "PIN (its parent): $($GIT log --format='%h %ad %s' --date=iso -1 $P | cut -c1-150)"
src=$($GIT show $P:tools/r0_client.py)
echo "at PIN: class R0Client $(echo "$src" | grep -c '^class R0Client')  class Ledger $(echo "$src" | grep -c '^class Ledger')"
echo "=== what changed in tools/metrics between 91b81bf and PIN"
$GIT log --format="%h %ad %s" --date=iso 91b81bf..$P -- tools/metrics tools/r0_client.py | cut -c1-150
echo "=== judge.py code diff 91b81bf..PIN"
$GIT diff -U0 91b81bf $P -- tools/metrics/judge.py | grep -E "^[+-]" | grep -vE "^[+-]\s*#|^(\+\+\+|---)|^[+-]\s*$" | head -40 | cut -c1-170
echo "=== r0_client code diff PIN..HEAD (only renames expected)"
$GIT diff -U0 $P HEAD -- tools/r0_client.py | grep -E "^[+-]" | grep -vE "^[+-]\s*#|^(\+\+\+|---)|^[+-]\s*$" | head -80 | cut -c1-170
echo "=== judge.py code diff PIN..HEAD"
$GIT diff -U0 $P HEAD -- tools/metrics/judge.py | grep -E "^[+-]" | grep -vE "^[+-]\s*#|^(\+\+\+|---)|^[+-]\s*$" | head -60 | cut -c1-170
echo "PINSHA $P"
