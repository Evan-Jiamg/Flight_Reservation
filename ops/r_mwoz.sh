B=/tmp2/hchsu/trec2026-usersim-benchmark/domains/multiwoz
ls -la $B | head -30
for f in $(ls $B | head -20); do [ -f $B/$f ] && { echo "=== $f ($(wc -c < $B/$f) bytes)"; head -c 700 $B/$f; echo; }; done
find $B -maxdepth 2 -type f | head -20
