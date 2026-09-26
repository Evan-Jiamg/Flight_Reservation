B=/tmp2/hchsu/trec2026-usersim-benchmark
cd $B; ls; echo "--- results/methods"; ls results 2>/dev/null | head -40; ls methods 2>/dev/null | head -40
echo "--- files mentioning ditto (not generations)"
grep -rli ditto --include=*.md --include=*.json --include=*.py --include=*.csv --include=*.yaml . 2>/dev/null | grep -v generations | head -30
echo "--- leaderboard-like files"
ls results/*/ 2>/dev/null | head -60
