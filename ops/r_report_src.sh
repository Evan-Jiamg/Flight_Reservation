# Collect report sources (read-only): Sep-Simulator docs, benchmark README/docs, Report template -> base64 tgz on stdout
B=/tmp2/hchsu/trec2026-usersim-benchmark
S=$(ls -d /home/mzjiang/Sep-Simulator /home/mzjiang/Sep-1st-Simulator /home/MingZhi/Sep-1st-Simulator 2>/dev/null | head -1)
R=/tmp2/MingZhi_HcWang/Report
T=/tmp2/mzjiang_usersim/.report_src; rm -rf $T; mkdir -p $T/sep $T/bench $T/report
echo "SEP_DIR=$S" > $T/INDEX.txt
[ -n "$S" ] && { find $S -maxdepth 3 -name "*.md" -size -400k -not -path "*/node_modules/*" -not -path "*/.git/*" | head -80 >> $T/INDEX.txt
  find $S -maxdepth 3 -name "*.md" -size -400k -not -path "*/node_modules/*" -not -path "*/.git/*" | head -80 | while read f; do mkdir -p $T/sep/$(dirname ${f#$S/}); cp "$f" $T/sep/${f#$S/}; done; }
echo "=== bench tree" >> $T/INDEX.txt
(cd $B && find . -maxdepth 3 -not -path "./.git*" -not -path "*/__pycache__*" | sort | head -400) >> $T/INDEX.txt
(cd $B && find . -maxdepth 4 \( -name "*.md" -o -name "*.bib" -o -name "*.txt" -o -name "*.cff" \) -size -300k -not -path "./.git*" -not -path "*/data/*" | head -120) | while read f; do mkdir -p $T/bench/$(dirname $f); cp "$B/$f" $T/bench/$f; done
echo "=== report tree" >> $T/INDEX.txt
(cd $R && find . -maxdepth 3 | sort) >> $T/INDEX.txt
(cd $R && find . -maxdepth 3 \( -name "*.tex" -o -name "*.bib" -o -name "*.md" \) -size -300k) | while read f; do mkdir -p $T/report/$(dirname $f); cp "$R/$f" $T/report/$f; done
cd $T && tar czf - . | base64 -w0
rm -rf $T
