NAME=Draft_zh.md
R=/tmp2/MingZhi_HcWang/Report
[ -d $R ] || { echo "NO REPORT DIR"; exit 1; }
if [ -e $R/${NAME:-Draft.md} ]; then echo "Draft.md EXISTS - not overwriting; writing Draft_claude.md instead"; F=$R/Draft_claude.md; else F=$R/${NAME:-Draft.md}; fi
echo "$PAYLOAD_B64" | base64 -d > $F && chmod 664 $F
ls -la $F; head -3 $F; wc -l $F
