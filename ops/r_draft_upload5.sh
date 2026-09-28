R=/tmp2/MingZhi_HcWang/Report
[ -d $R ] || { echo "NO REPORT DIR"; exit 1; }
T=$(mktemp -d); echo "$PAYLOAD_B64" | base64 -d | tar xzf - -C $T
for f in Draft.md Draft_zh.md; do
  [ -e $R/$f ] && [ ! -e $R/$f.rev4 ] && cp -p $R/$f $R/$f.rev4
  cp $T/$f $R/$f && chmod 664 $R/$f
done
mkdir -p $R/fig/out && cp $T/fig/fig_method_spec.txt $R/fig/ && chmod -R g+rwX $R/fig
rm -rf $T; ls -la $R/Draft* $R/fig; head -1 $R/Draft.md; head -1 $R/Draft_zh.md
