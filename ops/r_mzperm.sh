D=/tmp2/MingZhi_HcWang
echo "hcwang groups: $(id -nG hcwang 2>&1)"
echo "dir group: $(stat -c %G $D)"
if id -nG hcwang 2>/dev/null | tr ' ' '\n' | grep -qx "$(stat -c %G $D)"; then
  # group write on everything; setgid on directories so new files/dirs stay in the same group
  chmod -R g+w $D && find $D -type d -exec chmod g+s {} + && echo "applied: g+w (all), g+s (dirs)"
  ls -ld $D $D/MingZhi_Code $D/Report $D/Report/scai_sigconf
  ls -l $D/Report/scai_sigconf | head -6
  echo "not group-writable left: $(find $D ! -perm -g+w | wc -l)"
else
  echo "hcwang is NOT in group $(stat -c %G $D): nothing changed"
  echo "setfacl available: $(command -v setfacl || echo no)"
fi
