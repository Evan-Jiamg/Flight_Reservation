#!/usr/bin/env bash
# eps_from_pdf.sh -- rebuild every EPS in a figure directory from its PDF.
#
# Why not let matplotlib write the EPS directly?
#   Nimbus Roman ships here as an OTF, i.e. CFF/PostScript outlines. Matplotlib's
#   PS backend can only wrap a glyf-based TrueType font as Type 42, so embedding
#   the OTF produced an invalid font and Ghostscript refused every file with
#   "/invalidfont in definefont". The PDF backend takes a different path and is
#   correct, so the PDF is the good master and Poppler converts it faithfully.
#
# This is only safe because the figures carry no transparency: alpha is folded
# into opaque RGB in make_*_figs.py (see blend()). PostScript has no alpha, so
# any surviving transparency would silently rasterise here.
#
#   ./eps_from_pdf.sh figures/official_paper
set -euo pipefail

dir="${1:?usage: eps_from_pdf.sh <figure-dir>}"
cd "$dir"

for pdf in *.pdf; do
    eps="${pdf%.pdf}.eps"
    pdftops -eps -level3 "$pdf" "$eps"
    printf '  %-40s -> %s\n' "$pdf" "$eps"
done

echo
echo "validating ..."
fail=0
for eps in *.eps; do
    head -1 "$eps" | grep -q 'EPSF' || { echo "  $eps: not EPSF"; fail=1; }
    # a rasterised fallback would show up as a large image operator
    if grep -qc 'colorimage' "$eps" 2>/dev/null; then
        n=$(grep -c 'colorimage' "$eps" || true)
        [ "$n" -gt 0 ] && { echo "  $eps: RASTERISED ($n image ops)"; fail=1; }
    fi
    gs -q -dNOPAUSE -dBATCH -dEPSCrop -sDEVICE=nullpage "$eps" >/dev/null 2>/tmp/_gserr || true
    if [ -s /tmp/_gserr ]; then
        echo "  $eps: ghostscript errors"; sed 's/^/      /' /tmp/_gserr | head -3; fail=1
    fi
done
[ "$fail" -eq 0 ] && echo "  all EPS render clean, no rasterisation" || echo "  PROBLEMS ABOVE"
