#!/usr/bin/env bash
# crop_pdf.sh -- trim a PDF to its ink, then emit the matching EPS and PNG.
#
# For diagrams exported from draw.io and the like, which come out as a full A4
# page with the artwork sitting in one corner. \includegraphics would then pull
# in all that whitespace, and scaling to \columnwidth would shrink the drawing
# to nothing.
#
# Ghostscript measures the ink (-sDEVICE=bbox), the page is re-imposed at that
# size with PageOffset, and Poppler converts the result. Everything stays
# vector; pdfcrop is not needed and is not installed here.
#
#   ./crop_pdf.sh Pipeline.pdf [margin_pt]
set -euo pipefail

src="${1:?usage: crop_pdf.sh <file.pdf> [margin_pt]}"
margin="${2:-2}"
base="${src%.pdf}"

read -r llx lly urx ury < <(
    gs -q -dBATCH -dNOPAUSE -sDEVICE=bbox "$src" 2>&1 \
    | awk '/^%%HiResBoundingBox:/ {print $2, $3, $4, $5; exit}')

w=$(awk "BEGIN{printf \"%.0f\", $urx - $llx + 2*$margin}")
h=$(awk "BEGIN{printf \"%.0f\", $ury - $lly + 2*$margin}")
ox=$(awk "BEGIN{printf \"%.4f\", -($llx - $margin)}")
oy=$(awk "BEGIN{printf \"%.4f\", -($lly - $margin)}")

echo "  ink   : $llx $lly $urx $ury"
echo "  crop  : ${w} x ${h} pt (margin ${margin}pt)"

[ -f "${base}_uncropped.pdf" ] || cp "$src" "${base}_uncropped.pdf"

gs -q -dBATCH -dNOPAUSE -sDEVICE=pdfwrite \
   -dDEVICEWIDTHPOINTS="$w" -dDEVICEHEIGHTPOINTS="$h" -dFIXEDMEDIA \
   -dPDFSETTINGS=/prepress -dSubsetFonts=true -dEmbedAllFonts=true \
   -sOutputFile="${base}_c.pdf" \
   -c "<</PageOffset [$ox $oy]>> setpagedevice" -f "$src"

mv "${base}_c.pdf" "$src"
pdftops -eps -level3 "$src" "${base}.eps"
gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r300 -dTextAlphaBits=4 \
   -dGraphicsAlphaBits=4 -sOutputFile="${base}.png" "$src"

echo "  wrote : $(basename "$src") / $(basename "$base").eps / $(basename "$base").png"
echo "  kept  : $(basename "$base")_uncropped.pdf"
