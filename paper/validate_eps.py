#!/usr/bin/env python3
"""validate_eps.py -- submission check for the paper's EPS figures.

For each .eps: EPSF header present, Ghostscript renders it without error, the
content is pure vector, and every font is embedded and subsetted.

Two traps this checker deliberately avoids:
  * pdftops writes "/pdfImM { fCol imagemask skipEOD } def" into its prolog for
    every file. That is a procedure DEFINITION, not raster content -- grepping
    for the bare operator name flags every clean file. Real raster content shows
    up as an image dictionary (ImageType / DataSource / ImageMatrix).
  * pdffonts prints emb, sub and uni. Only emb and sub matter for submission;
    uni is the ToUnicode map, which affects copy-paste out of the PDF, not
    rendering or embedding, and is routinely absent after a PostScript round
    trip. Matching a bare "no" across those columns fails every good file.

  python3 validate_eps.py DIR [DIR ...]
"""
import glob
import os
import re
import subprocess
import sys


def raster_ops(body):
    """Actual image content, ignoring prolog procedure definitions."""
    n = len(re.findall(r"/(ImageType|DataSource|ImageMatrix)\b", body))
    for line in body.splitlines():
        if line.strip().endswith("def"):        # a definition, not a call
            continue
        if re.search(r"\b(colorimage|imagemask)\b", line):
            n += 1
    return n


def fonts_of(eps):
    """(names, all_embedded_and_subset) via a PostScript -> PDF round trip."""
    if subprocess.run(["ps2pdf", "-dEPSCrop", eps, "/tmp/_veps.pdf"],
                      capture_output=True).returncode != 0:
        return [], False
    out = subprocess.run(["pdffonts", "/tmp/_veps.pdf"],
                         capture_output=True).stdout.decode()
    names, ok = [], True
    for line in out.splitlines()[2:]:
        c = line.split()
        if len(c) < 6:
            continue
        # trailing columns are: emb sub uni object ID
        emb, sub = c[-5], c[-4]
        names.append(c[0].split("+")[-1])
        if emb != "yes" or sub != "yes":
            ok = False
    return names, ok


def main(dirs):
    bad = 0
    for d in dirs:
        print("=== %s ===" % d)
        print("%-34s %-18s %-5s %-7s %s"
              % ("file", "BoundingBox", "gs", "raster", "fonts"))
        print("-" * 100)
        for p in sorted(glob.glob(os.path.join(d, "*.eps"))):
            body = open(p, "rb").read().decode("latin-1")
            first = body.splitlines()[0]
            epsf = first.startswith("%!PS-Adobe") and "EPSF" in first
            m = re.search(r"^%%BoundingBox: (.+)$", body, re.M)
            bb = m.group(1).strip() if m else "MISSING"

            r = subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dEPSCrop",
                                "-sDEVICE=nullpage", p], capture_output=True)
            gs_ok = r.returncode == 0 and not r.stderr.strip()

            nimg = raster_ops(body)
            names, emb_ok = fonts_of(p)

            flag = ""
            if not epsf:
                flag += " NOT-EPSF"
            if not gs_ok:
                flag += " GS-FAIL"
            if nimg:
                flag += " RASTERISED"
            if not emb_ok:
                flag += " FONT-NOT-EMBEDDED"
            if flag:
                bad += 1

            print("%-34s %-18s %-5s %-7d %s%s"
                  % (os.path.basename(p), bb, "ok" if gs_ok else "FAIL", nimg,
                     ", ".join(sorted(set(names))), flag))
        print()

    if bad:
        print("PROBLEMS in %d file(s)" % bad)
        return 1
    print("ALL EPS PASS -- EPSF header, clean Ghostscript render, pure vector, "
          "all fonts embedded and subsetted")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["figures/official_paper"]))
