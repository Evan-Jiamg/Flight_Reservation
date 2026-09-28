"""Download the report-relevant PDFs of the two public Drive folders and extract their text (pypdf)."""
import json
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
items = json.load(open(os.path.join(HERE, "drive_list.json"), encoding="utf-8"))
keep = [x for x in items if not x["folder"] and (x["root"] == "internal-meetings" or "UserSim" in x["name"])]
os.makedirs(os.path.join(HERE, "pdf"), exist_ok=True)
os.makedirs(os.path.join(HERE, "txt"), exist_ok=True)
from pypdf import PdfReader
for x in keep:
    fn = re.sub(r"[^\w.\-]+", "_", x["name"])
    if not fn.lower().endswith(".pdf"):
        fn += ".pdf"
    p = os.path.join(HERE, "pdf", ("R_" if x["root"] == "TREC-reports" else "M_") + fn)
    if not os.path.exists(p) or os.path.getsize(p) < 1000:
        url = "https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t" % x["id"]
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=120).read()
        open(p, "wb").write(data)
    try:
        r = PdfReader(p)
        pages = [pg.extract_text() or "" for pg in r.pages]
        t = "\n".join("--- page %d ---\n%s" % (i + 1, s) for i, s in enumerate(pages))
        open(os.path.join(HERE, "txt", os.path.basename(p)[:-4] + ".txt"), "w", encoding="utf-8").write(t)
        print("OK %3d pages %7d chars  %s" % (len(pages), len(t), os.path.basename(p)))
    except Exception as e:
        print("FAIL %s: %s (%d bytes)" % (os.path.basename(p), e, os.path.getsize(p)))
