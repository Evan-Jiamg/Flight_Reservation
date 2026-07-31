#!/usr/bin/env python3
"""verify_bundle.py -- is the committed bundle a faithful copy of the raw grid?

Hashes every file the bundle carries against its source. The bundle is what a
clone gets, so this is the claim that has to hold: identical bytes, nothing
silently re-encoded or truncated.

  python3 analysis/verify_bundle.py
"""
import glob
import hashlib
import os
import sys

RAW = "/mnt/NewSSD/CS_project/neil/hcog_experiments/M-1_main-grid/phi4"
BUNDLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      os.pardir, "results", "M-1_main-grid", "phi4")
BUNDLE = os.path.normpath(BUNDLE)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = sorted(glob.glob(os.path.join(BUNDLE, "*/*/alpha_*/seed_*/*")))
    if not files:
        sys.exit("no bundle at " + BUNDLE)

    same = differ = orphan = 0
    for p in files:
        rel = os.path.relpath(p, BUNDLE)
        src = os.path.join(RAW, rel)
        if not os.path.exists(src):
            print("  no source: " + rel)
            orphan += 1
        elif sha(p) == sha(src):
            same += 1
        else:
            print("  DIFFERS: " + rel)
            differ += 1

    total = sum(os.path.getsize(p) for p in files)
    print()
    print("bundle files : %d  (%.1f MB)" % (len(files), total / 1048576))
    print("identical    : %d" % same)
    print("differing    : %d" % differ)
    print("no source    : %d" % orphan)

    kinds = {}
    for p in files:
        kinds[os.path.basename(p)] = kinds.get(os.path.basename(p), 0) + 1
    print()
    for k, v in sorted(kinds.items()):
        print("  %-24s %d" % (k, v))

    return 1 if (differ or orphan) else 0


if __name__ == "__main__":
    sys.exit(main())
