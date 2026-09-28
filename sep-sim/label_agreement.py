#!/usr/bin/env python3
"""Agreement between two label sets (3-class status), on their common parsed ids."""
import argparse
import json
from collections import Counter

CLASSES = ("SATISFIED", "PARTIAL", "NOT")


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in CLASSES) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else None   # undefined when both raters use one class


def load(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if r.get("parse_ok", True) and r["status"] in CLASSES:
            out[r["id"]] = r["status"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    pairs = [(A[i], B[i]) for i in common]
    conf = Counter(pairs)
    print(json.dumps({"n_common": len(common), "agreement": sum(a == b for a, b in pairs) / max(1, len(pairs)),
                      "cohen_kappa": kappa(pairs),
                      "confusion_a_by_b": {"%s|%s" % k: v for k, v in sorted(conf.items())},
                      "dist_a": dict(Counter(a for a, _ in pairs)), "dist_b": dict(Counter(b for _, b in pairs))},
                     indent=1))


if __name__ == "__main__":
    main()
