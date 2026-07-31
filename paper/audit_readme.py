#!/usr/bin/env python3
"""audit_readme.py -- check the README against the tree it describes.

Every path named in the layout block and in the fenced commands must exist, and
every top-level entry that exists must be named. A README that drifts from its
repository is worse than none: it sends a reviewer to files that are not there.

Also checks that requirements.txt covers the third-party imports the code
actually makes.
"""
import ast
import io
import os
import re
import sys

ROOT = ("/home/neil/Information_Management_Project/Echo-Chamber-Simulation/"
        "Hybrid-Network")
README = os.path.join(ROOT, "README.md")

STDLIB = set(sys.stdlib_module_names) | {"__future__"}
LOCAL = {"core", "config", "style", "agent", "numeric_agent", "prompt",
         "scorer", "utils", "model", "convergence", "make_official_figs"}
# import name -> distribution name
DIST = {"sklearn": "scikit-learn", "community": "python-louvain",
        "names_dataset": "names-dataset", "PIL": "pillow", "yaml": "pyyaml",
        "cv2": "opencv-python"}


def audit_paths(text):
    """Paths named in the layout tree and in fenced commands."""
    bad = []

    # Layout block. Depth comes from the column of the branch glyph, so a
    # bare leaf name is resolved against its parent rather than the root --
    # otherwise every nested file reads as missing.
    stack = {}
    for line in text.splitlines():
        m = re.match(r"^((?:[\s│]{4})*)[├└]──\s+([\w.\-]+/?)", line)
        if not m:
            continue
        depth = len(m.group(1)) // 4
        name = m.group(2)
        parent = stack.get(depth - 1, "") if depth else ""
        rel = os.path.join(parent, name.rstrip("/"))
        if name.endswith("/"):
            stack[depth] = rel
            for d in [k for k in stack if k > depth]:
                del stack[d]
        if not os.path.exists(os.path.join(ROOT, rel)):
            bad.append(("layout", rel))

    # python3 <path> and bash <path> inside fenced blocks
    for m in re.finditer(r"^(?:python3|bash)\s+([\w./\-]+)", text, re.M):
        p = m.group(1)
        if not os.path.exists(os.path.join(ROOT, p)):
            bad.append(("command", p))

    # --out / --grid style arguments that name a directory in the tree
    for m in re.finditer(r"--(?:out|outdir|grid)\s+([\w./\-]+)", text):
        p = m.group(1)
        if p.startswith("experiments/"):
            continue                      # symlinked, may be absent on a clone
        if not os.path.exists(os.path.join(ROOT, p)):
            bad.append(("argument", p))
    return bad


def audit_coverage(text):
    """Top-level entries that exist but the README never mentions."""
    skip = {".git", ".gitignore", "ops", "__pycache__", "README.md"}
    missing = []
    for e in sorted(os.listdir(ROOT)):
        if e in skip:
            continue
        if e not in text:
            missing.append(e)
    return missing


def audit_requirements():
    req = io.open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8").read()
    # Required lines install with -r; a commented "pip install ..." marks an
    # optional extra (vllm drags in a CUDA stack). Both count as declared, but
    # they are reported apart so an optional dep cannot masquerade as pinned.
    listed, optional = set(), set()
    for line in req.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith("#"):
            body = raw.lstrip("#").strip()
            if body.startswith("pip install"):
                for tok in body.split("pip install", 1)[1].split():
                    optional.add(re.split(r"[=<>!\[]", tok)[0].strip().lower())
            continue
        listed.add(re.split(r"[=<>!\[]", raw)[0].strip().lower())

    used = {}
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns
                  if d not in ("ops", "__pycache__", ".git", "results",
                               "experiments", "logs", "data")]
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dp, fn)
            try:
                tree = ast.parse(io.open(p, encoding="utf-8").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [(node.module or "").split(".")[0]]
                else:
                    continue
                for n in names:
                    if n and n not in STDLIB and n not in LOCAL:
                        used.setdefault(DIST.get(n, n).lower(),
                                        os.path.relpath(p, ROOT))
    missing = sorted((n, f) for n, f in used.items()
                     if n not in listed and n not in optional)
    return missing, sorted(listed), sorted(optional)


def main():
    text = io.open(README, encoding="utf-8").read()

    bad = audit_paths(text)
    print("== paths named in the README ==")
    if bad:
        for kind, p in bad:
            print("  MISSING (%s): %s" % (kind, p))
    else:
        print("  all resolve")

    print()
    print("== top-level entries not mentioned ==")
    miss = audit_coverage(text)
    print("  " + (", ".join(miss) if miss else "none"))

    print()
    print("== requirements coverage ==")
    uncovered, listed, optional = audit_requirements()
    if uncovered:
        for n, f in uncovered:
            print("  NOT LISTED: %-18s (imported by %s)" % (n, f))
    else:
        print("  every third-party import is pinned")
    print("  required: " + ", ".join(sorted(listed)))
    print("  optional: " + ", ".join(sorted(optional)))

    return 1 if (bad or miss or uncovered) else 0


if __name__ == "__main__":
    sys.exit(main())
