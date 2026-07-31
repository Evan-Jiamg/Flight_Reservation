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

import sys as _s
ROOT = _s.argv[1] if len(_s.argv) > 1 and not _s.argv[1].startswith("-") else (
    "/home/neil/Information_Management_Project/Echo-Chamber-Simulation/"
    "Hybrid-Network")
DOCS = [os.path.join(ROOT, n) for n in ("README.md", "Workflow.md")
        if os.path.exists(os.path.join(ROOT, n))]

STDLIB = set(sys.stdlib_module_names) | {"__future__"}
LOCAL = {"core", "agent", "numeric_agent", "prompt", "scorer", "utils",
         "model", "convergence", "make_official_figs", "hcog_paths"}
# import name -> distribution name
# pandas reads parquet through pyarrow without importing it, so an import
# scan will always call it unused.
INDIRECT = {"pyarrow"}
DIST = {"sklearn": "scikit-learn", "community": "python-louvain",
        "sentence_transformers": "sentence-transformers",
        "umap": "umap-learn", "cuml": "cuml-cu12",
        "cupy": "cupy-cuda12x",
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

    # python3 <path> and bash <path> inside fenced blocks. A preceding `cd`
    # changes what the following paths are relative to; without tracking it,
    # every command in a block that starts "cd Hybrid-Network" reads as missing.
    cwd = ""
    for line in text.splitlines():
        t = line.strip()
        m = re.match(r"^cd\s+([\w./\-]+)", t)
        if m:
            cwd = os.path.normpath(os.path.join(cwd, m.group(1)))
            continue
        if t.startswith("```"):
            cwd = ""
            continue
        m = re.match(r"^(?:python3|bash|\./)?\s*([\w./\-]+\.(?:py|sh))", t)
        if not m:
            continue
        p = m.group(1)
        if not os.path.exists(os.path.join(ROOT, cwd, p)):
            bad.append(("command", os.path.join(cwd, p)))

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
    skip = {".git", ".gitignore", "ops", "__pycache__", "README.md",
            "Workflow.md"}
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
        # data/ holds two generator scripts as well as the inputs, so it is
        # walked; only its pure-data subdirectories are skipped.
        dns[:] = [d for d in dns
                  if d not in ("ops", "__pycache__", ".git", "results",
                               "experiments", "logs", "networks", "agents",
                               "lexicons", "figures", "summaries")]
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
    # A pin nothing imports is dead weight, and dead weight in a dependency
    # list is the kind a reviewer installs and then wonders about.
    unused = sorted(n for n in listed if n not in used and n not in INDIRECT)
    return missing, sorted(listed), sorted(optional), unused


def main():
    text = chr(10).join(io.open(p, encoding="utf-8").read() for p in DOCS)
    print("auditing: " + ", ".join(os.path.basename(p) for p in DOCS))
    print()

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
    if not os.path.exists(os.path.join(ROOT, "requirements.txt")):
        # The root pins nothing itself; each stage carries its own.
        print("  no requirements.txt at this level (each stage pins its own)")
        return 1 if (bad or miss) else 0
    uncovered, listed, optional, unused = audit_requirements()
    if uncovered:
        for n, f in uncovered:
            print("  NOT LISTED: %-18s (imported by %s)" % (n, f))
    else:
        print("  every third-party import is pinned")
    if unused:
        for n in unused:
            print("  PINNED BUT UNUSED: %s" % n)
    print("  required: " + ", ".join(sorted(listed)))
    print("  optional: " + ", ".join(sorted(optional)))

    return 1 if (bad or miss or uncovered or unused) else 0


if __name__ == "__main__":
    sys.exit(main())
