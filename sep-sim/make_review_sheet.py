#!/usr/bin/env python3
"""S1 human review sheet: N (default 30) seeded-random labelled samples -> one Markdown file.

Each item shows what the label judge saw (goal text = scenario_text, the conversation so far) and
what it answered (status, unmet list), plus blanks for the reviewer's verdict. Only rows with a
parsed label are eligible. The header records seed and file SHA256s so the sheet is reproducible.
"""
import argparse
import random

import judge_metrics as JM


def choose(labels, samples, n, seed):
    eligible = sorted(s["id"] for s in samples if s["id"] in labels)
    if n > len(eligible):
        raise SystemExit("only %d labelled samples, asked for %d" % (len(eligible), n))
    return sorted(random.Random(seed).sample(eligible, n))


def quote(text):
    return "\n".join("> " + (line if line.strip() else "") for line in str(text).splitlines() or [""])


def render(items, header):
    out = ["# Goal-judge label review", "", header, "",
           "For each item: read WHAT THIS PERSON WANTS and the conversation, then mark whether the "
           "label is right. SATISFIED = everything they want has been adequately addressed; "
           "PARTIAL = some but not all; NOT = nothing yet. A need counts as addressed only when the "
           "assistant pointed to concrete datasets or sources that plausibly meet it.", ""]
    for k, (s, lab) in enumerate(items, 1):
        out += ["---", "", "## %d. `%s`" % (k, s["id"]), "",
                "source: %s | exchange t = %s | conversation_id: %s" % (s["source"], s["t"], s["conversation_id"]),
                "", "### What this person wants", "", quote(" ".join(s["scenario_text"].split())), "",
                "### Conversation so far", ""]
        for i, u in enumerate(s["hist_u"]):
            out += ["**USER (%d):**" % (i + 1), "", quote(u), ""]
            if i < len(s["hist_a"]):
                out += ["**ASSISTANT (%d):**" % (i + 1), "", quote(s["hist_a"][i]), ""]
        out += ["### Label", "", "- status: **%s**" % lab["status"]]
        if lab["unmet"]:
            out += ["- unmet:"] + ["  - %s" % u for u in lab["unmet"]]
        else:
            out += ["- unmet: (none)"]
        out += ["", "### Reviewer", "", "- [ ] label correct", "- [ ] label wrong -> correct status: ______",
                "- notes: ", ""]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260925)
    a = ap.parse_args()
    labels, stats = JM.load_labels(a.labels)
    samples = JM.load_samples(a.samples)
    by_id = {s["id"]: s for s in samples}
    ids = choose(labels, samples, a.n, a.seed)
    header = ("seed %d; n %d of %d parsed labels; samples sha256 %s; labels sha256 %s"
              % (a.seed, a.n, stats["parsed"], JM.sha256_file(a.samples), JM.sha256_file(a.labels)))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(render([(by_id[i], labels[i]) for i in ids], header))
    print("wrote", a.out, "items", len(ids))


if __name__ == "__main__":
    main()
