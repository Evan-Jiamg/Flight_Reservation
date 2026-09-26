"""Pure-Python helpers shared by the judge scripts (no torch; unit-tested in test_judge_pipeline.py).

  * files / labels:   sha256_file, load_samples, load_labels (parsed labels only, duplicates checked)
  * sampling:         stratified_sample (largest-remainder allocation, seeded, order-independent)
  * leakage:          fold_of, assert_no_leak, check_manifest
  * classification:   confusion, per_class_f1, macro_f1, accuracy, brier, nll (3-class status)
  * timing:           auc (rank-based, ties = 1/2), timing_points, timing_metrics
                      (overall / pooled pair-weighted within-turn / turn-index-only AUC),
                      session_bootstrap (percentile CIs, sessions resampled with replacement)
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict

CLASSES = ("SATISFIED", "PARTIAL", "NOT")


# ---- files ----------------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_samples(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_labels(path):
    """-> (labels {id: {"status","unmet"}} for parsed rows, stats). A resumable label file may repeat
    an id; repeats must agree, otherwise the file is rejected (never pick one silently)."""
    labels, stats = {}, Counter()
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        stats["rows"] += 1
        if not (r.get("parse_ok") and r.get("status") in CLASSES):
            stats["unparsed"] += 1
            continue
        lab = {"status": r["status"], "unmet": list(r.get("unmet") or [])}
        if r["id"] in labels:
            stats["duplicates"] += 1
            if labels[r["id"]] != lab:
                raise ValueError("conflicting duplicate label rows for %s" % r["id"])
            continue
        labels[r["id"]] = lab
    stats["parsed"] = len(labels)
    return labels, dict(stats)


# ---- sampling -------------------------------------------------------------------------------------
def allocate(sizes, n):
    """Largest-remainder proportional allocation of n over strata {name: size} (ties by name)."""
    total = sum(sizes.values())
    if n > total:
        raise ValueError("cannot draw %d from %d" % (n, total))
    quota = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(math.floor(q)) for k, q in quota.items()}
    rest = n - sum(alloc.values())
    for k in sorted(sizes, key=lambda k: (-(quota[k] - alloc[k]), k))[:rest]:
        alloc[k] += 1
    return alloc


def stratified_sample(ids_by_stratum, n, seed):
    """Seeded stratified sample; result does not depend on input order. -> sorted list of ids."""
    strata = {k: sorted(set(v)) for k, v in ids_by_stratum.items()}
    alloc = allocate({k: len(v) for k, v in strata.items()}, n)
    out = []
    for k in sorted(strata):
        rng = random.Random("%s|%s" % (seed, k))
        out += rng.sample(strata[k], alloc[k])
    return sorted(out)


# ---- leakage --------------------------------------------------------------------------------------
def fold_of(splits, fold):
    hit = [f for f in splits["folds"] if int(f["fold"]) == int(fold)]
    if len(hit) != 1:
        raise ValueError("fold %s not found exactly once in the split file" % fold)
    return hit[0]


def assert_no_leak(train_cids, fold_split):
    """Training conversation ids must lie in the fold's train list and avoid everything forbidden."""
    train_cids = set(train_cids)
    outside = train_cids - set(fold_split["train"])
    if outside:
        raise AssertionError("training ids outside splits[f].train: %s" % sorted(outside)[:5])
    bad = train_cids & set(fold_split["forbidden_for_training"])
    if bad:
        raise AssertionError("training ids in forbidden_for_training: %s" % sorted(bad)[:5])
    bad = train_cids & (set(fold_split.get("validation_all", [])) | set(fold_split.get("test_all", []))
                        | set(fold_split["validation"]) | set(fold_split["test"]))
    if bad:
        raise AssertionError("training ids in validation/test: %s" % sorted(bad)[:5])


def check_manifest(manifest, fold, split_sha, eval_cids):
    """A judge may score a split only if it was trained for this fold on this split file and never
    saw any of the evaluated conversations."""
    if int(manifest["fold"]) != int(fold):
        raise AssertionError("judge trained for fold %s, evaluating fold %s" % (manifest["fold"], fold))
    if manifest["split_file_sha256"] != split_sha:
        raise AssertionError("judge trained on a different split file")
    bad = set(manifest["train_scenarios"]) & set(eval_cids)
    if bad:
        raise AssertionError("judge trained on evaluated conversations: %s" % sorted(bad)[:5])


# ---- 3-class status metrics ------------------------------------------------------------------------
def confusion(gold, pred):
    return Counter(zip(gold, pred))


def per_class_f1(gold, pred):
    conf = confusion(gold, pred)
    out = {}
    for c in CLASSES:
        tp = conf[(c, c)]
        fp = sum(v for (g, p), v in conf.items() if p == c and g != c)
        fn = sum(v for (g, p), v in conf.items() if g == c and p != c)
        out[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return out


def macro_f1(gold, pred):
    """Mean F1 over classes that occur in gold or pred (a prediction UNKNOWN counts as wrong)."""
    f = [v for v in per_class_f1(gold, pred).values() if v is not None]
    return sum(f) / len(f) if f else None


def accuracy(gold, pred):
    return sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else None


def brier(gold, probs):
    """Multi-class Brier: mean over samples of sum_c (p_c - 1[c = gold])^2 (0 best, 2 worst)."""
    if not gold:
        return None
    return sum(sum((p[c] - (g == c)) ** 2 for c in CLASSES) for g, p in zip(gold, probs)) / len(gold)


def nll(gold, probs, floor=1e-12):
    if not gold:
        return None
    return -sum(math.log(max(p[g], floor)) for g, p in zip(gold, probs)) / len(gold)


# ---- stop-timing AUCs -----------------------------------------------------------------------------
def _midranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie), via Mann-Whitney ranks."""
    if not pos or not neg:
        return None
    r = _midranks(list(pos) + list(neg))
    rp = sum(r[:len(pos)])
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def timing_points(sessions, mode="last"):
    """sessions: {session_id: {"K": n_exchanges, "scores": {t: score}}} (t = 1..K).

    mode 'last':     positive = t == K (state after the reply to the human's final message),
                     negatives = t < K.
    mode 'prefinal': positive = t == K-1, negatives = t < K-1, t == K dropped (for corpora whose
                     final user message is a closing: the decision was taken after reply K-1).
    -> list of (session_id, t, score, is_pos)."""
    pts = []
    for sid, s in sessions.items():
        K = int(s["K"])
        pos_t = K if mode == "last" else K - 1
        if mode not in ("last", "prefinal"):
            raise ValueError(mode)
        for t, sc in s["scores"].items():
            t = int(t)
            if t > pos_t or t < 1:
                continue
            pts.append((sid, t, float(sc), t == pos_t))
    return pts


def timing_metrics(points):
    pos = [p for p in points if p[3]]
    neg = [p for p in points if not p[3]]
    by_t = defaultdict(lambda: ([], []))
    for _, t, sc, y in points:
        by_t[t][0 if y else 1].append(sc)
    num = den = 0.0
    for t, (p, n) in by_t.items():
        if p and n:
            num += auc(p, n) * len(p) * len(n)
            den += len(p) * len(n)
    return {"n_sessions": len({p[0] for p in points}), "n_pos": len(pos), "n_neg": len(neg),
            "overall_auc": auc([p[2] for p in pos], [p[2] for p in neg]),
            "turn_only_auc": auc([p[1] for p in pos], [p[1] for p in neg]),
            "within_turn_auc": num / den if den else None, "within_turn_pairs": int(den)}


def _pct(v, q):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def session_bootstrap(points, n_boot, seed, alpha=0.05):
    """Percentile CIs of timing_metrics' AUCs and of (overall - turn_only), sessions resampled."""
    by_s = defaultdict(list)
    for p in points:
        by_s[p[0]].append(p)
    sids = sorted(by_s)
    rng = random.Random(seed)
    keys = ("overall_auc", "turn_only_auc", "within_turn_auc", "overall_minus_turn_only")
    draws = {k: [] for k in keys}
    undefined = Counter()
    for _ in range(n_boot):
        pts = []
        for j, sid in enumerate(rng.choice(sids) for _ in sids):
            pts += [("%s#%d" % (sid, j), t, sc, y) for _, t, sc, y in by_s[sid]]
        m = timing_metrics(pts)
        if m["overall_auc"] is not None and m["turn_only_auc"] is not None:
            m["overall_minus_turn_only"] = m["overall_auc"] - m["turn_only_auc"]
        else:
            m["overall_minus_turn_only"] = None
        for k in keys:
            if m[k] is None:
                undefined[k] += 1
            else:
                draws[k].append(m[k])
    out = {k: {"lo": _pct(draws[k], alpha / 2), "hi": _pct(draws[k], 1 - alpha / 2),
               "n_defined": len(draws[k])} for k in keys}
    out.update({"n_boot": n_boot, "seed": seed, "undefined": dict(undefined)})
    return out
