"""Task 1 (teacher-forced) stop metrics over Task2Env.run_task1 outputs. Pure python.

For every real conversation of n user messages, turn t = 1..n, the Planner's end_session
(`ended_planner`) says "message t is the person's last". Under M2 (user decision D7) that decision is
the benchmark's END flag one row later: END at row t+1, or at the K+1 probe for t = n. A blank Speaker
message at row t is END at row t itself (`speaker_blank`; at the probe `k1_speaker_blank`).

Benchmark view (the definition in /tmp2/mzjiang_usersim/k1/reduce_k1.py, which reproduces the published
arms: TP = K+1 ends, FP = END flags on the n real rows, recall = TP / conversations):
  term_f1            2TP / (2TP + FP + FN)
  premature_end_rate FP / real rows (the benchmark's premature_end_rate)
  premature          share of conversations with an END flag on a real row
  k1_end_rate        TP / conversations
Raw decision view (diagnostic only): decision_f1 of end_session against real_final.
  unparsed_rate      unreadable plans / turns
These are the validation-time stop proxies used for checkpoint selection; the benchmark's own Task 1
metrics are computed once at the end with its tools.
"""
from __future__ import annotations


def bench_flags(rows, k1_blank=False):
    """M2 END flags of the n real rows and of the K+1 probe."""
    n = len(rows)
    flags = []
    for i, r in enumerate(rows):
        prev = bool(rows[i - 1]["ended_planner"]) if i >= 1 else False
        flags.append(prev or bool(r.get("speaker_blank")))
    k1 = bool(rows[n - 1]["ended_planner"]) or bool(k1_blank)
    return flags, k1


def task1_stop_metrics(convs):
    tp = fp = fn = rows_n = unparsed = premature = capped = 0
    dtp = dfp = dfn = 0
    for c in convs:
        rows = c["turns"] if isinstance(c, dict) else c
        k1_blank = bool(c.get("k1_speaker_blank")) if isinstance(c, dict) else False
        n = len(rows)
        if n == 0:
            raise ValueError("conversation without turns")
        if [r["t"] for r in rows] != list(range(1, n + 1)) or sum(bool(r["real_final"]) for r in rows) != 1 \
                or not rows[-1]["real_final"]:
            raise ValueError("turns must be 1..n with exactly the last one final")
        flags, k1 = bench_flags(rows, k1_blank)
        tp += k1
        fn += not k1
        fp += sum(flags)
        premature += any(flags)
        rows_n += n
        for r in rows:
            unparsed += bool(r.get("planner_unparsed"))
            capped += bool(r.get("emitted_capped"))
            e, f = bool(r["ended_planner"]), bool(r["real_final"])
            dtp += e and f
            dfp += e and not f
            dfn += f and not e
    k = len(convs)
    if k == 0:
        raise ValueError("no conversations")
    den, dden = 2 * tp + fp + fn, 2 * dtp + dfp + dfn
    return {"n_conversations": k, "n_turns": rows_n, "end_mapping": "M2",
            "term_f1": (2 * tp / den) if den else 0.0,
            "premature_end_rate": fp / rows_n, "premature": premature / k, "k1_end_rate": tp / k,
            "decision_f1": (2 * dtp / dden) if dden else 0.0,
            "unparsed_rate": unparsed / rows_n, "n_emitted_capped_turns": capped}


def task1_prob_metrics(points):
    """v16 (user 2026-09-28) continuous Task 1 metric from teacher-forced end probabilities. points: one per
    decision point (real turn t >= 2) {"real_final": bool, "p_end": float in [0, 1]} (an invalid decision point
    carries p_end 0: the benchmark reads an unparsed plan as "not ending").
      bal_p   = 1/2 mean_{final} p_end + 1/2 mean_{earlier} (1 - p_end)   (earlier part dropped when absent)
      auc     = P(p_end at a final point > p_end at an earlier point), ties 1/2 (None without both kinds)
      nll     = mean -log p(true label) over the VALID points only (v17 B5, user 2026-09-30: an unparsed / capped
                plan or one whose value tokens could not be located has no probability to score; they are counted in
                n_invalid), p clipped to [1e-6, 1 - 1e-6]; None without a valid point
      logloss = nll (the v16 name, kept as an alias with the same value)"""
    import math
    if not points:
        raise ValueError("no decision points")
    fin = [float(x["p_end"]) for x in points if x["real_final"]]
    mid = [float(x["p_end"]) for x in points if not x["real_final"]]
    for p in fin + mid:
        if not (0.0 <= p <= 1.0):
            raise ValueError("p_end %r outside [0, 1]" % p)
    if not fin:
        raise ValueError("no final decision point")
    bal = 0.5 * sum(fin) / len(fin) + 0.5 * sum(1 - p for p in mid) / len(mid) if mid else sum(fin) / len(fin)
    auc = None
    if mid:
        pairs = [(1.0 if a > b else 0.5 if a == b else 0.0) for a in fin for b in mid]
        auc = sum(pairs) / len(pairs)
    eps = 1e-6
    ll = [-math.log(min(1 - eps, max(eps, float(x["p_end"]) if x["real_final"] else 1 - float(x["p_end"]))))
          for x in points if x.get("valid", True)]
    nll = sum(ll) / len(ll) if ll else None
    return {"bal_p": bal, "auc": auc, "nll": nll, "logloss": nll, "n_points": len(points), "n_final": len(fin),
            "n_valid": len(ll), "n_invalid": sum(1 for x in points if not x.get("valid", True))}


def within_tolerance(metrics, base, tol):
    """No-degradation check against a base (e.g. the untrained policy): term_f1 may not drop, and
    premature may not rise, by more than tol (declared in advance)."""
    return (metrics["term_f1"] >= base["term_f1"] - tol) and (metrics["premature"] <= base["premature"] + tol)
