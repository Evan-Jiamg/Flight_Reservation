"""Task 1 (teacher-forced) stop metrics over Task2Env.run_task1 outputs. Pure python.

For every real conversation of n user messages and every turn t = 1..n, the Planner's end_session
(`ended_planner`) is a prediction of "message t is the person's last" (`real_final`, i.e. t == n).

  term_f1          F1 of the positive class (end) over all turns
  premature        share of conversations whose FIRST predicted end is before the real last message
  final_recall     share of conversations where the Planner ends at the real last message
  false_end_rate   predicted ends / non-final turns
  unparsed_rate    unreadable plans / turns
These are the validation-time stop proxies used for checkpoint selection; the benchmark's own Task 1
metrics are computed once at the end with its tools.
"""
from __future__ import annotations


def task1_stop_metrics(convs):
    tp = fp = fn = nonfinal = turns = unparsed = premature = final_hit = 0
    for c in convs:
        rows = c["turns"] if isinstance(c, dict) else c
        n = len(rows)
        if n == 0:
            raise ValueError("conversation without turns")
        if [r["t"] for r in rows] != list(range(1, n + 1)) or sum(bool(r["real_final"]) for r in rows) != 1 \
                or not rows[-1]["real_final"]:
            raise ValueError("turns must be 1..n with exactly the last one final")
        first_end = next((r["t"] for r in rows if r["ended_planner"]), None)
        premature += first_end is not None and first_end < n
        final_hit += bool(rows[-1]["ended_planner"])
        for r in rows:
            turns += 1
            unparsed += bool(r.get("planner_unparsed"))
            e, f = bool(r["ended_planner"]), bool(r["real_final"])
            tp += e and f
            fp += e and not f
            fn += f and not e
            nonfinal += not f
    k = len(convs)
    if k == 0:
        raise ValueError("no conversations")
    den = 2 * tp + fp + fn
    return {"n_conversations": k, "n_turns": turns,
            "term_f1": (2 * tp / den) if den else 0.0,
            "premature": premature / k, "final_recall": final_hit / k,
            "false_end_rate": fp / nonfinal if nonfinal else 0.0,
            "unparsed_rate": unparsed / turns}


def within_tolerance(metrics, base, tol):
    """No-degradation check against a base (e.g. the untrained policy): term_f1 may not drop, and
    premature may not rise, by more than tol (declared in advance)."""
    return (metrics["term_f1"] >= base["term_f1"] - tol) and (metrics["premature"] <= base["premature"] + tol)
