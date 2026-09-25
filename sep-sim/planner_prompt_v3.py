"""Planner prompt v3 and Speaker block v3 (audit defects D2, D3; AUDIT_AND_PLAN_v2 §3).

Built ON TOP of the frozen sepsim.planner_prompt so that everything not named here stays
byte-identical to the original:

  * PP.user_prompt is called with ledger=None and agenda_view=None, so the original code
    renders every other part (persona brief, goal topic, previous state, length band,
    turn line, conversation) exactly as before.
  * Inserted right before "\\n\\nTHEY HAVE SENT": a facts block holding only
    "- turns so far" and the Planner's own gain line (the Task-1-only ledger lines --
    useful/unhelpful counts, best offered, repeated offer, conditions -- are dropped, D2),
    followed by the GOAL STATUS block from the goal-satisfaction judge (replaces the
    lexical agenda "WHAT THEY STILL WANT", D3).
  * In the persona brief, the rule-derived parenthetical of the "gives up" line is removed
    (the unhelpful counter it refers to no longer exists in the prompt).
The Speaker block uses PP.render_block with the judge's unmet list in the "- pending:"
slot instead of the lexical agenda.
"""
from __future__ import annotations

from sepsim import persona as P
from sepsim import planner_prompt as PP

FACTS_HEAD = "\n\nWHAT HAS ACTUALLY HAPPENED (counted, not judged)\n"
GOAL_HEAD = "\n\nGOAL STATUS (assessed from the conversation by a separate model)\n"
ANCHOR = "\n\nTHEY HAVE SENT "
STATUSES = ("SATISFIED", "PARTIAL", "NOT", "NOT ASSESSED", "UNKNOWN")
FIRST_TURN_UNMET = "the whole request (nothing has been answered yet)"


def facts_v3(ledger):
    """Only counts that are true in Task 2: turns, and the Planner's own gain estimates."""
    lines = ["- turns so far: %d" % ledger.turns]
    mv = ledger.marginal_value()
    if mv:
        lines.append("- gain this turn %.2f against a session average of %.2f" % (mv[0], mv[1]))
    return "\n".join(lines)


def unmet_text(goal_status):
    if goal_status is None or goal_status.get("status") == "NOT ASSESSED":
        return FIRST_TURN_UNMET
    items = [u for u in (goal_status.get("unmet") or []) if u]
    return "; ".join(items) if items else "(nothing identified)"


def goal_block(goal_status):
    st = (goal_status or {}).get("status", "NOT ASSESSED")
    if st not in STATUSES:
        raise ValueError("unknown goal status %r" % st)
    return GOAL_HEAD + "- status: %s\n- still unmet: %s" % (st, unmet_text(goal_status))


def gives_up_suffix(scenario):
    per = (scenario or {}).get("persona") or {}
    return " (so about %d unhelpful replies before frustration governs)" % P.patience_budget(per)


def user_prompt_v3(scenario, prev_block, hist_u, hist_a, turn, ledger, goal_status,
                   prev_ann=None, p_end=None):
    if ledger is None:
        raise ValueError("v3 needs the ledger for the turn count")
    base = PP.user_prompt(scenario, prev_block, hist_u, hist_a, turn, ledger=None,
                          prev_ann=prev_ann, agenda_view=None, p_end=p_end)
    # persona brief: drop the rule-derived parenthetical of the gives-up line (first
    # occurrence only, inside the WHO THEY ARE section)
    suffix = gives_up_suffix(scenario)
    head, sep, rest = base.partition("\n\nWHAT THEY CAME FOR\n")
    if not sep:
        raise ValueError("unexpected prompt layout")
    lines = head.split("\n")
    hit = [i for i, l in enumerate(lines) if l.startswith("- gives up: ") and l.endswith(suffix)]
    if len(hit) != 1:
        raise ValueError("gives-up line not found exactly once")
    lines[hit[0]] = lines[hit[0]][: -len(suffix)]
    base = "\n".join(lines) + sep + rest
    static = base.split("\n\nTHE CONVERSATION SO FAR\n", 1)[0]
    if static.count(ANCHOR) != 1:
        raise ValueError("turn anchor must occur exactly once in the static prefix")
    i = base.find(ANCHOR)
    return base[:i] + FACTS_HEAD + facts_v3(ledger) + goal_block(goal_status) + base[i:]


def speaker_block_v3(fields, goal_status):
    return PP.render_block(fields, (unmet_text(goal_status), ""))
