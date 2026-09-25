"""Planner prompt v3, system prompt v3, read_plan v3 and Speaker block v3.

Audit defects D1 (stop override), D2 (Task-1-only ledger facts), D3 (lexical agenda) and the
"no hard-coded rules" requirement (length band + clamp, stop-rule weights/combination rule).
Summary of every change against the frozen sepsim prompt (everything else byte-identical):
  system prompt  rules block -> reasons as vocabulary only; ABOUT LENGTH -> Planner decides;
                 JSON gains "end_session"; "length_words" no longer tied to a band.
  user prompt    persona gives-up parenthetical removed; HOW LONG THEY WRITE band removed;
                 ledger block -> turns + own gain only; agenda -> GOAL STATUS from the judge.
  read_plan      override OFF (SEPSIM_ACT_PRIOR=nostopclobber); stop = end_session; length =
                 the Planner's own number, never clamped.
  speaker block  "- pending:" carries the judge's unmet list.

Original notes on the user-prompt construction:

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

from sepsim import acts, state, stopping
from sepsim import persona as P
from sepsim import planner_prompt as PP

# ---------------------------------------------------------------- system prompt v3
# No hard-coded decision rules (user requirement, 2026-09-25):
#  * the stop decision is the Planner's own boolean `end_session`, argued both ways first; the
#    read_plan override that turned a named rule into a forced Complete act is OFF
#    (SEPSIM_ACT_PRIOR=nostopclobber); named reasons remain only as vocabulary, without the
#    strong/weak weights and without the "met first governs" combination rule;
#  * length is the Planner's own decision from the person's style and their messages: no word
#    band in the prompt and no clamp to a band after parsing.
ACT_PRIOR_V3 = "nostopclobber"

OLD_LENGTH_PARA = (
    "ABOUT LENGTH. You will be given the word band this person writes in. Place THIS turn inside that\n"
    "band and say why in a few words. A turn that closes things is short; a turn that opens them has\n"
    "to carry the whole requirement. Do not drift to the middle of the band every time.")
NEW_LENGTH_PARA = (
    "ABOUT LENGTH. Decide how many words THIS turn has from how this person writes: their stated\n"
    "style and their own messages above. Say why in a few words. A turn that closes things is short;\n"
    "a turn that opens them has to carry the whole requirement.")
OLD_STOP_FIELD = ' "stop_rule": "<the named rule you are applying, or none>",\n'
NEW_STOP_FIELDS = (' "stop_rule": "<the named reason that fits your decision, or none>",\n'
                   ' "end_session": <true if this person leaves now and writes nothing more, else false>,\n')
OLD_LEN_FIELD = ' "length_words": <a number inside the band you were given>,\n'
NEW_LEN_FIELD = ' "length_words": <how many words this turn has>,\n'
# The original JSON has no "patience" field, so state.render always wrote "patience full" into the
# Planner's own previous state and into the Speaker block -- a constant, false fact (same class as
# D2). v3 asks the Planner for it; state.from_plan already reads and validates the field.
OLD_AFFECT_FIELD = ' "affect": "<their mood in a few words>",\n'
NEW_AFFECT_FIELDS = (' "affect": "<their mood in a few words>",\n'
                     ' "patience": "<full | wearing | thin | spent: how much patience this person has left>",\n')
# D10: in Task 2 the USER messages are simulated, so "typed by the real person" is false.
OLD_REAL_PERSON = ("Every USER message you are shown was typed by the real person. Their wording, punctuation and\n"
                   "length are evidence, not a simulation of it.")
NEW_REAL_PERSON = ("The USER messages below are this person's messages so far. Their wording, punctuation and\n"
                   "length show how this person writes.")


def rules_block_v3():
    """Named reasons as vocabulary only: no weights, no combination rule, no performance claims."""
    lines = ["Reasons people give for stopping. Use them to put your case into words; name the one",
             "that fits your decision, or none:"]
    for k, (_exit, _weight, desc) in stopping.RULES.items():
        first = desc.split(". ")[0].rstrip(".") + "."
        lines.append("    %-22s %s" % (k, first))
    lines += [
        "",
        "    Before deciding, put both cases in one sentence each: the case for leaving",
        "    now, and the case for one more turn. Then decide end_session yourself:",
        "    true means this person leaves now and writes nothing more; false means they",
        "    write again, and the acts you name are for that next message. People do not",
        "    apply fixed criteria; they stop on a feeling of good enough. So judge THIS",
        "    person, with THIS patience, at THIS point.",
    ]
    return "\n".join(lines)


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("expected exactly one occurrence of %r" % old[:60])
    return text.replace(old, new, 1)


def system_prompt_v3():
    if ACT_PRIOR_V3 not in acts.prior_mode() or "dispute" in acts.prior_mode():
        raise RuntimeError("v3 requires SEPSIM_ACT_PRIOR=nostopclobber (and no dispute block)")
    s = PP.system_prompt()
    s = _replace_once(s, stopping.rules_block(), rules_block_v3())
    s = _replace_once(s, OLD_LENGTH_PARA, NEW_LENGTH_PARA)
    s = _replace_once(s, OLD_STOP_FIELD, NEW_STOP_FIELDS)
    s = _replace_once(s, OLD_LEN_FIELD, NEW_LEN_FIELD)
    s = _replace_once(s, OLD_REAL_PERSON, NEW_REAL_PERSON)
    s = _replace_once(s, OLD_AFFECT_FIELD, NEW_AFFECT_FIELDS)
    return s


def read_plan_v3(raw, turn, scenario, rng, ledger):
    """PP.read_plan with the override OFF, then the Planner's own length and end decision.

    Returns (fields, diag, end_session). end_session is True only for a JSON true (or the string
    "true"); anything else is False and is counted as unparsed in diag. No value is repaired or
    clamped: a missing/invalid length leaves no length target (the selector then keeps its default).
    """
    if ACT_PRIOR_V3 not in acts.prior_mode():
        raise RuntimeError("stop override must be OFF for v3")
    fields, diag = PP.read_plan(raw, turn, scenario, rng, ledger)
    if fields is None:
        return None, diag, False
    d = state.json_of(raw) or {}
    es = d.get("end_session")
    end = es is True or (isinstance(es, str) and es.strip().lower() == "true")
    diag["end_session_raw"] = es
    diag["end_session_valid"] = isinstance(es, bool) or (isinstance(es, str) and es.strip().lower() in ("true", "false"))
    try:
        lw = int(round(float(d.get("length_words"))))
    except (TypeError, ValueError):
        lw = None
    if lw is not None and lw > 0:
        fields["length_words"] = lw
    else:
        fields.pop("length_words", None)
    diag["length_from_planner"] = lw
    diag["length_clamped"] = False
    diag["complete_act_without_end"] = bool(state.ends_session(fields)) and not end
    return fields, diag, end

FACTS_HEAD = "\n\nWHAT HAS ACTUALLY HAPPENED (counted, not judged)\n"
GOAL_HEAD = "\n\nGOAL STATUS (assessed from the conversation by a separate model)\n"
ANCHOR = "\n\nTHEY HAVE SENT "
STATUSES = ("SATISFIED", "PARTIAL", "NOT", "NOT ASSESSED", "UNKNOWN")
FIRST_TURN_UNMET = "the whole request (nothing has been answered yet)"
UNKNOWN_UNMET = "(not available this turn: the assessment could not be read)"


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
    if goal_status.get("status") == "UNKNOWN":
        return UNKNOWN_UNMET          # the judge's reply did not parse: say so, never "nothing unmet"
    items = [u for u in (goal_status.get("unmet") or []) if u]
    return "; ".join(items) if items else "(nothing identified)"


def goal_block(goal_status):
    st = (goal_status or {}).get("status", "NOT ASSESSED")
    if st not in STATUSES:
        raise ValueError("unknown goal status %r" % st)
    return GOAL_HEAD + "- status: %s\n- still unmet: %s" % (st, unmet_text(goal_status))


def band_block(scenario):
    """The exact 'HOW LONG THEY WRITE' block PP.user_prompt renders for this persona."""
    per = (scenario or {}).get("persona") or {}
    label, lo, hi, why = P.length_guidance(per, "Reveal")
    return ("\n\nHOW LONG THEY WRITE\n- band: %s, roughly %d to %d words (%s)\n"
            "- the band shifts with the move: closing turns are much shorter, "
            "opening turns longer" % (label, lo, hi, why))


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
    # drop the persona word band (a fixed style->words mapping); the Planner decides length itself
    base = _replace_once(base, band_block(scenario), "")
    static = base.split("\n\nTHE CONVERSATION SO FAR\n", 1)[0]
    if static.count(ANCHOR) != 1:
        raise ValueError("turn anchor must occur exactly once in the static prefix")
    i = base.find(ANCHOR)
    return base[:i] + FACTS_HEAD + facts_v3(ledger) + goal_block(goal_status) + base[i:]


def speaker_block_v3(fields, goal_status):
    return PP.render_block(fields, (unmet_text(goal_status), ""))
