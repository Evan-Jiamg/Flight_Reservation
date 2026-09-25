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

import os

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
# E1.6 SEPSIM_ACT_FULL (kept in Final): one act_distribution entry per move, each with its own
# length_words, so the length that rides along is the length of the act actually drawn. v3 keeps
# the per-act length and removes only the band it was tied to.
OLD_ACT_FULL_LEN = "and its own length_words inside that move's band."
NEW_ACT_FULL_LEN = "and its own length_words: how many words this turn has if they made that move."


def act_full():
    return os.environ.get("SEPSIM_ACT_FULL", "0") == "1"


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
    if act_full():
        s = _replace_once(s, OLD_ACT_FULL_LEN, NEW_ACT_FULL_LEN)
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
    if turn == 1 and end:
        # Task 2 definition, not a stop rule: the session exists because the person opened it, so
        # they write at least the first message. The Planner's raw answer is kept in diag.
        diag["end_session_t1_ignored"] = True
        end = False
    diag["end_session_raw"] = es
    diag["end_session_valid"] = isinstance(es, bool) or (isinstance(es, str) and es.strip().lower() in ("true", "false"))
    lw_src, lw_val = "top", d.get("length_words")
    if act_full():
        # the drawn act's own entry, matched exactly as E1.6 read_plan matches it (never clamped)
        for e in (d.get("act_distribution") or []):
            if isinstance(e, dict) and e.get("length_words") is not None and \
                    acts.normalise(str(e.get("move", "")), str(e.get("act", ""))) == (fields["move"], fields["act"]):
                lw_src, lw_val = "act_entry", e.get("length_words")
                break
    try:
        lw = int(round(float(lw_val)))
    except (TypeError, ValueError):
        lw = None
    diag["length_source"] = lw_src
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
    band = ("\n\nHOW LONG THEY WRITE\n- band: %s, roughly %d to %d words (%s)\n"
            "- the band shifts with the move: closing turns are much shorter, "
            "opening turns longer" % (label, lo, hi, why))
    if act_full():
        # E1.6 appends one band row per move under ACT_FULL; it goes together with the band
        rows = []
        for mv in ("Disclose", "Reveal", "Inquire", "Navigate", "Note", "Complete"):
            _, l2, h2, _ = P.length_guidance(per, mv)
            rows.append("- %s: %d to %d words" % (mv, l2, h2))
        band += "\n" + "\n".join(rows)
    return band


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


# ================================================================ pend (user direction, 2026-09-25)
# Planner + Ditto + Selector, no goal judge. The Planner itself judges whether the goal is met
# (goal_met / still_wanted, from the assistant's replies) and decides end_session. end_session=true
# means the message being planned is the LAST one: Ditto writes it (a close, which need not thank
# anyone) and the episode ends after it -- E1/E1.6 PLANNER_END semantics, not a silent exit.
# Built on v3 (all v3 fixes: override off, no band/clamp, Task-2-true facts only, patience, D10,
# no truncation) with: GOAL STATUS block -> none; goal topic -> topic + context (D4);
# "- pending:" in the Speaker block -> the Planner's own still_wanted (replaces the lexical agenda, D3).

GOAL_MET = ("yes", "partly", "no")
NEW_STOP_FIELDS_PEND = (
    ' "goal_met": "<yes | partly | no: judged from the assistant\'s replies so far, has this person got what they came for?>",\n'
    ' "still_wanted": "<in a few words, what they still want from the assistant, or nothing>",\n'
    ' "stop_rule": "<the named reason that fits your decision, or none>",\n'
    ' "end_session": <true if the message you are planning is their LAST one (they write it, then leave), else false>,\n')


def rules_block_pend():
    lines = rules_block_v3().split("\n")
    i = lines.index("    Before deciding, put both cases in one sentence each: the case for leaving")
    return "\n".join(lines[:i] + [
        "    Before deciding, put both cases in one sentence each: the case for leaving",
        "    now, and the case for one more turn. Then decide end_session yourself:",
        "    true means the message you are planning now is their LAST one: they write it",
        "    and leave. Its act is their Complete act, and next_step says how they close;",
        "    a close need not thank anyone (people often just stop with a last remark, a",
        "    short acknowledgement or a final ask). false means they write again after the",
        "    assistant replies. People do not apply fixed criteria; they stop on a feeling",
        "    of good enough. So judge THIS person, with THIS patience, at THIS point.",
    ])


CRITIQUE_FIELD = '"critique": "<one sentence on what your previous state got wrong, judged against what happened>",\n'
PROFILE_NOTE_FIELD = (' "profile_note": "<one or two sentences about HOW THIS PERSON WRITES that the message you last '
                      'produced for them got wrong (length, tone, casing, phrasing, how directly they ask); describe '
                      'their writing only, not what the assistant should do; empty on their first message>",\n')


def system_prompt_pend(implicit_profile=False):
    s = system_prompt_v3()
    s = _replace_once(s, rules_block_v3(), rules_block_pend())
    s = _replace_once(s, NEW_STOP_FIELDS, NEW_STOP_FIELDS_PEND)
    if implicit_profile:
        s = _replace_once(s, CRITIQUE_FIELD, CRITIQUE_FIELD + PROFILE_NOTE_FIELD)
        # the Speaker now also receives the notes and a few example messages: say so
        s = _replace_once(s, "and it sees only the state you write.",
                          "and it sees the state you write, your notes on how this person writes, and a few "
                          "real messages from other people with a similar style.")
    return s


def goal_lines(scenario):
    """D4: the whole goal the person came with (topic, context, what they already know)."""
    sc = scenario or {}
    goal = sc.get("goal") or {}
    out = []
    ctx = " ".join(str(goal.get("context", "")).split())
    if ctx:
        out.append("- context: " + ctx)
    known = (sc.get("persona_goal_interaction") or {}).get("known_datasets") or []
    if known:
        out.append("- they already know: " + ", ".join(str(k) for k in known))
    return "\n".join(out)


def user_prompt_pend(scenario, prev_block, hist_u, hist_a, turn, ledger, prev_ann=None, ip_sections=""):
    """user_prompt_v3 without the GOAL STATUS block, with the full goal under WHAT THEY CAME FOR, and
    (optionally) the Implicit Profile sections placed in the static prefix right before the turn line,
    so prompt fitting never drops them."""
    marker = "\n\nGOAL STATUS"
    up = user_prompt_v3(scenario, prev_block, hist_u, hist_a, turn, ledger,
                        {"status": "NOT ASSESSED", "unmet": []}, prev_ann=prev_ann)
    blk = goal_block({"status": "NOT ASSESSED", "unmet": []})
    up = _replace_once(up, blk, "")
    assert marker not in up
    topic = " ".join(str(((scenario or {}).get("goal") or {}).get("topic", "")).split())
    extra = goal_lines(scenario)
    if extra:
        old = "\n\nWHAT THEY CAME FOR\n" + topic + "\n\n"
        up = _replace_once(up, old, "\n\nWHAT THEY CAME FOR\n" + topic + "\n" + extra + "\n\n")
    if ip_sections:
        static = up.split("\n\nTHE CONVERSATION SO FAR\n", 1)[0]
        if static.count(ANCHOR) != 1:
            raise ValueError("turn anchor must occur exactly once in the static prefix")
        i = up.find(ANCHOR)
        up = up[:i] + ip_sections + up[i:]
    return up


def read_plan_pend(raw, turn, scenario, rng, ledger):
    """read_plan_v3 (override off, own length, turn-1 end ignored), plus goal_met / still_wanted.

    When end_session is true the message is the close: its act is the Planner's own Complete entry
    (ACT_FULL lists one entry per move, so it always exists), with that entry's length. This makes
    the act agree with the Planner's decision; it does not decide anything itself. Both the act that
    was drawn before and the replacement are kept in diag."""
    fields, diag, end = read_plan_v3(raw, turn, scenario, rng, ledger)
    if fields is None:
        return None, diag, False
    d = state.json_of(raw) or {}
    gm = str(d.get("goal_met", "") or "").strip().lower()
    fields["goal_met"] = gm if gm in GOAL_MET else None
    diag["goal_met_raw"] = d.get("goal_met")
    sw = " ".join(str(d.get("still_wanted", "") or "").split())
    fields["still_wanted"] = sw
    fields["profile_note"] = " ".join(str(d.get("profile_note", "") or "").split())      # never cut
    if end:
        comp = []
        for e in (d.get("act_distribution") or []):
            if not isinstance(e, dict):
                continue
            mv, ac = acts.normalise(str(e.get("move", "")), str(e.get("act", "")))
            if mv == "Complete":
                try:
                    p = float(e.get("p", 0) or 0)
                except (TypeError, ValueError):
                    p = 0.0
                comp.append((p, mv, ac, e.get("length_words")))
        diag["drawn_before_end"] = [fields.get("move"), fields.get("act")]
        if comp:
            comp.sort(key=lambda x: -x[0])          # stable: the first listed wins a tie
            _, mv, ac, lw = comp[0]
            if (mv, ac) != (fields.get("move"), fields.get("act")):
                fields["move"], fields["act"] = mv, ac
                b = acts.bench_of(ac, turn)
                fields["bench_act"] = ("%s / %s" % b) if b else "(unmapped)"
                try:
                    lwi = int(round(float(lw)))
                except (TypeError, ValueError):
                    lwi = None
                if lwi is not None and lwi > 0:
                    fields["length_words"] = lwi
                else:
                    fields.pop("length_words", None)
                diag["length_source"] = "complete_entry"
                diag["length_from_planner"] = lwi
            diag["end_act_from_complete_entry"] = True
        else:
            diag["end_without_complete_entry"] = True
    elif fields.get("move") == "Complete":
        # the reverse inconsistency: a Complete (closing) act drawn while the Planner decided to go on.
        # Redraw the act from the Planner's own NON-Complete entries (renormalised), so the message and
        # the decision agree (approved by the user 2026-09-25; recorded in diag).
        rest = []
        for e in (d.get("act_distribution") or []):
            if not isinstance(e, dict):
                continue
            mv, ac = acts.normalise(str(e.get("move", "")), str(e.get("act", "")))
            try:
                p = float(e.get("p", 0) or 0)
            except (TypeError, ValueError):
                p = 0.0
            if mv not in ("Complete", "Other") and p > 0:
                rest.append((p, mv, ac, e.get("length_words")))
        diag["complete_without_end"] = [fields.get("move"), fields.get("act")]
        if rest:
            x = (rng or __import__("random")).random() * sum(r[0] for r in rest)
            acc = 0.0
            pick = rest[-1]
            for r in rest:
                acc += r[0]
                if x <= acc:
                    pick = r
                    break
            _, mv, ac, lw = pick
            fields["move"], fields["act"] = mv, ac
            b = acts.bench_of(ac, turn)
            fields["bench_act"] = ("%s / %s" % b) if b else "(unmapped)"
            try:
                lwi = int(round(float(lw)))
            except (TypeError, ValueError):
                lwi = None
            if lwi is not None and lwi > 0:
                fields["length_words"] = lwi
            else:
                fields.pop("length_words", None)     # never keep the Complete entry's length
            diag["complete_redrawn_to"] = [mv, ac]
    fields["last_message"] = bool(end)
    return fields, diag, end


LAST_MESSAGE_LINE = "\n- this is their last message: they close the conversation with it and then leave"


def speaker_block_pend(fields):
    blk = PP.render_block(fields, (fields.get("still_wanted") or "(nothing named)", ""))
    if fields.get("last_message"):
        blk += LAST_MESSAGE_LINE            # the closing signal is explicit, not only implied by the act
    return blk
