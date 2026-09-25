# -*- coding: utf-8 -*-
"""Implicit Profile + few-shot style examples (user design, 2026-09-25). Pure python.

Both tasks run from the first message on. After every message the Planner writes a `profile_note`:
how THIS person's writing differs from the message the simulator last produced for them and what to
do differently. The notes accumulate into the Implicit Profile, which goes back into the Planner's
input and into the Speaker block on every later turn.

  Task 1 (teacher-forced, gold available): the note compares the simulator's PREDICTION for message
          t-1 with the person's REAL message t-1 (which is part of the given history at turn t), plus a
          deterministic measurement of the difference. Nothing from message t or later is ever used.
  Task 2 (no gold): the note is a self-critique of the simulator's own last message, judged against
          who the person is, the notes so far, and how the assistant responded to it.

The Speaker additionally gets up to k short real messages written by OTHER people of the same
interaction style and English proficiency, from an allowed pool (a fold's train side; for a
whole-corpus run, leave-one-out) that excludes the current conversation and every conversation that
shares its goal or its persona. A candidate that copies an example (a run of >= COPY_NGRAM words) is
rejected like any other guard failure.
"""
from __future__ import annotations

import hashlib
import random
import re

MAX_NOTES = 6
MAX_NOTE_CHARS = 400
COPY_NGRAM = 8
GREET = re.compile(r"^\s*(hi|hello|hey|dear|good (morning|afternoon|evening)|thanks|thank you)\b", re.I)
IP_HEAD = "\n\nWHAT YOU HAVE LEARNED ABOUT HOW THIS PERSON WRITES (your own notes, oldest first)\n"
T1_HEAD = "\n\nYOUR PREDICTION FOR THEIR LAST MESSAGE, AND WHAT THEY REALLY WROTE\n"
T2_HEAD = "\n\nCHECK YOUR LAST MESSAGE\n"
SPK_NOTES = "- how this person writes (notes): "
SPK_EXAMPLES = "- how people like them write (other people; for style only, never copy): "


def words(text):
    return re.findall(r"\S+", text or "")


def style_stats(text):
    t = (text or "").strip()
    return {"words": len(words(t)), "questions": t.count("?"), "greeting": bool(GREET.match(t)),
            "lowercase_start": bool(t[:1]) and t[:1].islower(), "exclaims": t.count("!"),
            "lines": len([x for x in t.splitlines() if x.strip()])}


def measured_diff(pred, gold):
    """Deterministic difference between the simulator's prediction and the real message."""
    p, g = style_stats(pred), style_stats(gold)
    parts = ["they wrote %d words, you wrote %d" % (g["words"], p["words"])]
    if g["questions"] != p["questions"]:
        parts.append("they asked %d question(s), you asked %d" % (g["questions"], p["questions"]))
    if g["greeting"] != p["greeting"]:
        parts.append("they %s with a greeting or thanks, you %s" % ("opened" if g["greeting"] else "did not open",
                                                                  "did" if p["greeting"] else "did not"))
    if g["lowercase_start"] != p["lowercase_start"]:
        parts.append("they started %s, you %s" % ("in lower case" if g["lowercase_start"] else "capitalised",
                                                 "in lower case" if p["lowercase_start"] else "capitalised"))
    if g["lines"] != p["lines"]:
        parts.append("they used %d line(s), you %d" % (g["lines"], p["lines"]))
    return "; ".join(parts)


def task1_context(real_users, preds, t):
    """Context for turn t of a teacher-forced run: prediction and real text of message t-1 ONLY.
    real_users / preds are the texts of messages 1..; entries at index >= t-1 are never read."""
    if t < 2:
        return None
    if len(preds) < t - 1:
        raise ValueError("prediction for message %d missing" % (t - 1))
    return {"mode": "task1", "pred_prev": preds[t - 2], "gold_prev": real_users[t - 2]}


def render_planner_sections(notes, ctx):
    notes = [n for n in (notes or []) if n][-MAX_NOTES:]
    s = IP_HEAD + ("\n".join("- " + n for n in notes) if notes else "- (no notes yet)")
    if ctx and ctx.get("mode") == "task1":
        s += (T1_HEAD + "- you predicted: \"%s\"\n- they really wrote the last USER message below\n- measured: %s\n"
              "- in profile_note, say how their real message differs from yours and what to do differently"
              % (" ".join((ctx["pred_prev"] or "").split()), measured_diff(ctx["pred_prev"], ctx["gold_prev"])))
    elif ctx and ctx.get("mode") == "task2":
        s += (T2_HEAD + "- the last USER message below is the one you produced for them\n"
              "- in profile_note, judge it against who this person is, your notes, and how the assistant "
              "responded to it, and say what to do differently")
    return s


def clean_note(note):
    n = " ".join(str(note or "").split())
    return n[:MAX_NOTE_CHARS]


def speaker_lines(notes, examples):
    out = []
    notes = [n for n in (notes or []) if n][-MAX_NOTES:]
    if notes:
        out.append(SPK_NOTES + " | ".join(notes))
    if examples:
        out.append(SPK_EXAMPLES + " / ".join('"%s"' % " ".join(e["text"].split()) for e in examples))
    return ("\n" + "\n".join(out)) if out else ""


def norm_text(text):
    return " ".join((text or "").lower().split())


def duplicate_of(text, earlier):
    """True when the candidate's text is exactly one of the earlier candidates of the same turn
    (case/whitespace-insensitive). Exact identity only: no similarity threshold."""
    n = norm_text(text)
    return bool(n) and any(n == norm_text(e) for e in earlier)


def leaks_scaffold(text):
    """The Speaker wrote our block headers (the E1.6 template guard only knows the older field names)."""
    low = (text or "").lower()
    return any(h.strip().lstrip("- ").rstrip(": ").lower() in low for h in (SPK_NOTES, SPK_EXAMPLES)) \
        or "profile_note" in low


def copies_example(text, examples, n=COPY_NGRAM):
    w = [x.lower() for x in words(text)]
    if len(w) < n:
        return False
    grams = {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
    for e in examples or []:
        ew = [x.lower() for x in words(e["text"])]
        if any(tuple(ew[i:i + n]) in grams for i in range(len(ew) - n + 1)):
            return True
    return False


def style_key(persona):
    per = persona or {}
    return (str((per.get("individual_traits") or {}).get("interaction_style", "")).strip().lower(),
            str((per.get("general_info") or {}).get("proficiency_in_english", "")).strip().lower())


class FewShotPool:
    """Real user messages from an allowed set of conversations, keyed by style."""

    def __init__(self, recs_by_cid, allowed, goal_of, persona_of, split_messages, max_words=80):
        self.goal_of, self.persona_of = dict(goal_of), dict(persona_of)
        self.allowed = set(allowed)
        self.items = []
        for cid in sorted(self.allowed):
            rec = recs_by_cid[cid]
            users, _ = split_messages(rec)
            key = style_key((rec.get("scenario") or {}).get("persona"))
            for i, u in enumerate(users):
                txt = " ".join((u.get("text") or "").split())
                if txt and len(words(txt)) <= max_words:
                    self.items.append({"cid": cid, "t": i + 1, "text": txt, "key": key})

    def describe(self):
        return {"n_conversations": len(self.allowed), "n_messages": len(self.items)}

    def select(self, cid, persona, turn, k=3, variant=0):
        """k examples of the same style, never from this conversation or one sharing its goal/persona;
        the first-message examples are preferred for turn 1, later messages otherwise. Deterministic per
        (conversation, turn, variant); each candidate of a turn uses its own variant, so the Speaker
        prompts of the candidates differ."""
        key = style_key(persona)
        g, p = self.goal_of.get(cid), self.persona_of.get(cid)
        allowed = [x for x in self.items if x["cid"] != cid
                   and (g is None or self.goal_of.get(x["cid"]) != g)
                   and (p is None or self.persona_of.get(x["cid"]) != p)]
        ok = [x for x in allowed if x["key"] == key]
        if len(ok) < k:
            # too few people of the same style AND proficiency: back off to the same interaction style
            ok = [x for x in allowed if x["key"][0] == key[0]]
        first = [x for x in ok if (x["t"] == 1) == (turn == 1)]
        cands = first if len(first) >= k else ok
        rng = random.Random(int(hashlib.sha256(("fs|%s|%d|%d" % (cid, turn, variant)).encode()).hexdigest()[:8], 16))
        return rng.sample(cands, min(k, len(cands)))
