#!/usr/bin/env python3
"""SPEC v18 §1.1 (user 2026-10-01, Q1): our own act labeller for the real people's messages -- gpt-oss-120b (the local
vLLM server, :8029, reasoning effort low) with OUR L2 move taxonomy only (sepsim.acts: the six moves of the Planner
prompt + Other; the text of acts.coarse_block() / fine_block()). The benchmark's labels and instruments are never read;
the benchmark act codebook is opened ONLY to refuse a prompt that shares an 8-gram with it.

Per message t of every conversation of splits[fold][--split] (train_all for the reward, validation_all for §7 only):
  input   the dialogue up to the person's message t (people 1..t, assistant 1..t-1), texts only (no annotations, nothing
          after message t)
  output  {"move", "act"} normalised by acts.normalise; K = 3 votes (seeds 0, 1, 2; temperature 1.0; max 2048 completion
          tokens); an empty / unparseable answer is retried with a new seed (seed + 100 * retry) up to 3 times; a vote
          still missing is counted (n_votes_missing, n_retries); a message with < 2 valid votes gives no move information
          (n_valid_votes < 2 -> the reward treats r_act / r_len as not applicable)
  soft    q7(m) = the share of the valid votes on move m (7 moves incl. Complete, Other); the fine acts are recorded only
  stop    more than 5% of all votes missing -> exit 1 (nothing silently degraded)
Quality record (<out>.meta.json): Fleiss kappa (coarse, messages with 3 valid votes), complete agreement, move shares,
counts, prompt sha + text, model, effort, seeds, corpus / splits sha, the 8-gram check, every opened file (audit hook).
A 20-row human-readable sample (<out>.review.md) is written for the user; the training refuses the labels until the user
writes <out>.APPROVED containing the label file's sha256 (train_planner_rl --labels-approval; run_v18_fold.sh).
Cache: an existing <out> whose meta key (prompt, model, effort, seeds, corpus, splits, ids) matches is reused, never
overwritten by a different key.

  python label_acts.py --fold 2 --split train_all --out /tmp2/mzjiang_usersim/grpo_planner/labels_v18/act_labels_train_f2.jsonl
  python label_acts.py --fold 2 --split validation_all --out .../act_labels_val_f2.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rl_reward as RR  # noqa: E402

BENCH = "/tmp2/hchsu/trec2026-usersim-benchmark"
CODEBOOK = os.path.join(BENCH, "instruments/prompts/act_codebook_system.txt")   # read ONLY for the 8-gram refusal
CORPUS = "/home/mzjiang/v5-latency/data.jsonl"
SPLITS = "/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json"
LABELS_DIR = "/tmp2/mzjiang_usersim/grpo_planner/labels_v18"
K_VOTES = 3
SEEDS = (0, 1, 2)
MAX_TOKENS = 2048
MAX_RETRIES = 3
FAIL_RATE_STOP = 0.05
NGRAM = 8
REVIEW_N = 20
OPENED = []


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        OPENED.append(os.path.abspath(os.fsdecode(args[0])))


def sha_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ the prompt (our L2 taxonomy only)
def build_system_prompt(acts):
    """The labeller's instructions: the six moves and the fine acts exactly as the Planner prompt shows them
    (acts.coarse_block() / fine_block()), plus Other; JSON answer."""
    return ("You label the conversational move of ONE message written by a person who is searching for datasets with "
            "the help of an assistant.\n\n"
            "The moves (choose exactly one):\n" + acts.coarse_block() + "\n"
            "    %-10s none of the above\n\n" % "Other" +
            "The acts that belong to each move:\n" + acts.fine_block() + "\n\n"
            "Label only the message marked >>> (the person's latest message), using what was said before it as context. "
            "Answer with one JSON object and nothing else:\n"
            '{"move": "<one of the moves>", "act": "<one of that move\'s acts>"}\n'
            'Use the move Other (with the act other) only when none of the six moves fits.')


def build_user_prompt(users, agents, t):
    """People 1..t and the assistant 1..t-1 (texts only); the message to label is marked."""
    lines = ["THE CONVERSATION SO FAR"]
    for i in range(t):
        mark = ">>> " if i == t - 1 else ""
        lines.append("%sPERSON (message %d): %s" % (mark, i + 1, users[i]))
        if i < t - 1 and i < len(agents):
            lines.append("ASSISTANT: %s" % agents[i])
    return "\n".join(lines)


def words(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def ngrams(text, n=NGRAM):
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def ngram_overlap(a, b, n=NGRAM):
    """Number of shared word n-grams (lowercase alphanumeric tokens)."""
    return len(ngrams(a, n) & ngrams(b, n))


# ------------------------------------------------------------------ answers
def parse_answer(text, acts):
    """-> (move, act) or None for an empty / unparseable answer (no JSON object, or neither field). parse_vote with the
    model's move kept (audit A, SHOULD)."""
    got = parse_vote(text, acts)
    return None if got is None else (got["move"], got["act"])


def parse_vote(text, acts):
    """Audit A (SHOULD): the model's own MOVE is the vote whenever it names one of the six moves -- acts.normalise would
    replace it by "Other" for a missing / misspelled act, or by the act's move when they disagree. Then the act is kept
    if it belongs to that move, else recorded as "unknown" (the fine act only rides along). A move that is not one of the
    six (or Other) falls back to acts.normalise of (move, act). -> {"move", "act", "raw_move", "raw_act", "normalised",
    "move_overridden_by_normalise"} or None for an empty / unparseable answer."""
    t = (text or "").strip()
    if not t:
        return None
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(d, dict) or (not str(d.get("move", "") or "").strip() and not str(d.get("act", "") or "").strip()):
        return None
    rm, ra = str(d.get("move", "") or ""), str(d.get("act", "") or "")
    norm = acts.normalise(rm, ra)
    m = rm.strip().title()
    a = ra.strip().lower().replace(" ", "_").replace("-", "_")
    if m in acts.COARSE_ORDER and m != "Other":
        act = a if acts.COARSE_OF.get(a) == m else "unknown"
        move = m
    elif m == "Other":
        move, act = "Other", "other"
    else:
        move, act = norm
    return {"move": move, "act": act, "raw_move": rm, "raw_act": ra, "normalised": list(norm),
            "move_overridden_by_normalise": norm[0] != move}


def http_chat(base_url, model, system, user, seed, temperature=1.0, max_tokens=MAX_TOKENS, effort="low", timeout=600):
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature, "max_tokens": max_tokens, "seed": int(seed), "reasoning_effort": effort}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer %s" % os.environ.get("OPENAI_API_KEY", "unused")})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    ch = d["choices"][0]
    return (ch.get("message") or {}).get("content") or "", ch.get("finish_reason")


MAX_CONSECUTIVE_TRANSPORT_ERRORS = 5
_TRANSPORT = {"consecutive": 0}


def vote(chat, system, user, seed, acts):
    """One vote with up to MAX_RETRIES re-draws (new seeds) of an empty / unparseable answer.
    -> {"seed", "move", "act", "retries", "raw_move", ...} (move None = missing). MAX_CONSECUTIVE_TRANSPORT_ERRORS
    transport errors in a row (the server is down) stop the run instead of burning every vote (audit A)."""
    tries = []
    for k in range(MAX_RETRIES + 1):
        s = seed + 100 * k
        try:
            raw, fin = chat(system, user, s)
            _TRANSPORT["consecutive"] = 0
        except Exception as e:                         # a transport error counts as a failed try (recorded)
            raw, fin = "", "error: %r" % (e,)
            _TRANSPORT["consecutive"] += 1
            if _TRANSPORT["consecutive"] >= MAX_CONSECUTIVE_TRANSPORT_ERRORS:
                raise SystemExit("%d transport errors in a row (last: %r): the labeller server is not answering -- "
                                 "stopped, nothing written" % (_TRANSPORT["consecutive"], e))
        got = parse_vote(raw, acts)
        tries.append({"seed": s, "finish_reason": fin, "raw": (raw or "")[:400]})
        if got is not None:
            return dict(got, seed=s, retries=k, tries=tries)
    return {"seed": seed, "move": None, "act": None, "retries": MAX_RETRIES, "tries": tries}


def soft_label(votes):
    valid = [v for v in votes if v.get("move")]
    n = len(valid)
    q7 = {}
    for v in valid:
        q7[v["move"]] = q7.get(v["move"], 0.0) + 1.0 / n
    return q7, n


def fleiss_kappa(table, cats):
    """table: per item the counts over cats (same number of raters per item). -> kappa or None."""
    items = [r for r in table if sum(r) > 1]
    if not items:
        return None
    n = sum(items[0])
    if any(sum(r) != n for r in items):
        raise ValueError("Fleiss kappa needs the same number of ratings per item")
    N = len(items)
    p = [sum(r[j] for r in items) / (N * n) for j in range(len(cats))]
    P = [(sum(x * x for x in r) - n) / (n * (n - 1)) for r in items]
    Pbar, Pe = sum(P) / N, sum(x * x for x in p)
    return None if Pe >= 1.0 else (Pbar - Pe) / (1.0 - Pe)


# ------------------------------------------------------------------ main
def split_messages(rec):
    """sepsim pipeline.split_messages, texts only (participant "user" / "agent"; annotations never read)."""
    msgs = rec.get("chat_messages") or []
    users = [m["text"] for m in msgs if str(m.get("participant_name", "")).lower() == "user"]
    agents = [m["text"] for m in msgs if str(m.get("participant_name", "")).lower() == "agent"]
    return users, agents


def label_split(a, acts, chat):
    sp = json.load(open(a.splits, encoding="utf-8"))
    f = {int(x["fold"]): x for x in sp["folds"]}[int(a.fold)]
    ids = sorted(f[a.split])
    forb = set(f["forbidden_for_training"])
    assert not set(ids) & set(f.get("test_all") or []) and not set(ids) & set(f.get("test") or []), \
        "the test side is never labelled (§1.1)"
    if a.split == "train_all":
        assert not set(ids) & forb, "train_all intersects forbidden_for_training"
    elif a.split == "validation_all":
        assert set(ids) <= forb and not set(ids) & set(f["train_all"]), "validation_all must be forbidden / not train"
    else:
        raise SystemExit("--split must be train_all or validation_all (the test side is never labelled, §1.1)")
    recs = {}
    for l in open(a.corpus, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r["conversation_id"] in ids:
                recs[r["conversation_id"]] = r
    missing = sorted(set(ids) - set(recs))
    if missing:
        raise SystemExit("conversations missing from the corpus: %s" % missing[:3])
    system = build_system_prompt(acts)
    cb = a.codebook
    if os.path.exists(cb):
        ov = ngram_overlap(system, open(cb, encoding="utf-8").read())
        checked = True
    elif a.dry_run:
        ov, checked = 0, False
    else:
        raise SystemExit("the benchmark act codebook %s is not readable: the 8-gram check cannot run" % cb)
    if ov:
        raise SystemExit("the label prompt shares %d %d-grams with the benchmark act codebook: refused (§1.1)" % (ov, NGRAM))
    key = {"prompt_sha256": sha_text(system), "model": a.model, "reasoning_effort": a.effort, "seeds": list(SEEDS),
           "temperature": 1.0, "max_tokens": MAX_TOKENS, "corpus_sha256": sha_file(a.corpus),
           "splits_sha256": sha_file(a.splits), "fold": a.fold, "split": a.split, "ids": ids,
           "acts_sha256": RR.acts_sha()}
    mp = a.out + ".meta.json"
    if os.path.exists(a.out) and not os.path.exists(mp):
        raise SystemExit("%s exists without its %s: not reused, not overwritten (use a new --out)" % (a.out, mp))
    if os.path.exists(a.out) and os.path.exists(mp):
        m = json.load(open(mp, encoding="utf-8"))
        if m.get("key") == key and m.get("labels_sha256") == sha_file(a.out):
            print("cache reused: %s (%s)" % (a.out, m["labels_sha256"][:12]), flush=True)
            return m
        raise SystemExit("%s exists with another key / content: not reused, not overwritten (use a new --out)" % a.out)
    rows, n_missing, n_retries, n_ins = [], 0, 0, 0
    _TRANSPORT["consecutive"] = 0
    for cid in ids:
        users, agents = split_messages(recs[cid])
        for t in range(1, len(users) + 1):
            user = build_user_prompt(users, agents, t)
            vs = [vote(chat, system, user, s, acts) for s in SEEDS]
            q7, nv = soft_label(vs)
            n_missing += sum(1 for v in vs if not v["move"])
            n_retries += sum(v["retries"] for v in vs if v["move"]) + sum(MAX_RETRIES for v in vs if not v["move"])
            n_ins += int(nv < 2)
            rows.append({"conversation_id": cid, "t": t, "n_turns": len(users), "n_words": len(users[t - 1].split()),
                         "text_sha256": sha_text(users[t - 1]), "votes": vs, "q7": q7, "n_valid_votes": nv,
                         "majority": RR.label_majority(q7) if nv else None,
                         "fine_votes": [v["act"] for v in vs if v["move"]]})
            print("  %s t%d %s (%d valid)" % (cid[:10], t, rows[-1]["majority"], nv), flush=True)
    n_votes = K_VOTES * len(rows)
    rate = n_missing / n_votes if n_votes else 0.0
    cats = list(RR.V18_Q7)
    table = [[sum(1 for v in r["votes"] if v["move"] == c) for c in cats] for r in rows if r["n_valid_votes"] == K_VOTES]
    kappa = fleiss_kappa(table, cats) if table else None
    agree = (sum(1 for r in table if max(r) == K_VOTES) / len(table)) if table else None
    shares = {c: sum(r["q7"].get(c, 0.0) for r in rows) / len(rows) for c in cats} if rows else {}
    meta = {"key": key, "n_messages": len(rows), "n_votes": n_votes, "n_votes_missing": n_missing, "n_retries": n_retries,
            # audit A: valid votes whose move acts.normalise would have changed (we keep the model's move) / whose fine
            # act did not belong to the move (recorded "unknown")
            "n_move_overridden": sum(1 for r in rows for v in r["votes"] if v.get("move_overridden_by_normalise")),
            "n_act_unknown": sum(1 for r in rows for v in r["votes"] if v.get("act") == "unknown"),
            "n_insufficient": n_ins, "missing_rate": rate, "fleiss_kappa_coarse": kappa, "complete_agreement": agree,
            "move_shares": shares, "kappa_note": "self-consistency of the labeller, not correctness (§1.1)",
            "prompt_sha256": key["prompt_sha256"], "prompt_text": system, "model": a.model, "reasoning_effort": a.effort,
            "seeds": list(SEEDS), "ngram8_overlap": ov, "ngram8_checked": checked, "codebook_path": cb,
            "codebook_sha256": sha_file(cb) if os.path.exists(cb) else None, "split": a.split, "fold": a.fold,
            "time": time.time()}
    if rate > FAIL_RATE_STOP:
        json.dump(dict(meta, stopped=True), open(a.out + ".failed.json", "w", encoding="utf-8"), indent=1)
        raise SystemExit("labelling failure rate %.3f > %.2f (%d of %d votes missing): stopped, nothing written as labels "
                         "(see %s.failed.json)" % (rate, FAIL_RATE_STOP, n_missing, n_votes, a.out))
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))
    os.replace(tmp, a.out)
    meta["labels_sha256"] = sha_file(a.out)
    meta["opened_files"] = sorted(set(OPENED))
    write_review(a.out + ".review.md", rows, recs, meta)
    meta["review_sample"] = a.out + ".review.md"
    json.dump(meta, open(mp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: meta[k] for k in ("n_messages", "n_votes_missing", "n_retries", "n_insufficient", "missing_rate",
                                           "fleiss_kappa_coarse", "complete_agreement", "labels_sha256")}), flush=True)
    print("REVIEW the sample %s; to approve, write the label sha into %s.APPROVED, e.g.\n  echo %s > %s.APPROVED"
          % (meta["review_sample"], a.out, meta["labels_sha256"], a.out), flush=True)
    return meta


def write_review(path, rows, recs, meta, n=REVIEW_N):
    """A 20-row human-readable sample (seeded): the message, the assistant's previous reply, the three votes."""
    rng = random.Random(0)
    pick = sorted(rng.sample(range(len(rows)), min(n, len(rows))))
    out = ["# Act labels: review sample (%d of %d messages)" % (len(pick), len(rows)), "",
           "Labeller: %s, effort %s, seeds %s; Fleiss kappa %s (self-consistency, not correctness); missing votes %d."
           % (meta["model"], meta["reasoning_effort"], meta["seeds"], meta["fleiss_kappa_coarse"], meta["n_votes_missing"]),
           "Moves: Disclose / Reveal / Inquire / Navigate / Note / Complete / Other (our L2 taxonomy).", ""]
    for i in pick:
        r = rows[i]
        users, agents = split_messages(recs[r["conversation_id"]])
        prev = agents[r["t"] - 2] if r["t"] >= 2 and r["t"] - 2 < len(agents) else ""
        out += ["## %s, message %d of %d" % (r["conversation_id"], r["t"], r["n_turns"]), "",
                "- assistant before: %s" % (prev[:300].replace("\n", " ") + ("..." if len(prev) > 300 else "") if prev else "(none)"),
                "- **person**: %s" % users[r["t"] - 1].replace("\n", " "),
                "- votes: %s" % ", ".join("%s/%s" % (v["move"], v["act"]) if v["move"] else "(missing)" for v in r["votes"]),
                "- soft label: %s (majority %s)" % (json.dumps(r["q7"], sort_keys=True), r["majority"]), ""]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out))


def load_acts_module():
    os.environ.setdefault("SEPSIM_ACT_PRIOR", "nostopclobber")    # the pend Planner's prior mode (no dispute block)
    acts = RR.load_acts()
    return acts


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--split", choices=("train_all", "validation_all"), required=True)
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--codebook", default=CODEBOOK, help=argparse.SUPPRESS)
    ap.add_argument("--base-url", default="http://127.0.0.1:8029/v1")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="a fake chat function (tests)")
    a = ap.parse_args(argv)
    if os.path.abspath(a.out).startswith(BENCH):
        ap.error("never write into the benchmark tree (CLAUDE.md §0)")
    return a


def main(argv=None, chat=None):
    a = parse(argv)
    sys.addaudithook(_audit)
    acts = load_acts_module()
    if chat is None:
        if a.dry_run:
            raise SystemExit("--dry-run needs a chat function (tests)")
        chat = lambda system, user, seed: http_chat(a.base_url, a.model, system, user, seed, effort=a.effort)
    return label_split(a, acts, chat)


if __name__ == "__main__":
    main()
