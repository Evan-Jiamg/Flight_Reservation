"""Audit Q fixes in step0_coverage.py: reply pairing along chat order, per-conversation judge incident counters
(unclean conversations out of the AUC), one failure does not sink the batch, rows written per conversation."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''    def one(cid):
        users, agents = pipeline.split_messages(recs[cid])
        n = len(users)
        led = Ledger(reqs[cid]["req"], judge=judge)
        rows, prev = [], 0.0
        for t in range(1, n + 1):
            reply = agents[t - 1]["text"] if t - 1 < len(agents) else ""
            led.update(t, users[t - 1]["text"], reply)
            c = float(led.coverage())
            rows.append({"conversation_id": cid, "t": t, "n": n, "real_final": t == n, "has_reply": t - 1 < len(agents),
                         "n_req": len(reqs[cid]["req"]), "cov": c, "gain": c - prev, "complete": bool(led.complete())})
            prev = c
        return rows

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        allrows = [r for rows in ex.map(one, ids) for r in rows]
    with open(a.out, "w", encoding="utf-8") as fo:
        for r in allrows:
            fo.write(json.dumps(r) + "\\n")''', '''    import threading
    io_lock = threading.Lock()
    open(a.out, "w").close()
    incidents = {}

    def pairs(rec):
        """(user text, the agent text that answers it) along the chat order: every agent message after user message t
        and before user message t + 1 is the reply to t (the users/agents lists of split_messages are NOT paired)."""
        users, _ = pipeline.split_messages(rec)          # the Task 1 definition of the n user messages
        out, cur, alternating = [], None, True
        for m in rec.get("chat_messages") or []:
            role = m["participant_name"].lower()
            if role == "user":
                if cur is not None and not cur[1]:
                    alternating = False                  # two user messages in a row
                cur = [m["text"], []]
                out.append(cur)
            elif role == "agent":
                if cur is None:
                    alternating = False                  # the agent speaks first: not a reply to any user message
                    continue
                if cur[1]:
                    alternating = False
                cur[1].append(m["text"])
        assert len(out) == len(users), "pairing disagrees with split_messages"
        return [(u, "\\n\\n".join(r)) for u, r in out], alternating

    def one(cid):
        TE.episode_begin()                                # this thread's judge incident counters (as in Task 2)
        try:
            prs, alternating = pairs(recs[cid])
            n = len(prs)
            led = Ledger(reqs[cid]["req"], judge=judge)
            rows, prev = [], 0.0
            for t in range(1, n + 1):
                u, reply = prs[t - 1]
                led.update(t, u, reply)
                c = float(led.coverage())
                rows.append({"conversation_id": cid, "t": t, "n": n, "real_final": t == n, "has_reply": bool(reply),
                             "n_req": len(reqs[cid]["req"]), "cov": c, "gain": c - prev, "complete": bool(led.complete()),
                             "alternating": alternating})
                prev = c
            err = None
        except Exception as e:                            # recorded; the other conversations go on
            rows, err = [], repr(e)
        cnt = TE.episode_end()
        clean = err is None and not any(cnt.get(k) for k in ("judge_empty", "judge_error", "judge_unparseable",
                                                              "judge_retry_failed"))
        for r in rows:
            r["clean"] = clean
        with io_lock:
            incidents[cid] = {"error": err, "counters": {k: v for k, v in cnt.items() if v}, "clean": clean}
            with open(a.out, "a", encoding="utf-8") as fo:
                for r in rows:
                    fo.write(json.dumps(r) + "\\n")
            print("  %s n=%d clean=%s %s" % (cid[:10], len(rows), clean, err or ""), flush=True)
        return rows

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        allrows = [r for rows in ex.map(one, ids) for r in rows]''')
rep('''    by = {}
    for r in allrows:
        by.setdefault(r["conversation_id"], {})[r["t"]] = r''', '''    by = {}
    for r in allrows:
        if r["clean"]:                                    # a lost / failed judge verdict would read as coverage 0
            by.setdefault(r["conversation_id"], {})[r["t"]] = r''')
rep('''    res = {"fold": a.fold, "n_conversations": len(by), "n_decision_points": len(pts),''',
    '''    res = {"fold": a.fold, "n_train_conversations": len(ids), "n_conversations": len(by),
           "n_unclean_or_failed": sum(1 for v in incidents.values() if not v["clean"]),
           "incidents": {k[:10]: v for k, v in incidents.items() if not v["clean"] or v["counters"]},
           "n_non_alternating": sum(1 for rs in by.values() if not rs[1]["alternating"]), "n_decision_points": len(pts),''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
