"""Audit-driven tests for v16 (appended to test_v16.py) + the updated anneal/floor tamper test."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = '''def test_verify_catches_early_anneal_and_floor():
    sp, out = fresh("--stop-sup-floor", "0.3", "--ablation", "test-floor", updates=4)
    p = os.path.join(out, "updates.jsonl")
    rows = T.read_jsonl(p)
    rows[1]["train_aggregate"]["aux_weight"] = 0.2                            # below floor AND an anneal without trigger
    with open(p, "w") as f:
        f.write("".join(json.dumps(r) + "\\n" for r in rows))
    rep = v16_report(out)
    assert rep.checks["rl.aux_floor"]["fail"] == 1 and rep.checks["rl.d2_trigger"]["fail"] == 1'''
new = '''def _rewrite(path, rows):
    with open(path, "w") as f:
        f.write("".join(json.dumps(r) + "\\n" for r in rows))


def test_verify_catches_early_anneal_and_floor():
    sp, out = fresh("--stop-sup-floor", "0.3", "--ablation", "test-floor", updates=4)
    p = os.path.join(out, "updates.jsonl")
    rows = T.read_jsonl(p)
    rows[1]["train_aggregate"]["aux_annealed"] = 0.2                          # an anneal without a D2 trigger
    rows[1]["train_aggregate"]["aux_weight"] = 0.2                            # ... and below the floor
    _rewrite(p, rows)
    rep = v16_report(out)
    assert rep.checks["rl.aux_floor"]["fail"] >= 1 and rep.checks["rl.d2_trigger"]["fail"] == 1'''
assert s.count(old) == 1
s = s.replace(old, new)
s += '''

# ------------------------------------------------------------------ audit-driven tests
def make_big_splits(d, n_train=24, n_extra=6):
    ids = ["b%03d" % i for i in range(n_train + n_extra + 12)]
    tr, ex = ids[:n_train], ids[n_train:n_train + n_extra]
    val, te = ids[n_train + n_extra:n_train + n_extra + 2], ids[n_train + n_extra + 2:]
    sp = {"folds": [{"fold": 0, "train": tr, "validation": val, "test": te, "train_all": tr + ex,
                     "validation_all": val, "test_all": te, "forbidden_for_training": val + te}]}
    p = os.path.join(d, "big_splits.json")
    json.dump(sp, open(p, "w"))
    return p


def test_item1_refill_cap(monkeypatch):
    # every group degenerate (the fake policy never ends) -> refill until the cap (task1_convs extra conversations)
    orig = T.FakeEnv.task1_sample

    def never_end(self, *a, **k):
        out = orig(self, *a, **k)
        for x in out:
            x["ended_planner"] = False
            x["reward"] = float(not x["real_final"])
        return out
    monkeypatch.setattr(T.FakeEnv, "task1_sample", never_end)
    d = tempfile.mkdtemp()
    sp, out = make_big_splits(d), os.path.join(d, "run")
    run(sp, out, updates=2)
    for u in T.read_jsonl(os.path.join(out, "updates.jsonl")):
        th = u["train_aggregate"]["task1_train"]
        assert th["refill_stop"] == "cap" and th["n_refill_convs"] == 8 and th["n_informative_groups"] == 0
        assert len(th["refill_convs"]) == 8 and not set(th["refill_convs"]) & set(th["base_convs"])
    rep = v16_report(out)
    assert all(c["fail"] == 0 for c in rep.checks.values()), {k: c for k, c in rep.checks.items() if c["fail"]}


def test_item1_crash_mid_refill_then_resume(monkeypatch):
    d = tempfile.mkdtemp()
    sp = make_splits(d)
    a, b = os.path.join(d, "a"), os.path.join(d, "b")
    run(sp, a, updates=4)
    orig, state = T.append_jsonl, {"n": 0, "armed": True}

    def crashing(path, row):
        orig(path, row)
        if state["armed"] and path.endswith("rollouts_task1.jsonl") and row.get("refill") and row["update"] >= 2:
            state["n"] += 1
            if state["n"] == 2:
                state["armed"] = False
                raise SystemExit("simulated crash mid-refill")
    monkeypatch.setattr(T, "append_jsonl", crashing)
    with pytest.raises(SystemExit):
        run(sp, b, updates=4)
    assert not state["armed"], "the crash was never reached (no refill in the dry run?)"
    monkeypatch.setattr(T, "append_jsonl", orig)
    run(sp, b, "--resume", updates=4)
    load = lambda p: sorted((r["update"], r["conversation_id"], r["t"], r["refill"], tuple(x["reward"] for x in r["samples"]))
                            for r in T.read_jsonl(os.path.join(p, "rollouts_task1.jsonl")))
    assert load(a) == load(b)
    pa = [u["policy_sha_after"] for u in T.read_jsonl(os.path.join(a, "updates.jsonl"))]
    pb = [u["policy_sha_after"] for u in T.read_jsonl(os.path.join(b, "updates.jsonl"))]
    assert pa == pb
    rep = v16_report(b)
    assert all(c["fail"] == 0 for c in rep.checks.values())


def test_verify_task1_rows_of_an_aborted_attempt_are_ignored():
    sp, out = fresh(updates=3)
    p = os.path.join(out, "rollouts_task1.jsonl")
    rows = T.read_jsonl(p)
    extra = [dict(r, conversation_id="c13") for r in rows if r["update"] == 2][:2]   # stray rows of another attempt
    _rewrite(p, extra + rows + [r for r in rows if r["update"] == 3])                # and duplicated u3 rows
    rep = v16_report(out)
    assert all(c["fail"] == 0 for c in rep.checks.values()), {k: c for k, c in rep.checks.items() if c["fail"]}


def test_verify_d2_recomputed_and_aux_n_and_adv():
    sp, out = fresh("--t1-trigger-margin", "-1", "--ablation", "test-d2", updates=6)
    assert all(c["fail"] == 0 for c in v16_report(out).checks.values())
    pv = os.path.join(out, "validation.jsonl")
    val = T.read_jsonl(pv)
    for r in val:
        if r["kind"] == "summary" and r["update"] == 2:
            r["d2"]["streak"] = 2                                           # a logged streak that is not true
    _rewrite(pv, val)
    assert v16_report(out).checks["rl.d2_trigger"]["fail"] >= 1
    sp, out = fresh(updates=3)
    pu = os.path.join(out, "updates.jsonl")
    ups = T.read_jsonl(pu)
    ups[0]["learner_stats"]["aux_n"] = 0                                    # supervision silently empty
    del ups[1]["learner_stats"]["adv_abs_mean"]
    _rewrite(pu, ups)
    rep = v16_report(out)
    assert rep.checks["rl.aux_floor"]["fail"] == 1 and rep.checks["rl.adv_norm"]["fail"] == 1


def test_verify_d2_streak_resets():
    # met, not met, met: no trigger (recomputation from the summaries' bal_p)
    d = tempfile.mkdtemp()
    out = os.path.join(d, "run")
    os.makedirs(out)
    meta = {"kind": "start", "config": {"args": {"task1_G": 8, "task1_convs": 0, "t1_trigger_margin": 0.1, "splits": "x",
                                                 "fold": 0}, "algo_cfg": {"grpo_std_norm": False}}}
    _rewrite(os.path.join(out, "run_meta.jsonl"), [meta])
    summ = []
    for u, b, streak in ((0, 0.5, None), (5, 0.7, 1), (10, 0.55, 0), (15, 0.7, 1)):
        summ.append({"kind": "summary", "update": u, "policy_sha": "p%d" % u, "task1": {"bal_p": b},
                     "d2": None if streak is None else {"met": b >= 0.6, "streak": streak, "triggered_at": None}})
    _rewrite(os.path.join(out, "validation.jsonl"), summ)
    rep = V.Report()
    V.check_v16(out, rep)
    assert rep.checks["rl.d2_trigger"]["fail"] == 0
    summ[3]["d2"]["triggered_at"] = 15                                      # a trigger without two in a row
    _rewrite(os.path.join(out, "validation.jsonl"), summ)
    rep = V.Report()
    V.check_v16(out, rep)
    assert rep.checks["rl.d2_trigger"]["fail"] >= 1


def test_legacy_summaries_select_on_term_f1():
    # a pre-v16 validation summary (no selection_task1_metric) is recomputed with term_f1
    d = tempfile.mkdtemp()
    vp = os.path.join(d, "validation.jsonl")
    s_ = {"kind": "summary", "update": 5, "val_temperature": 0.7, "val_seeds": [0, 1], "selection_score": 0.906 - 0.625 + 0.25,
          "w_sel_cov": 1.0, "w_sel_w1": 1.0, "w_sel_task1": 1.0, "turn_stats": {"coverage_mean": 0.906, "turn_w1": 0.625},
          "task1": {"term_f1": 0.25}}
    _rewrite(vp, [s_])
    json.dump({"update": 5, "selection_score": s_["selection_score"]}, open(os.path.join(d, "best.json"), "w"))
    rep = V.Report()
    V.check_selection(vp, d, rep)
    assert rep.checks["rl.selection"]["fail"] == 0
    rep = V.Report()
    V.check_v16(d, rep)                                                     # no run_meta: nothing to check
    assert not any(c["fail"] for c in rep.checks.values())


def test_item8_pooled_equals_task1_stop():
    import random as _r
    rng = _r.Random(3)
    d = tempfile.mkdtemp()
    convs, rows_t1 = {}, []
    for i in range(12):
        n = rng.randint(1, 7)
        dec = [rng.random() < 0.3 for _ in range(n)]            # raw Planner end decisions
        blank = [rng.random() < 0.1 for _ in range(n)]
        k1b = rng.random() < 0.1
        turns = [{"t": t, "real_final": t == n, "ended_planner": dec[t - 1], "speaker_blank": blank[t - 1]} for t in range(1, n + 1)]
        rows_t1.append({"turns": turns, "k1_speaker_blank": k1b})
        flags, k1 = T1.bench_flags(turns, k1b)                  # the M2 flags task1_v4.py writes (greedy_ended / k1 ended)
        convs["c%02d" % i] = (flags, k1)
    f = _write_fold(d, "x.jsonl", convs)
    m = TP.metrics(TP.load_arm([f]))
    ref = T1.task1_stop_metrics(rows_t1)
    for k in ("term_f1", "premature_end_rate", "premature", "k1_end_rate", "n_conversations", "n_turns"):
        assert abs(m[k] - ref[k]) < 1e-12, k
    res = TP.main(["--arm", "a=%s" % f])
    assert set(res["files_sha256"]["a"]) == {f, f + ".k1.jsonl"}
'''
open(p, "w", encoding="utf-8").write(s)
print("patched")
