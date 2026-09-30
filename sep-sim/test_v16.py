"""SPEC v16 (ops/SPEC_v16_grpo_opt.md, user 2026-09-28), kept for ARCHIVED runs (SPEC v17 B8/B9, user 2026-09-30).

The v17 trainer no longer implements v16, so the v16 branch of verify_pipeline is tested on run directories the v16
trainer wrote (fixtures_v16/, made once by fixtures_v16/make_fixtures_v16.py at commit 9afebdd): every check must pass on
the untouched run and catch each tampering. The pure v16 pieces that remain in the code (Dr. GRPO advantages, the w_dist
controller bound, the continuous Task 1 metric, the pooled Task 1 tool) keep their own tests. No GPU."""
import json
import math
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_algos as RA  # noqa: E402
import rl_controllers as RC  # noqa: E402
import task1_pooled as TP  # noqa: E402
import task1_stop as T1  # noqa: E402
import train_planner_rl as T  # noqa: E402
import verify_pipeline as V  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures_v16")


def fixture(name):
    """A private copy of an archived v16 run: (splits path, run dir)."""
    d = tempfile.mkdtemp()
    shutil.copytree(os.path.join(FIX, name), os.path.join(d, name))
    return os.path.join(d, name, "splits.json"), os.path.join(d, name, "run")


def v16_report(out, sp):
    rep = V.Report()
    V.check_v16(out, rep, splits_path=sp)
    return rep


def no_fail(rep):
    return all(c["fail"] == 0 for c in rep.checks.values())


def _rewrite(path, rows):
    with open(path, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def test_fixtures_are_v16_runs():
    man = json.load(open(os.path.join(FIX, "MANIFEST.json")))
    assert set(man["runs"]) == {"main", "d2", "cap", "iv"}
    for name in man["runs"]:
        sp, out = fixture(name)
        assert V.spec_version(out) == "v16"                      # B8: no spec_version, task1_G -> the v16 branch
        rep = v16_report(out, sp)
        assert no_fail(rep), (name, {k: c for k, c in rep.checks.items() if c["fail"]})
        assert rep.checks["rl.task1_refill"]["n"] > 0 and rep.checks["rl.task1_prob"]["n"] > 0


# ------------------------------------------------------------------ item 3: Dr. GRPO advantages (still in rl_algos)
def test_item3_no_std_norm():
    assert RA.ALGO_DEFAULTS["grpo_std_norm"] is False
    assert RA.group_advantages([0, 1, 1, 0]) == [-0.5, 0.5, 0.5, -0.5]
    assert RA.group_advantages([2.0, 2.0, 2.0]) is None                     # zero spread still skipped
    seq, stop = RA.split_group_advantages([1.0, 3.0], [0.5, 2.0])
    assert [a + b for a, b in zip(seq, stop)] == [-1.0, 1.0]                 # = R - mean R
    assert stop == [-0.75, 0.75]
    seq, stop = RA.split_group_advantages([1.0, 3.0], [0.5, 2.0], eps=0.0, std_norm=True)
    assert [round(a + b, 12) for a, b in zip(seq, stop)] == [-1.0, 1.0]      # std 1: unchanged by construction
    a, sk = RA.advantages_for_groups([[0, 1], [1, 1]], "grpo", RA.algo_cfg())
    assert a[0] == [-0.5, 0.5] and a[1] is None and sk == 1
    with pytest.raises(ValueError):
        RA.algo_cfg(grpo_std_norm="no")


# ------------------------------------------------------------------ item 6: w_dist floor (controller bound)
def test_item6_w_dist_floor():
    c = RC.LLMFactorController(RC.initial_cfg(version="v4", lambda_unparsed=1.0, lambda_hit_max_new=1.0),
                               transport=lambda req: {"choices": [{"message": {"content": json.dumps(
                                   {"factors": {"w_dist": 0.5}, "rationale": "r"})}}]})
    assert c.opt["bounds"]["w_dist"] == [1.0, 5.0]
    hist = [{"split": "train", "update": u, "reward_mean": 0.0, "shadow_reward_mean": 0.0, "components_mean": {}}
            for u in range(1, 6)]
    assert c.propose(hist)["w_dist"] == 1.0


# ------------------------------------------------------------------ items 1 + 2 + 4 on the archived run
def test_items12_groups_refill_and_aux_verified():
    sp, out = fixture("main")
    rows = T.read_jsonl(os.path.join(out, "rollouts_task1.jsonl"))
    assert rows and all(len(r["samples"]) == 8 for r in rows)
    assert any(r["refill"] for r in rows), "the archived run exercised the refill"
    rep = v16_report(out, sp)
    assert no_fail(rep)
    # tampering: a stop reason that does not fit, a floor that is not the run's, silent supervision, no adv log
    p = os.path.join(out, "updates.jsonl")
    ups = T.read_jsonl(p)
    for u in ups:
        th = u["train_aggregate"]["task1_train"]
        th["refill_stop"] = "cap" if th["refill_stop"] != "cap" else "filled"
    ups[0]["train_aggregate"]["aux_floor"] = 0.3
    ups[1]["learner_stats"]["aux_n"] = 0
    del ups[2]["learner_stats"]["adv_abs_mean"]
    _rewrite(p, ups)
    rep = v16_report(out, sp)
    assert rep.checks["rl.task1_refill"]["fail"] >= 1 and rep.checks["rl.aux_floor"]["fail"] >= 2
    assert rep.checks["rl.adv_norm"]["fail"] == 1


def test_item1_refill_cap_verified():
    sp, out = fixture("cap")
    for u in T.read_jsonl(os.path.join(out, "updates.jsonl")):
        th = u["train_aggregate"]["task1_train"]
        assert th["refill_stop"] == "cap" and th["n_refill_convs"] == 8 and th["n_informative_groups"] == 0
    assert no_fail(v16_report(out, sp))


def test_r1_refill_conversation_without_rows_is_not_a_false_failure():
    sp, out = fixture("cap")
    pu, pt = os.path.join(out, "updates.jsonl"), os.path.join(out, "rollouts_task1.jsonl")
    ups, t1 = T.read_jsonl(pu), T.read_jsonl(pt)
    th = ups[0]["train_aggregate"]["task1_train"]
    gone = th["refill_convs"][0]                                             # pretend it produced no row
    th["refill_convs"] = th["refill_convs"][1:]
    th["groups"] = [g for g in th["groups"] if g[0] != gone]
    th["n_refill_groups"] = sum(1 for g in th["groups"] if g[2])
    _rewrite(pu, ups)
    _rewrite(pt, [r for r in t1 if not (r["conversation_id"] == gone and r["update"] == 1)])
    rep = v16_report(out, sp)
    assert rep.checks["rl.task1_refill"]["fail"] == 0, rep.checks["rl.task1_refill"]


def test_verify_task1_rows_of_an_aborted_attempt_are_ignored():
    sp, out = fixture("main")
    p = os.path.join(out, "rollouts_task1.jsonl")
    rows = T.read_jsonl(p)
    extra = [dict(r, conversation_id="c13") for r in rows if r["update"] == 2][:2]   # stray rows of another attempt
    _rewrite(p, extra + rows + [r for r in rows if r["update"] == 3])                # and duplicated u3 rows
    assert no_fail(v16_report(out, sp))


def test_n1_splits_path_from_cli_and_sha(tmp_path):
    sp, out = fixture("main")
    mp = os.path.join(out, "run_meta.jsonl")
    meta = T.read_jsonl(mp)
    for m in meta:
        m["config"]["args"]["splits"] = str(tmp_path / "gone.json")          # the run's own path does not exist here
    _rewrite(mp, meta)
    rep = V.Report()
    V.check_v16(out, rep)
    assert rep.checks["rl.task1_refill"]["fail"] >= 1                        # never skipped silently
    assert no_fail(v16_report(out, sp))                                      # the verifier's --splits
    other = str(tmp_path / "other.json")
    s_ = json.load(open(sp))
    s_["folds"][0]["test"] = s_["folds"][0]["test"][:-1]
    json.dump(s_, open(other, "w"))
    assert v16_report(out, other).checks["rl.task1_refill"]["fail"] >= 1     # another split file: sha differs


# ------------------------------------------------------------------ item 7: continuous Task 1 metric
def test_item7_prob_metrics_values():
    pts = [{"real_final": True, "p_end": 0.8}, {"real_final": True, "p_end": 0.4},
           {"real_final": False, "p_end": 0.2}, {"real_final": False, "p_end": 0.4}, {"real_final": False, "p_end": 0.0, "valid": False}]
    m = T1.task1_prob_metrics(pts)
    assert abs(m["bal_p"] - (0.5 * 0.6 + 0.5 * (0.8 + 0.6 + 1.0) / 3)) < 1e-12
    # pairs (final, earlier): 0.8>0.2,0.8>0.4,0.8>0 ; 0.4>0.2, 0.4=0.4 (1/2), 0.4>0  -> 5.5 / 6
    assert abs(m["auc"] - 5.5 / 6) < 1e-12
    # v17 B5: nll over the VALID points only (the invalid one is counted, not scored); logloss is its alias
    ll = [-math.log(0.8), -math.log(0.4), -math.log(0.8), -math.log(0.6)]
    assert abs(m["nll"] - sum(ll) / 4) < 1e-12 and m["logloss"] == m["nll"]
    assert m["n_points"] == 5 and m["n_final"] == 2 and m["n_invalid"] == 1 and m["n_valid"] == 4
    assert T1.task1_prob_metrics([{"real_final": True, "p_end": 0.3}])["bal_p"] == 0.3
    assert T1.task1_prob_metrics([{"real_final": True, "p_end": 0.0, "valid": False}])["nll"] is None
    with pytest.raises(ValueError):
        T1.task1_prob_metrics([{"real_final": False, "p_end": 0.3}])
    with pytest.raises(ValueError):
        T1.task1_prob_metrics([{"real_final": True, "p_end": 1.3}])


def test_item7_validation_recomputed_and_tampering_caught():
    sp, out = fixture("main")
    val = T.read_jsonl(os.path.join(out, "validation.jsonl"))
    for v in [v for v in val if v["kind"] == "summary"]:
        rows = [r for r in val if r["kind"] == "task1" and r["update"] == v["update"]]
        pts = [p for r in rows for p in r["end_probs"]]
        assert len(pts) == sum(len(r["task1"]["turns"]) - 1 for r in rows)
        assert abs(T1.task1_prob_metrics(pts)["bal_p"] - v["task1"]["bal_p"]) < 1e-12
    p = os.path.join(out, "validation.jsonl")
    for r in val:
        if r["kind"] == "summary" and r["update"] == 2:
            r["task1"]["bal_p"] += 0.1
    _rewrite(p, val)
    assert v16_report(out, sp).checks["rl.task1_prob"]["fail"] == 1


def test_h_unscored_point_rule_and_n4_check():
    tr = T.Trainer.__new__(T.Trainer)

    class Env:
        def run_task1(self, cid, keep_prompts=False):
            return {"conversation_id": cid, "n_real": 3, "turns": [
                {"t": t, "real_final": t == 3, "ended_planner": False, "user_prompt": "p"} for t in (1, 2, 3)]}

        def task1_end_probe(self, cid, t, prompt, real_final):
            if t == 2:
                return {"valid": False, "decision_valid": False, "greedy_end": True}     # unparsed plan -> 0
            return {"valid": False, "decision_valid": True, "greedy_end": True}          # no value located -> greedy
    tr.env, tr.learner = Env(), None
    row = tr.task1_eval_row(5, "sha", "c1")
    ps = {p["t"]: p for p in row["end_probs"]}
    assert ps[2]["p_end"] == 0.0 and ps[3]["p_end"] == 1.0 and ps[3]["decision_valid"] is True
    assert all("user_prompt" not in x for x in row["task1"]["turns"])
    # N4: the verifier checks the rule on the archived run's rows
    sp, out = fixture("main")
    pv = os.path.join(out, "validation.jsonl")
    val = T.read_jsonl(pv)
    for r in val:
        if r["kind"] == "task1" and r["update"] == 2:
            r["end_probs"][0].update(valid=False, decision_valid=False, greedy_end=True, p_end=0.5)
            break
    _rewrite(pv, val)
    assert v16_report(out, sp).checks["rl.task1_prob"]["fail"] >= 1


# ------------------------------------------------------------------ D2 trigger (recomputed, never trusted)
def test_verify_d2_recomputed():
    sp, out = fixture("d2")
    summ = [v for v in T.read_jsonl(os.path.join(out, "validation.jsonl")) if v["kind"] == "summary"]
    d2 = {v["update"]: v["d2"] for v in summ}
    assert d2[0] is None and d2[2]["streak"] == 1 and d2[4]["triggered_at"] == 4
    assert no_fail(v16_report(out, sp))
    pv = os.path.join(out, "validation.jsonl")
    val = T.read_jsonl(pv)
    for r in val:
        if r["kind"] == "summary" and r["update"] == 2:
            r["d2"]["streak"] = 2                                           # a logged streak that is not true
    _rewrite(pv, val)
    assert v16_report(out, sp).checks["rl.d2_trigger"]["fail"] >= 1


def test_verify_catches_early_anneal_and_floor():
    sp, out = fixture("main")
    p = os.path.join(out, "updates.jsonl")
    rows = T.read_jsonl(p)
    rows[1]["train_aggregate"]["aux_annealed"] = 0.2                          # an anneal without a D2 trigger
    rows[1]["train_aggregate"]["aux_weight"] = 0.2                            # ... and below the floor
    _rewrite(p, rows)
    rep = v16_report(out, sp)
    assert rep.checks["rl.aux_floor"]["fail"] >= 1 and rep.checks["rl.d2_trigger"]["fail"] == 1


def _meta_run(summ):
    d = tempfile.mkdtemp()
    out = os.path.join(d, "run")
    os.makedirs(out)
    meta = {"kind": "start", "config": {"args": {"task1_G": 8, "task1_convs": 0, "t1_trigger_margin": 0.1, "splits": "x",
                                                 "fold": 0}, "algo_cfg": {"grpo_std_norm": False}}}
    _rewrite(os.path.join(out, "run_meta.jsonl"), [meta])
    _rewrite(os.path.join(out, "validation.jsonl"), summ)
    return out


def test_verify_d2_streak_resets():
    # met, not met, met: no trigger (recomputation from the summaries' bal_p)
    summ = []
    for u, b, streak in ((0, 0.5, None), (5, 0.7, 1), (10, 0.55, 0), (15, 0.7, 1)):
        summ.append({"kind": "summary", "update": u, "policy_sha": "p%d" % u, "task1": {"bal_p": b},
                     "d2": None if streak is None else {"met": b >= 0.6, "streak": streak, "triggered_at": None}})
    out = _meta_run(summ)
    rep = V.Report()
    V.check_v16(out, rep)
    assert rep.checks["rl.d2_trigger"]["fail"] == 0
    summ[3]["d2"]["triggered_at"] = 15                                      # a trigger without two in a row
    _rewrite(os.path.join(out, "validation.jsonl"), summ)
    rep = V.Report()
    V.check_v16(out, rep)
    assert rep.checks["rl.d2_trigger"]["fail"] >= 1


def test_n5_validation_without_task1_resets_the_streak():
    out = _meta_run([{"kind": "summary", "update": 0, "task1": {"bal_p": 0.5}, "d2": None},
                     {"kind": "summary", "update": 5, "task1": {"bal_p": 0.7}, "d2": {"met": True, "streak": 1, "triggered_at": None}},
                     {"kind": "summary", "update": 10, "task1": None, "d2": None},
                     {"kind": "summary", "update": 15, "task1": {"bal_p": 0.7}, "d2": {"met": True, "streak": 1, "triggered_at": None}}])
    rep = V.Report()
    V.check_v16(out, rep)
    assert rep.checks["rl.d2_trigger"]["fail"] == 0


# ------------------------------------------------------------------ selection / re-selection (v16 branch only)
def _selection_fails(out):
    rep = V.Report()
    V.check_selection(os.path.join(out, "validation.jsonl"), out, rep)
    return rep.checks["rl.selection"]["fail"]


def test_f2_v16_run_must_select_on_bal_p():
    sp, out = fixture("main")
    assert _selection_fails(out) == 0
    p = os.path.join(out, "validation.jsonl")
    rows = T.read_jsonl(p)
    for r in rows:
        if r["kind"] == "summary":
            r["selection_task1_metric"] = "term_f1"
            if r["selection_score"] is not None:
                r["selection_score"] += r["w_sel_task1"] * (r["task1"]["term_f1"] - r["task1"]["bal_p"])
    _rewrite(p, rows)
    assert _selection_fails(out) >= 1


def test_f3_reselect_summaries_recomputed():
    sp, out = fixture("main")
    p = os.path.join(out, "reselect.jsonl")
    rows = T.read_jsonl(p)
    for r in rows:
        if r["kind"] == "summary" and r["selection_score"] is not None:
            r["selection_score"] += 0.25
            break
    _rewrite(p, rows)
    assert v16_report(out, sp).checks["rl.selection"]["fail"] >= 1


def test_reselect_ids_seeds_and_best():
    sp, out = fixture("main")
    f0 = json.load(open(sp))["folds"][0]
    rep = V.Report()
    V.check_reselect_ids(out, set(f0["validation"]), set(f0["train_all"]), rep)
    assert no_fail(rep) and rep.checks["rl.reselect_seeds"]["n"] > 0
    p = os.path.join(out, "reselect.jsonl")
    rows = T.read_jsonl(p)
    for r in rows:
        if r["kind"] == "episode":
            r["seed"] = 99
            break
    _rewrite(p, rows)
    b = json.load(open(os.path.join(out, "reselect_best.json")))
    b["best"]["update"] = 99
    json.dump(b, open(os.path.join(out, "reselect_best.json"), "w"))
    rep = V.Report()
    V.check_reselect_ids(out, set(f0["validation"]), set(f0["train_all"]), rep)
    assert rep.checks["rl.reselect_seeds"]["fail"] == 1 and rep.checks["rl.reselect_best"]["fail"] == 1


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


# ------------------------------------------------------------------ intervention (archived run with one)
def test_intervention_verified_on_archived_run():
    sp, out = fixture("iv")
    rep = V.Report()
    V.check_intervention(out, rep)
    assert rep.checks["rl.intervention"]["fail"] == 0 and rep.checks["rl.intervention"]["n"] > 0
    assert no_fail(v16_report(out, sp))
    rows = T.read_jsonl(os.path.join(out, "updates.jsonl"))
    rows[-1]["cfg_used"]["w_dist"] = 0.5                                    # outside the intervention bounds
    _rewrite(os.path.join(out, "updates.jsonl"), rows)
    rep = V.Report()
    V.check_intervention(out, rep)
    assert rep.checks["rl.intervention"]["fail"] == 1


# ------------------------------------------------------------------ item 8: pooled Task 1 evaluation
def _write_fold(d, name, convs):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        for cid, (flags, _) in convs.items():
            for t, e in enumerate(flags, 1):
                f.write(json.dumps({"conversation_id": cid, "turn_index": t, "greedy_ended": e}) + "\n")
    with open(p + ".k1.jsonl", "w") as f:
        for cid, (_, k1) in convs.items():
            f.write(json.dumps({"conversation_id": cid, "ended": k1}) + "\n")
    return p


def test_item8_pooled():
    d = tempfile.mkdtemp()
    f0 = _write_fold(d, "a0.jsonl", {"c1": ([False, False, True], True), "c2": ([False, False], False)})
    f1 = _write_fold(d, "a1.jsonl", {"c3": ([False, False, False, False], True)})
    g0 = _write_fold(d, "b0.jsonl", {"c1": ([False, False, False], True), "c2": ([False, False], True)})
    g1 = _write_fold(d, "b1.jsonl", {"c3": ([False, True, False, False], False)})
    arm = TP.load_arm([f0, f1])
    m = TP.metrics(arm)
    # same numbers as task1_stop on equivalent rows (flags given directly: TP 2, FN 1, FP 1)
    assert m["n_conversations"] == 3 and m["n_turns"] == 9
    assert abs(m["term_f1"] - 4 / (4 + 1 + 1)) < 1e-12 and abs(m["premature_end_rate"] - 1 / 9) < 1e-12
    assert abs(m["premature"] - 1 / 3) < 1e-12 and abs(m["k1_end_rate"] - 2 / 3) < 1e-12
    res = TP.main(["--arm", "a=%s,%s" % (f0, f1), "--arm", "b=%s,%s" % (g0, g1), "--compare", "a", "b", "--n-boot", "500"])
    c = res["comparisons"][0]
    assert c["n_conversations"] == 3 and set(c["metrics"]) == set(TP.KEYS)
    assert TP.paired(TP.load_arm([f0, f1]), TP.load_arm([g0, g1]), 500) == TP.paired(TP.load_arm([f0, f1]), TP.load_arm([g0, g1]), 500)
    dup = _write_fold(d, "dup.jsonl", {"c1": ([False, True], True)})
    with pytest.raises(SystemExit):
        TP.load_arm([f0, dup])                                               # a conversation in two folds
    with pytest.raises(SystemExit):
        TP.paired(TP.load_arm([f0]), TP.load_arm([g0, g1]), 10)               # different conversation sets


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
