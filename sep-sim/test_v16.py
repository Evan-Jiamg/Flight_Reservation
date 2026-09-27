"""SPEC v16 (ops/SPEC_v16_grpo_opt.md, user 2026-09-28): one or more tests per item; dry-run loop, no GPU."""
import json
import math
import os
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
from test_rl_advantages import args, make_splits  # noqa: E402


def run(sp, out, *extra, updates=4, controller="llm"):
    return T.main(args(sp, out, *extra, controller=controller, updates=updates))


def fresh(*extra, updates=4):
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    run(sp, out, *extra, updates=updates)
    return sp, out


def v16_report(out):
    rep = V.Report()
    V.check_v16(out, rep)
    return rep


# ------------------------------------------------------------------ item 3: Dr. GRPO advantages
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


# ------------------------------------------------------------------ item 6: w_dist floor
def test_item6_w_dist_floor():
    c = RC.LLMFactorController(RC.initial_cfg(version="v4", lambda_unparsed=1.0, lambda_hit_max_new=1.0),
                               transport=lambda req: {"choices": [{"message": {"content": json.dumps(
                                   {"factors": {"w_dist": 0.5}, "rationale": "r"})}}]})
    assert c.opt["bounds"]["w_dist"] == [1.0, 5.0]
    hist = [{"split": "train", "update": u, "reward_mean": 0.0, "shadow_reward_mean": 0.0, "components_mean": {}}
            for u in range(1, 6)]
    assert c.propose(hist)["w_dist"] == 1.0


# ------------------------------------------------------------------ item 4: aux floor + controller prompt
def test_item4_floor_args():
    base = ["--fold", "0", "--out", "o", "--splits", "s"]
    with pytest.raises(SystemExit):                                             # not dry-run: the value is required
        T.parse_args(base + ["--planner-path", "Qwen3-4B-Instruct-2507"])
    a = T.parse_args(base + ["--dry-run"])
    assert a.stop_sup_floor == 0.0
    with pytest.raises(SystemExit):
        T.parse_args(base + ["--dry-run", "--stop-sup-weight", "0", "--stop-sup-floor", "0.5", "--ablation", "x"])
    with pytest.raises(SystemExit):
        T.parse_args(base + ["--dry-run", "--stop-sup-floor", "6"])
    a = T.parse_args(base + ["--planner-path", "Qwen3-4B-Instruct-2507", "--stop-sup-floor", "0.5"])
    assert a.stop_sup_floor == 0.5 and a.task1_G == 8 and a.task1_convs == 8 and a.t1_trigger_margin == 0.10
    with pytest.raises(SystemExit):                                             # off-spec values need --ablation
        T.parse_args(base + ["--dry-run", "--task1-G", "4"])


def test_item4_floor_weight():
    class A:
        stop_sup_floor, stop_sup_anneal = 0.5, 10
    tr = T.Trainer.__new__(T.Trainer)
    tr.a, tr.aux_anneal_start = A(), None
    assert tr.aux_weight(3, {"w_aux": 1.0}) == 1.0
    tr.aux_anneal_start = 10
    assert abs(tr.aux_weight(13, {"w_aux": 1.0}) - 0.7) < 1e-12
    assert tr.aux_weight(18, {"w_aux": 1.0}) == 0.5                             # 0.2 -> floor
    assert tr.aux_weight(40, {"w_aux": 1.0}) == 0.5
    assert tr.aux_weight(3, {"w_aux": 0.2}) == 0.5                              # controller below the floor
    A.stop_sup_floor = 0.0
    assert tr.aux_weight(40, {"w_aux": 1.0}) == 0.0                             # floor 0 = the old behaviour


def test_item4_controller_prompt_names_floor():
    sp, out = fresh("--stop-sup-floor", "0.5", updates=5)
    log = [json.loads(l) for l in open(os.path.join(out, "llm_controller.jsonl"))]
    sysmsg = [r for r in log if r.get("request")][0]["request"]["messages"][0]["content"]
    assert "fixed floor of 0.5" in sysmsg and "annealed to 0" not in sysmsg
    for u in T.read_jsonl(os.path.join(out, "updates.jsonl")):
        assert u["train_aggregate"]["aux_weight"] >= 0.5 and u["train_aggregate"]["aux_floor"] == 0.5


# ------------------------------------------------------------------ items 1 + 2: Task 1 groups, refill
def test_items12_task1_groups_and_refill():
    sp, out = fresh(updates=4)
    rows = T.read_jsonl(os.path.join(out, "rollouts_task1.jsonl"))
    assert rows and all(len(r["samples"]) == 8 for r in rows)
    sp_ = json.load(open(sp))["folds"][0]
    upd = {u["update"]: u for u in T.read_jsonl(os.path.join(out, "updates.jsonl"))}
    saw_refill = False
    for u in range(1, 5):
        rs = [r for r in rows if r["update"] == u]
        base = {r["conversation_id"] for r in rs if not r["refill"]}
        ref = {r["conversation_id"] for r in rs if r["refill"]}
        assert len(base) == 8 and not base & ref and len(ref) <= 8
        assert (base | ref) <= set(sp_["train_all"])
        th = upd[u]["train_aggregate"]["task1_train"]
        assert th["n_base_groups"] == sum(1 for r in rs if not r["refill"])
        assert th["n_refill_groups"] == sum(1 for r in rs if r["refill"])
        assert th["refill_stop"] in ("filled", "pool_empty", "cap", "none_needed")
        info = sum(1 for r in rs if RA.pstd([x["reward"] for x in r["samples"]]) > 1e-8)
        assert th["n_informative_groups"] == info
        if ref:
            saw_refill = True
            assert th["refill_stop"] != "none_needed"
        else:
            assert th["refill_stop"] in ("none_needed",) or info >= th["n_base_groups"] or th["refill_stop"] == "cap"
        # aux only from the base groups
        assert upd[u]["learner_stats"]["aux_n"] <= th["n_base_groups"]
    assert saw_refill, "the dry run never exercised the refill"
    rep = v16_report(out)
    assert all(c["fail"] == 0 for c in rep.checks.values()), {k: c for k, c in rep.checks.items() if c["fail"]}


def test_item1_refill_same_after_resume():
    d = tempfile.mkdtemp()
    sp = make_splits(d)
    a, b = os.path.join(d, "a"), os.path.join(d, "b")
    run(sp, a, updates=4)
    run(sp, b, updates=2)
    run(sp, b, "--resume", updates=4)
    key = lambda r: (r["update"], r["conversation_id"], r["t"])
    ra = sorted(((key(r), r["refill"], [x["reward"] for x in r["samples"]]) for r in T.read_jsonl(os.path.join(a, "rollouts_task1.jsonl"))))
    rb = sorted(((key(r), r["refill"], [x["reward"] for x in r["samples"]]) for r in T.read_jsonl(os.path.join(b, "rollouts_task1.jsonl"))))
    assert ra == rb


# ------------------------------------------------------------------ item 7: continuous Task 1 metric
def test_item7_prob_metrics_values():
    pts = [{"real_final": True, "p_end": 0.8}, {"real_final": True, "p_end": 0.4},
           {"real_final": False, "p_end": 0.2}, {"real_final": False, "p_end": 0.4}, {"real_final": False, "p_end": 0.0, "valid": False}]
    m = T1.task1_prob_metrics(pts)
    assert abs(m["bal_p"] - (0.5 * 0.6 + 0.5 * (0.8 + 0.6 + 1.0) / 3)) < 1e-12
    # pairs (final, earlier): 0.8>0.2,0.8>0.4,0.8>0 ; 0.4>0.2, 0.4=0.4 (1/2), 0.4>0  -> 5.5 / 6
    assert abs(m["auc"] - 5.5 / 6) < 1e-12
    ll = [-math.log(0.8), -math.log(0.4), -math.log(0.8), -math.log(0.6), -math.log(1 - 1e-6)]
    assert abs(m["logloss"] - sum(ll) / 5) < 1e-12
    assert m["n_points"] == 5 and m["n_final"] == 2 and m["n_invalid"] == 1
    assert T1.task1_prob_metrics([{"real_final": True, "p_end": 0.3}])["bal_p"] == 0.3
    with pytest.raises(ValueError):
        T1.task1_prob_metrics([{"real_final": False, "p_end": 0.3}])
    with pytest.raises(ValueError):
        T1.task1_prob_metrics([{"real_final": True, "p_end": 1.3}])


def test_item7_validation_selection_and_verify():
    sp, out = fresh(updates=4)
    val = T.read_jsonl(os.path.join(out, "validation.jsonl"))
    summ = [v for v in val if v["kind"] == "summary"]
    for v in summ:
        assert v["selection_task1_metric"] == "bal_p" and "bal_p" in v["task1"] and "auc" in v["task1"]
        rows = [r for r in val if r["kind"] == "task1" and r["update"] == v["update"]]
        pts = [p for r in rows for p in r["end_probs"]]
        assert len(pts) == sum(len(r["task1"]["turns"]) - 1 for r in rows)
        assert all("user_prompt" not in x for r in rows for x in r["task1"]["turns"])      # prompts not logged
        assert abs(T1.task1_prob_metrics(pts)["bal_p"] - v["task1"]["bal_p"]) < 1e-12
        if v["selection_score"] is not None:
            want = (v["w_sel_cov"] * v["turn_stats"]["coverage_mean"] - v["w_sel_w1"] * v["turn_stats"]["turn_w1"]
                    + v["w_sel_task1"] * v["task1"]["bal_p"])
            assert abs(v["selection_score"] - want) < 1e-12
    st = json.load(open(os.path.join(out, "ckpt", "u00004", "state.json")))
    assert "bal_p" in st["task1_base"]
    rep = v16_report(out)
    assert all(c["fail"] == 0 for c in rep.checks.values())
    # verify catches a tampered summary bal_p
    p = os.path.join(out, "validation.jsonl")
    rows = T.read_jsonl(p)
    for r in rows:
        if r["kind"] == "summary" and r["update"] == 2:
            r["task1"]["bal_p"] += 0.1
    with open(p, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    rep = v16_report(out)
    assert rep.checks["rl.task1_prob"]["fail"] == 1


def test_item7_d2_needs_two_consecutive():
    # margin -1: every validation after the base is "over the margin" -> trigger exactly at the 2nd one (u4)
    sp, out = fresh("--t1-trigger-margin", "-1", "--ablation", "test-d2", updates=6)
    summ = [v for v in T.read_jsonl(os.path.join(out, "validation.jsonl")) if v["kind"] == "summary"]
    d2 = {v["update"]: v["d2"] for v in summ}
    assert d2[0] is None and d2[2]["streak"] == 1 and d2[2]["triggered_at"] is None
    assert d2[4]["streak"] == 2 and d2[4]["triggered_at"] == 4
    st = json.load(open(os.path.join(out, "ckpt", "u00006", "state.json")))
    assert st["aux_anneal_start"] == 4
    upd = {u["update"]: u for u in T.read_jsonl(os.path.join(out, "updates.jsonl"))}
    assert upd[5]["train_aggregate"]["aux_weight"] < upd[5]["cfg_used"]["w_aux"]      # annealing after u4
    assert upd[4]["train_aggregate"]["aux_weight"] == upd[4]["cfg_used"]["w_aux"]
    rep = v16_report(out)
    assert all(c["fail"] == 0 for c in rep.checks.values())
    # margin +1: never triggered, never annealed
    sp, out = fresh("--t1-trigger-margin", "1", "--ablation", "test-d2", updates=4)
    summ = [v for v in T.read_jsonl(os.path.join(out, "validation.jsonl")) if v["kind"] == "summary"]
    assert all((v["d2"] or {}).get("triggered_at") is None for v in summ)


def test_verify_catches_early_anneal_and_floor():
    sp, out = fresh("--stop-sup-floor", "0.3", updates=4)
    p = os.path.join(out, "updates.jsonl")
    rows = T.read_jsonl(p)
    rows[1]["train_aggregate"]["aux_weight"] = 0.2                            # below floor AND an anneal without trigger
    with open(p, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    rep = v16_report(out)
    assert rep.checks["rl.aux_floor"]["fail"] == 1 and rep.checks["rl.d2_trigger"]["fail"] == 1


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
