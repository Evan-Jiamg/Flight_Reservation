"""Pure-Python tests for the judge pipeline (no torch, no model, no server).

sampling/stratification, harmony final-channel parsing, cross-check agreement + gate, review sheet,
fold selection / manifest / leak assertions, validation metrics + gate, timing AUCs on synthetic
data with hand-computed answers, session bootstrap, eval input building, resumable scoring.
"""
import json
import os
import random
import subprocess
import sys
import tempfile

import eval_judge_timing as T
import eval_multiwoz_judge as MW
import judge_metrics as JM
import label_agreement as LA
import label_crosscheck as LC
import make_review_sheet as RS
import train_goal_judge_fold as TF

HERE = os.path.dirname(os.path.abspath(__file__))


def close(a, b, eps=1e-9):
    return a is not None and abs(a - b) < eps


# ---- sampling ---------------------------------------------------------------------------------------
def test_sampling():
    assert JM.allocate({"ditto_rep0": 680, "ditto_rep1": 680, "human": 154}, 200) == \
        {"ditto_rep0": 90, "ditto_rep1": 90, "human": 20}
    assert sum(JM.allocate({"a": 1, "b": 1, "c": 1}, 2).values()) == 2
    strata = {"ditto_rep0": ["r0_%d" % i for i in range(680)], "ditto_rep1": ["r1_%d" % i for i in range(680)],
              "human": ["h_%d" % i for i in range(154)]}
    s1 = JM.stratified_sample(strata, 200, 7)
    shuffled = {k: random.Random(3).sample(v, len(v)) for k, v in strata.items()}
    assert s1 == JM.stratified_sample(shuffled, 200, 7), "must not depend on input order"
    assert s1 != JM.stratified_sample(strata, 200, 8)
    assert len(set(s1)) == 200
    assert sum(x.startswith("h_") for x in s1) == 20 and sum(x.startswith("r0_") for x in s1) == 90
    try:
        JM.allocate({"a": 3}, 4)
        raise AssertionError
    except ValueError:
        pass
    # label_crosscheck.choose_ids only draws from parsed labels
    samples = [{"id": "x%d" % i, "source": "human" if i % 5 == 0 else "ditto_rep0"} for i in range(50)]
    labels = {"x%d" % i: {"status": "NOT", "unmet": []} for i in range(50) if i != 10}
    ids = LC.choose_ids(labels, samples, 20, 1)
    assert "x10" not in ids and len(ids) == 20 and ids == LC.choose_ids(labels, list(reversed(samples)), 20, 1)
    print("sampling ok")


# ---- harmony parsing ----------------------------------------------------------------------------------
def test_harmony():
    A = '<|channel|>analysis<|message|>They want X. Maybe {"status": "NOT", "unmet": []}?<|end|>'
    F = '<|start|>assistant<|channel|>final<|message|>{"status": "PARTIAL", "unmet": ["license"]}<|return|>'
    res, ok, info = LC.parse_harmony(A + F)
    assert ok and res == {"status": "PARTIAL", "unmet": ["license"]}, res       # final, not analysis
    assert info["channels"] == ["analysis", "final"] and info["final_terminated"]
    res, ok, _ = LC.parse_harmony(A + '<|start|>assistant<|channel|>final <|constrain|>json<|message|>'
                                  '{"status": "SATISFIED", "unmet": ["x"]}<|end|>')
    assert ok and res == {"status": "SATISFIED", "unmet": []}
    res, ok, info = LC.parse_harmony(A)                                        # analysis only: never guess
    assert not ok and res["status"] == "UNKNOWN" and info["reason"] == "no_final_channel"
    res, ok, info = LC.parse_harmony('<|channel|>analysis<|message|>long thinking that hit max_new ...')
    assert not ok and info["reason"] == "no_final_channel"
    res, ok, info = LC.parse_harmony(A + '<|start|>assistant<|channel|>final<|message|>I think partial.<|return|>')
    assert not ok and info["reason"] == "final_unparsed"
    res, ok, info = LC.parse_harmony(A + '<|start|>assistant<|channel|>final<|message|>{"status": "NOT", "unmet": ["a"]}')
    assert ok and res["status"] == "NOT" and info["final_terminated"] is False
    two = F.replace("PARTIAL", "NOT") + F
    res, ok, info = LC.parse_harmony(A + two)
    assert ok and res["status"] == "PARTIAL" and info["n_final"] == 2          # last final message
    res, ok, _ = LC.parse_harmony('{"status": "NOT", "unmet": []}')            # no harmony header
    assert not ok
    res, ok, _ = LC.parse_harmony(None)
    assert not ok
    print("harmony final-channel parsing ok")


# ---- cross-check agreement + gate ---------------------------------------------------------------------
def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_agreement(tmp):
    prim = os.path.join(tmp, "prim.jsonl")
    chk = os.path.join(tmp, "chk.jsonl")
    # 10 common: 8 agree. primary dist S4 P3 N3; check dist S4 P4 N2 (pairs below)
    pairs = [("SATISFIED", "SATISFIED")] * 4 + [("PARTIAL", "PARTIAL")] * 2 + [("NOT", "NOT")] * 2 + \
            [("PARTIAL", "NOT")] + [("NOT", "PARTIAL")]
    write_jsonl(prim, [{"id": "i%d" % k, "status": a, "unmet": [], "parse_ok": True} for k, (a, _) in enumerate(pairs)]
                + [{"id": "extra", "status": "NOT", "unmet": [], "parse_ok": True}])
    write_jsonl(chk, [{"id": "i%d" % k, "status": b, "unmet": [], "parse_ok": True} for k, (_, b) in enumerate(pairs)]
                + [{"id": "bad", "status": "UNKNOWN", "unmet": [], "parse_ok": False}])
    po = 8 / 10
    ca = {"SATISFIED": 4, "PARTIAL": 3, "NOT": 3}
    cb = {"SATISFIED": 4, "PARTIAL": 3, "NOT": 3}
    pe = sum(ca[c] * cb[c] for c in ca) / 100
    rep = LC.agreement(prim, chk, 0.4)
    assert rep["n_common"] == 10 and close(rep["cohen_kappa"], (po - pe) / (1 - pe)) and rep["gate"] == "PASS"
    assert LC.agreement(prim, chk, 0.9)["gate"] == "FAIL"
    assert LA.kappa([("NOT", "NOT")] * 3) is None
    # all-disagree -> FAIL
    write_jsonl(chk, [{"id": "i%d" % k, "status": "SATISFIED" if a != "SATISFIED" else "NOT", "unmet": [],
                       "parse_ok": True} for k, (a, _) in enumerate(pairs)])
    assert LC.agreement(prim, chk, 0.4)["gate"] == "FAIL"
    # CLI in --agreement-only mode: writes ids file, refuses a different subset on rerun
    samples = os.path.join(tmp, "samples.jsonl")
    write_jsonl(samples, [{"id": "i%d" % k, "source": "human" if k < 3 else "ditto_rep0"} for k in range(10)]
                + [{"id": "extra", "source": "human"}])
    out = os.path.join(tmp, "cc.jsonl")
    write_jsonl(out, [{"id": "i%d" % k, "status": b, "unmet": [], "parse_ok": True} for k, (_, b) in enumerate(pairs)])
    cmd = [sys.executable, os.path.join(HERE, "label_crosscheck.py"), "--samples", samples, "--labels", prim,
           "--model", "unused", "--out", out, "--n", "6", "--agreement-only"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    assert r.returncode == 0, r.stderr
    assert "KAPPA GATE (>= 0.40): PASS" in r.stdout, r.stdout[-400:]
    ids = open(out + ".ids.txt").read().split()
    assert len(ids) == 6
    r = subprocess.run(cmd[:-3] + ["--n", "5", "--agreement-only"], capture_output=True, text=True, cwd=HERE)
    assert r.returncode != 0 and "refusing" in (r.stdout + r.stderr)
    print("cross-check agreement / gate ok")


# ---- review sheet --------------------------------------------------------------------------------------
def test_review_sheet(tmp):
    samples = [{"id": "s%d" % i, "source": "human", "t": 2, "conversation_id": "c%d" % i,
                "scenario_text": "wants  plant\nimages", "hist_u": ["hi %d" % i, "more\n\nplease"],
                "hist_a": ["here A", "here B"]} for i in range(40)]
    labels = {"s%d" % i: {"status": "PARTIAL", "unmet": ["metadata"]} for i in range(40) if i % 4}
    ids = RS.choose(labels, samples, 10, 5)
    assert ids == RS.choose(labels, samples, 10, 5) and all(i in labels for i in ids) and len(set(ids)) == 10
    sp, lp = os.path.join(tmp, "rs_s.jsonl"), os.path.join(tmp, "rs_l.jsonl")
    write_jsonl(sp, samples)
    write_jsonl(lp, [{"id": k, **v, "parse_ok": True} for k, v in labels.items()]
                + [{"id": "s0", "status": "UNKNOWN", "unmet": [], "parse_ok": False}])
    md = os.path.join(tmp, "review.md")
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_review_sheet.py"), "--samples", sp, "--labels", lp,
                        "--out", md, "--n", "10", "--seed", "5"], capture_output=True, text=True, cwd=HERE)
    assert r.returncode == 0, r.stderr
    text = open(md, encoding="utf-8").read()
    assert text.count("## ") >= 10 and text.count("- status: **PARTIAL**") == 10
    assert "> wants plant images" in text and "**ASSISTANT (2):**" in text and "  - metadata" in text
    assert "seed 5; n 10 of 30 parsed labels" in text
    print("review sheet ok")


# ---- fold selection / manifest / leakage -------------------------------------------------------------
def make_split():
    return {"folds": [{"fold": 0, "train": ["a", "b"], "validation": ["v"], "test": ["x"],
                       "train_all": ["a", "b", "c"], "validation_all": ["v", "w"], "test_all": ["x", "y"],
                       "forbidden_for_training": ["v", "w", "x", "y"]}]}


def test_selection_and_leaks(tmp):
    fs = JM.fold_of(make_split(), 0)
    samples = [{"id": "%s%d" % (c, i), "conversation_id": c, "source": "human", "t": i}
               for c in "abcvwxy" for i in (1, 2)]
    labels = {s["id"]: {"status": "NOT", "unmet": []} for s in samples if s["id"] != "a2"}
    train, val, st = TF.select(samples, labels, fs)
    assert {s["conversation_id"] for s in train} == {"a", "b"} and len(train) == 3
    assert {s["conversation_id"] for s in val} == {"v"}
    assert st["train_unlabelled_or_unparsed"] == 1 and st["samples_outside_train_and_validation"] == 8
    bad = json.loads(json.dumps(fs))
    bad["train"].append("x")                               # a corrupted split file must be refused
    for fn in (lambda: TF.select(samples, labels, bad), lambda: JM.assert_no_leak(["a", "x"], fs),
               lambda: JM.assert_no_leak(["c"], fs)):     # c is in train_all but not in train
        try:
            fn()
            raise AssertionError("leak not detected")
        except AssertionError as e:
            assert "leak not detected" not in str(e)
    JM.assert_no_leak(["a", "b"], fs)
    try:
        JM.fold_of(make_split(), 3)
        raise AssertionError
    except ValueError:
        pass
    man = {"fold": 0, "split_file_sha256": "abc", "train_scenarios": ["a", "b"]}
    JM.check_manifest(man, 0, "abc", ["v", "w"])
    for args in ((1, "abc", ["v"]), (0, "zzz", ["v"]), (0, "abc", ["v", "b"])):
        try:
            JM.check_manifest(man, *args)
            raise AssertionError("manifest check passed wrongly")
        except AssertionError as e:
            assert "passed wrongly" not in str(e)
    p = os.path.join(tmp, "train_manifest.json")
    assert TF.write_or_check_manifest(p, {"fold": 0, "x": 1}) is True
    assert TF.write_or_check_manifest(p, {"fold": 0, "x": 1}) is False
    try:
        TF.write_or_check_manifest(p, {"fold": 0, "x": 2})
        raise AssertionError
    except SystemExit as e:
        assert "differs" in str(e)
    # the split guard of the timing eval
    sp = make_split()
    assert T.resolve_split_ids(sp, 0, "validation", False, False) == ["v", "w"]
    assert T.resolve_split_ids(sp, 0, "validation", False, True) == ["v"]
    try:
        T.resolve_split_ids(sp, 0, "test", False, False)
        raise AssertionError
    except SystemExit as e:
        assert "--final" in str(e)
    assert T.resolve_split_ids(sp, 0, "test", True, False) == ["x", "y"]
    print("selection / manifest / leak assertions ok")


# ---- validation metrics + gate -----------------------------------------------------------------------
def test_eval_rows():
    g = ["SATISFIED", "PARTIAL", "NOT", "NOT"]
    p = ["SATISFIED", "NOT", "NOT", "UNKNOWN"]
    f1 = JM.per_class_f1(g, p)
    assert f1["SATISFIED"] == 1.0 and f1["PARTIAL"] == 0.0 and close(f1["NOT"], 2 / 4)
    assert close(JM.macro_f1(g, p), 0.5) and JM.accuracy(g, p) == 0.5
    probs = [{"SATISFIED": 1.0, "PARTIAL": 0.0, "NOT": 0.0}, {"SATISFIED": 0.0, "PARTIAL": 0.5, "NOT": 0.5}]
    assert close(JM.brier(["SATISFIED", "PARTIAL"], probs), (0 + 0.5) / 2)
    uni = {c: 1 / 3 for c in JM.CLASSES}
    assert close(JM.brier(["NOT"], [uni]), 2 / 3) and close(JM.nll(["NOT"], [uni]), 1.0986122886681098)
    rows = [{"label": "NOT", "pred": "NOT", "parse_ok": True, "probs": uni, "source": "human"}] * 49 + \
           [{"label": "SATISFIED", "pred": "UNKNOWN", "parse_ok": False, "probs": uni, "source": "ditto_rep0"}]
    rep = TF.evaluate_rows(rows, 0.5, 0.02)
    assert rep["unparsed_rate"] == 0.02 and rep["gate"]["result"] == "FAIL"     # strict < 2%
    rep = TF.evaluate_rows(rows[:49] + [dict(rows[49], pred="SATISFIED", parse_ok=True)], 0.5, 0.02)
    assert rep["gate"]["result"] == "PASS" and rep["macro_f1"] == 1.0 and set(rep["by_source"]) == {"human", "ditto_rep0"}
    print("validation metrics / gate ok")


# ---- timing AUCs -------------------------------------------------------------------------------------
def brute_auc(pos, neg):
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def test_timing():
    rng = random.Random(0)
    for _ in range(50):
        pos = [rng.choice([0.1, 0.2, 0.3, rng.random()]) for _ in range(rng.randint(1, 6))]
        neg = [rng.choice([0.1, 0.2, 0.3, rng.random()]) for _ in range(rng.randint(1, 6))]
        assert close(JM.auc(pos, neg), brute_auc(pos, neg))
    assert JM.auc([], [1]) is None and JM.auc([1, 1], [1]) == 0.5
    sess = {"A": {"K": 2, "scores": {1: .1, 2: .9}}, "B": {"K": 3, "scores": {1: .2, 2: .5, 3: .8}},
            "C": {"K": 2, "scores": {1: .3, 2: .4}}}
    m = JM.timing_metrics(JM.timing_points(sess, "last"))
    assert (m["n_pos"], m["n_neg"], m["within_turn_pairs"]) == (3, 4, 2)
    assert close(m["overall_auc"], 11 / 12) and close(m["turn_only_auc"], 11 / 12) and close(m["within_turn_auc"], 0.5)
    m = JM.timing_metrics(JM.timing_points(sess, "prefinal"))
    assert (m["n_pos"], m["n_neg"]) == (3, 1) and close(m["overall_auc"], 2 / 3) and close(m["within_turn_auc"], 0.5)
    # a score that is the turn index itself carries no within-turn signal
    many = {}
    for i in range(40):
        K = rng.randint(1, 8)
        many["s%d" % i] = {"K": K, "scores": {t: float(t) for t in range(1, K + 1)}}
    m = JM.timing_metrics(JM.timing_points(many, "last"))
    assert close(m["within_turn_auc"], 0.5) and close(m["overall_auc"], m["turn_only_auc"])
    # a score that marks the last state perfectly -> within-turn AUC 1
    for s in many.values():
        s["scores"] = {t: (1.0 if t == s["K"] else 0.0) for t in s["scores"]}
    m = JM.timing_metrics(JM.timing_points(many, "last"))
    assert m["within_turn_auc"] == 1.0 and m["overall_auc"] == 1.0
    ci = JM.session_bootstrap(JM.timing_points(many, "last"), 200, 1)
    assert ci["overall_auc"]["lo"] == 1.0 and ci["within_turn_auc"]["n_defined"] <= 200
    assert ci == JM.session_bootstrap(JM.timing_points(many, "last"), 200, 1)          # seeded
    ci = JM.session_bootstrap(JM.timing_points(sess, "last"), 300, 2)
    assert 0 <= ci["overall_auc"]["lo"] <= 11 / 12 <= ci["overall_auc"]["hi"] <= 1
    print("timing AUCs / bootstrap ok")


# ---- eval inputs and resumable scoring --------------------------------------------------------------
class FakeJudge:
    def __init__(self):
        self.calls = 0

    def prompt_text(self, goal, hu, ha):
        return "%s|%d|%d" % (goal, len(hu), len(ha)), {"prompt_tokens": 5, "dropped_exchanges": 0}

    def _score_prompt(self, text):
        self.calls += 1
        n = int(text.split("|")[1])
        s = min(0.9, 0.2 * n)
        return {"SATISFIED": s, "PARTIAL": (1 - s) / 2, "NOT": (1 - s) / 2}, 0.99


def test_eval_inputs(tmp):
    si = T.session_inputs("g", ["u1", "u2", "u3"], ["a1", "a2"])
    assert si["K"] == 2 and si["unanswered_final_user_message"] and si["states"][1] == (2, ["u1", "u2"], ["a1", "a2"])
    recs = [{"cid": "m1", "goal_message": ["Find a", "cheap hotel."], "turns": [
                {"user": "hi", "assistant": "hello"}, {"user": "cheap", "assistant": "X hotel"},
                {"user": "thanks bye", "assistant": "bye"}]},
            {"cid": "m2", "goal_message": ["Book a train."], "turns": [
                {"user": "train", "assistant": "when?"}, {"user": "bye", "assistant": ""}]}]
    sess, st = MW.multiwoz_sessions(recs)
    assert sess["m1"]["goal"] == "Find a cheap hotel." and sess["m1"]["K"] == 3
    assert sess["m2"]["K"] == 1 and st["stopped_at_empty_reply"] == 1 and st["stopped_at_final_turn"] == 1
    judge = FakeJudge()
    cache = os.path.join(tmp, "scores.jsonl")
    scored = T.score_sessions(judge, sess, cache)
    assert judge.calls == 4 and close(scored["m1"]["probs"][3]["SATISFIED"], 0.6)
    judge2 = FakeJudge()
    assert T.score_sessions(judge2, sess, cache) == scored and judge2.calls == 0      # resumed, no rescoring
    rep = T.report(scored, 50, 3)
    assert set(rep) == {"%s|%s" % (s, m) for s in T.SCORES for m in T.MODES}
    assert rep["p_satisfied|last"]["point"]["n_pos"] == 2
    print("eval inputs / resumable scoring ok")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_sampling()
        test_harmony()
        test_agreement(tmp)
        test_review_sheet(tmp)
        test_selection_and_leaks(tmp)
        test_eval_rows()
        test_timing()
        test_eval_inputs(tmp)
    print("JUDGE PIPELINE TESTS OK")


if __name__ == "__main__":
    main()
