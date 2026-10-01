"""SPEC v18 (ops/SPEC_v18_multiobj_rerank.md §12): dry-run tests, no GPU. The reward components, the fixed-scale
advantages and kappa, turn 1, clean_v18, the selector + reranker, the labeller, the reranker fit, the selection / stop
rules, resume, final.json, the verifier (and that it catches tampering), the test evaluation, the benchmark generation
gates. v17 behaviour is pinned by test_v17.py (the default --spec stays v17)."""
import copy
import json
import os
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rl_algos as RA  # noqa: E402
import rl_reward as RR  # noqa: E402
import style_select as SS  # noqa: E402
import train_planner_rl as T  # noqa: E402
import v18_fixtures as F  # noqa: E402
import v18_rules as V18  # noqa: E402
import verify_pipeline as V  # noqa: E402
import verify_v18 as VV  # noqa: E402
from test_rl_advantages import make_splits  # noqa: E402
from test_bench_tf_generate import world  # noqa: E402,F401  (the benchmark-generation fixture)

E1R = os.environ.get("E1R_TREE") or os.path.abspath(os.path.join(HERE, "..", "audit_e1r"))
HAVE_TREE = os.path.isdir(os.path.join(E1R, "sepsim"))
pytestmark = pytest.mark.skipif(not HAVE_TREE, reason="E1.6 tree (sepsim.acts) not present")


def fresh(*extra, updates=3, margin=100.0, passed=True):
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    T.main(F.v18_args(d, sp, out, *extra, updates=updates, margin=margin, passed=passed))
    return d, sp, out


def report(out, sp):
    rep = V.Report()
    VV.check_v18(out, rep, sp)
    return rep


def fails(rep, skip=("rl.v18_labels",)):
    """Failing checks (the dry run has no label_acts / train_reranker records: rl.v18_labels is checked separately)."""
    return {k: c["examples"] for k, c in rep.checks.items() if c["fail"] and k not in skip}


def rewrite(path, rows):
    with open(path, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


# ================================================================== §3.1.2 r_act
def D(*entries):
    return [dict(zip(("move", "act", "p", "length_words"), e)) for e in entries]


LAB = {"q7": {"Inquire": 2 / 3, "Navigate": 1 / 3}, "n_valid_votes": 3}


def test_r_act_drop_rules_sum_and_uniform():
    # a non-dict, a malformed act ("other" while the act is not "other"), a non-float p, p <= 0 are dropped / not counted
    dist = ["junk", {"move": "Inquire", "act": "nonsense", "p": 0.5}, {"move": "Navigate", "act": "detail", "p": "x"},
            {"move": "Note", "act": "confirm", "p": -1}, {"move": "Inquire", "act": "ask_more", "p": 0.2},
            {"move": "Inquire", "act": "ask_more", "p": 0.2}, {"move": "Navigate", "act": "rank", "p": 0.2}]
    p, fb = RR.plan_act_probs(dist)
    assert not fb and abs(p["Inquire"] - 2 / 3) < 1e-12 and abs(p["Navigate"] - 1 / 3) < 1e-12   # same move summed
    r, why, _ = RR.r_act_of(dist, LAB, 2, 5)
    assert why is None and abs(r - 1.0) < 1e-12
    # all mass on Complete / Other -> the uniform distribution over A (fallback flagged)
    p, fb = RR.plan_act_probs(D(("Complete", "settle", 1.0, 3), ("Other", "other", 0.5, 4)))
    assert fb and all(abs(v - 0.2) < 1e-12 for v in p.values())
    r, _, fb = RR.r_act_of(D(("Complete", "settle", 1.0, 3)), LAB, 2, 5)
    want = 1 - 0.5 * ((0.2 - 2 / 3) ** 2 + (0.2 - 1 / 3) ** 2 + 3 * 0.2 ** 2)
    assert fb and abs(r - want) < 1e-12


def test_r_act_skips():
    d = D(("Inquire", "ask_more", 1.0, 5))
    assert RR.r_act_of(d, LAB, 5, 5)[1] == "final"                                    # t = n
    assert RR.r_act_of(d, {"q7": {"Complete": 1.0}, "n_valid_votes": 3}, 2, 5)[1] == "complete"
    assert RR.r_act_of(d, {"q7": {"Other": 2 / 3, "Inquire": 1 / 3}, "n_valid_votes": 3}, 2, 5)[1] == "other"
    assert RR.r_act_of(d, {"q7": {"Inquire": 1.0}, "n_valid_votes": 1}, 2, 5)[1] == "votes"
    assert RR.r_act_of(d, None, 2, 5)[1] == "votes"
    # a tie is broken by acts.COARSE_ORDER: Inquire before Navigate; Complete before Other
    assert RR.label_majority({"Navigate": 0.5, "Inquire": 0.5}) == "Inquire"
    assert RR.label_majority({"Other": 0.5, "Complete": 0.5}) == "Complete"
    # Other / Complete votes are removed from q before the Brier score
    r, why, _ = RR.r_act_of(D(("Inquire", "ask_more", 1.0, 5)), {"q7": {"Inquire": 2 / 3, "Other": 1 / 3},
                                                                 "n_valid_votes": 3}, 2, 5)
    assert why is None and abs(r - 1.0) < 1e-12


# ================================================================== §3.1.3 r_len
def test_r_len_entry_band_missing_other():
    d = D(("Inquire", "ask_more", 0.6, "x"), ("Inquire", "ask_more", 0.2, 14), ("Inquire", "ask_more", 0.2, 99),
          ("Note", "confirm", 0.0, 5))
    # the first entry of m* with a positive integer length: 14 (the "x" one is skipped)
    assert RR.plan_length_for(d, "Inquire") == 14
    r, why = RR.r_len_of(d, LAB, 20)              # rho = 15 / 21 = 0.714 >= 2/3 -> 1
    assert why is None and r == 1.0
    r, _ = RR.r_len_of(d, LAB, 40)                # rho = 15 / 41; / (2/3)
    assert abs(r - (15 / 41) / (2 / 3)) < 1e-12
    r, why = RR.r_len_of(D(("Note", "confirm", 1.0, 5)), LAB, 20)
    assert r == 0.0 and why == "missing"          # no entry of m*
    assert RR.r_len_of(d, {"q7": {"Other": 1.0}, "n_valid_votes": 3}, 20) == (None, "other")
    assert RR.r_len_of(d, {"q7": {"Inquire": 1.0}, "n_valid_votes": 1}, 20) == (None, "votes")
    # Complete as m* is applicable for r_len (the close's own length)
    r, why = RR.r_len_of(D(("Complete", "settle", 1.0, 3)), {"q7": {"Complete": 1.0}, "n_valid_votes": 2}, 3)
    assert why is None and r == 1.0


def test_task1_components_states():
    lab = LAB
    x = {"status": "valid", "p_end": 0.3, "act_distribution_raw": D(("Inquire", "ask_more", 1.0, 10))}
    c = RR.task1_components(x, 3, 5, lab, 10)
    assert abs(c["r_stop"] - (1 - 0.09)) < 1e-12 and c["r_fmt"] == 1.0 and c["r_act"] is not None and c["r_len"] == 1.0
    c = RR.task1_components(dict(x, p_end=None), 1, 5, lab, 10)             # turn 1: no r_stop
    assert c["r_stop"] is None and c["r_act"] is not None
    c = RR.task1_components({"status": "invalid"}, 3, 5, lab, 10)          # invalid: only r_fmt = 0
    assert c["r_fmt"] == 0.0 and c["r_stop"] is None and c["r_act"] is None and c["r_len"] is None
    c = RR.task1_components({"status": "dropped"}, 3, 5, lab, 10)
    assert all(c[k] is None for k in ("r_stop", "r_act", "r_len", "r_fmt"))
    c = RR.task1_components(dict(x, p_end=0.9), 5, 5, lab, 10)              # t = n: y = 1
    assert abs(c["r_stop"] - 0.99) < 1e-12 and c["r_act"] is None and c["skip"]["act"] == "final"


# ================================================================== §3.2 Task 2 components, clean_v18
def ep(T_, H, steps=None, counters=None, capped=0, compacted=0, cov=0.5):
    tr = steps or [{"t": i + 1} for i in range(max(T_, 1))]
    return {"emitted_user_turns": T_, "human_turns": H, "decision_steps": len(tr), "trace": tr,
            "episode_counters": counters or {}, "emitted_capped_steps": capped, "compacted_steps": compacted,
            "coverage": cov}


def test_reward_v5_and_coverage_diag():
    r = RR.reward_v5(ep(3, 5))
    assert r["r_turn"] == 1 - 2 / 10 and r["r_fmt2"] == 1.0
    assert RR.reward_v5(ep(10, 14))["r_turn"] == 1.0                # min(H, t_max)
    st = [{"t": 1, "planner_unparsed": True, "planner_hit_max_new": True}, {"t": 2, "no_survivor": True}, {"t": 3},
          {"t": 4}]
    assert RR.reward_v5(ep(4, 4, steps=st))["r_fmt2"] == 1 - 2 / 4  # a step counted once
    with pytest.raises(ValueError):
        RR.reward_v5(ep(3, 5), w_cov=0.5)
    assert RR.coverage_diag(ep(3, 5, counters={"judge_error": 1})) is None
    assert RR.coverage_diag(ep(3, 5, cov=0.25)) == 0.25


def test_clean_v18_definition():
    import task2_env as TE
    assert RR.clean_v18(ep(3, 5, counters={"judge_empty": 1, "judge_unparseable": 2, "judge_error": 1}))
    for bad in (ep(3, 5, counters={"r0_len_truncated": 1}), ep(3, 5, counters={"r0_empty": 1}), ep(3, 5, capped=1),
                ep(3, 5, compacted=1), ep(0, 5)):
        assert not RR.clean_v18(bad)
    assert TE.episode_clean_v18({"judge_error": 3}, 0, 0, 2) and not TE.episode_clean_v18({"r0_empty": 1}, 0, 0, 2)
    assert not TE.episode_clean({"judge_error": 1})                 # the v17 definition keeps excluding it


# ================================================================== §4 fixed scales, z, kappa, masks
def test_measure_scales_spread_only_and_freezing():
    groups = [[{"a": 0.0}, {"a": 1.0}], [{"a": 0.5}, {"a": 0.5}], [{"a": None}, {"a": 0.2}], [{"a": 0.0}, {"a": 0.5}]]
    m = RA.measure_scales(groups, ["a"], {"a": 2}, 0.01)
    # only groups with spread: |c| = 0.5, 0.5, 0.25, 0.25 -> 0.375 (the no-spread / < 2 applicable groups not diluting)
    assert m["a"]["n_spread_groups"] == 2 and abs(m["a"]["scale"] - 0.375) < 1e-12 and not m["a"]["frozen"]
    assert RA.measure_scales(groups, ["a"], {"a": 3}, 0.01)["a"]["frozen"]                  # < 3 spread groups
    tiny = [[{"a": 0.0}, {"a": 0.001}]] * 3
    assert RA.measure_scales(tiny, ["a"], {"a": 3}, 0.01)["a"]["why"].startswith("scale")    # < 0.01
    f = RA.measure_scales(groups, ["fmt"], {}, 0.01, fixed={"fmt": 0.5})["fmt"]
    assert f["scale"] == 0.5 and f["why"] == "fixed" and not f["frozen"]


def test_z_clip_frozen_and_not_applicable():
    g = [{"k": 0.0, "j": None}, {"k": 1.0, "j": 0.3}, {"k": 0.5, "j": None}]
    rows, spread = RA.fixed_scale_z(g, ["k", "j"], {"k": 0.1, "j": 1.0}, set(), 3)
    assert [r["z"]["k"] for r in rows] == [-3.0, 3.0, 0.0] and rows[0]["clipped"]["k"]
    assert all(r["z"]["j"] == 0.0 for r in rows) and not spread["j"]                          # < 2 applicable -> 0
    rows, _ = RA.fixed_scale_z(g, ["k"], {"k": 0.1}, {"k"}, 3)
    assert all(r["z"]["k"] == 0.0 for r in rows)                                              # frozen -> 0
    # bounds (§4.2): |A_plan| <= 3 kappa (w_act + w_len + w_fmt)
    w = {"stop": 1, "act": 1, "len": 0.5, "fmt": 1}
    m = [{"stop": 0.0, "act": 0.0, "len": 0.0, "fmt": 0.0}, {"stop": 1.0, "act": 1.0, "len": 1.0, "fmt": 1.0}]
    rows, _ = V18.advantages_t1(m, {"stop": 1e-3, "act": 1e-3, "len": 1e-3, "fmt": 1e-3}, set(), 3, w, 0.1)
    assert abs(rows[1]["A_plan"] - 0.1 * 3 * 2.5) < 1e-12 and abs(rows[1]["A_pre"] - 0.3) < 1e-12
    # a group without any spread (non-frozen) is skipped
    same = [{"stop": 0.5, "act": 0.5, "len": 0.5, "fmt": 1.0}] * 2
    assert V18.advantages_t1(same, {"stop": 1, "act": 1, "len": 1, "fmt": 0.5}, set(), 3, w, 1.0)[0] is None
    sp_len = [{"stop": 0.5, "act": 0.5, "len": 0.1, "fmt": 1.0}, {"stop": 0.5, "act": 0.5, "len": 0.9, "fmt": 1.0}]
    assert V18.advantages_t1(sp_len, {"stop": 1, "act": 1, "len": 1, "fmt": 0.5}, {"len"}, 3, w, 1.0)[0] is None


def test_kappa_tau_and_masks():
    # tau = sum |a_tok| / tokens (zero tokens in the denominator), kappa = target / tau(1)
    s1 = {"source": "task1", "gen_ids": [1, 2, 3, 4], "adv_plan": 0.5, "plan_mask": [1, 1, 0, 1],
          "stop_mask": [0, 0, 1, 0], "prefix_mask": [1, 1, 0, 0], "adv_prefix": 0.2, "note_mask": [0, 1, 0, 0]}
    a = RA.token_advantages(s1, 4)
    assert a == [0.7, 0.0, 0.0, 0.5]                    # note -> 0, the value token -> 0, prefix + plan add up
    s2 = {"source": "task2", "gen_ids": [1, 2], "adv": -1.0, "stop_mask": [0, 1], "adv_stop": 0.0}
    assert RA.token_advantages(s2, 2) == [-1.0, -1.0]   # Task 2: the value token included
    tau = RA.token_weighted_tau([s1, s2])
    assert abs(tau["task1"]["tau"] - 1.2 / 4) < 1e-12 and tau["task2"]["tau"] == 1.0
    assert RA.calibrate_kappa(0.3, 1.0, 0.0197, 0.0762) == (0.0197 / 0.3, 0.0762)
    with pytest.raises(AssertionError):
        RA.calibrate_kappa(0.0, 1.0, 0.0197, 0.0762)
    with pytest.raises(AssertionError):                 # a plan mask over a value token
        RA.token_advantages({"gen_ids": [1, 2], "adv_plan": 1.0, "plan_mask": [1, 1], "value_mask": [0, 1]}, 2)
    assert RA.plan_mask_of(3, [0, 1, 0]) == [1, 0, 1] and RA.plan_mask_of(2) == [1, 1]
    # the masks of the three sample states
    g = {"gen_ids": [5, 6, 7], "stop_mask": [0, 1, 0], "value_mask": None}
    assert V18.task1_sample_masks({"status": "valid", "planner_gen": g}, 3) == ("prefix+plan", [1, 0, 0], [1, 0, 1])
    g1 = {"gen_ids": [5, 6, 7], "stop_mask": None, "value_mask": [0, 1, 0]}
    assert V18.task1_sample_masks({"status": "valid", "planner_gen": g1}, 1) == ("plan_t1", None, [1, 0, 1])
    assert V18.task1_sample_masks({"status": "invalid", "planner_gen": g1}, 1) == ("all", None, [1, 1, 1])


def test_batch_fingerprint():
    t2 = [{"update": 1, "slot": 0, "replicate": 1, "policy_sha": "p", "time": 2.5},
          {"update": 1, "slot": 0, "replicate": 0, "policy_sha": "p", "time": 1.5}]
    t1 = [{"update": 1, "conversation_id": "c", "t": 1, "policy_sha": "p", "time": 3.0}]
    a = RA.batch_fingerprint(t2, t1)
    assert a == RA.batch_fingerprint(list(reversed(t2)), t1)                  # order-free
    assert a != RA.batch_fingerprint(t2, [dict(t1[0], time=3.5)])              # another row identity


# ================================================================== the trainer end to end (dry run)
def test_v18_run_and_verify_clean():
    d, sp, out = fresh()
    rep = report(out, sp)
    assert not fails(rep), fails(rep)
    for k in ("rl.v18_reward", "rl.v18_advantage", "rl.v18_tau", "rl.v18_scales", "rl.v18_task1", "rl.v18_selection",
              "rl.v18_validation", "rl.v18_init", "rl.v18_task2"):
        assert rep.checks[k]["n"] > 0, k
    sc = json.load(open(os.path.join(out, "adv_scales.json")))
    u1 = T.read_jsonl(os.path.join(out, "updates.jsonl"))[0]
    assert abs(u1["tau"]["task1"]["tau"] - 0.0197) < 1e-9 and abs(u1["tau"]["task2"]["tau"] - 0.0762) < 1e-9
    assert u1["kappa"] == sc["kappa"] and u1["learner_stats"]["optimizer_steps"] == 8
    rows = T.read_jsonl(os.path.join(out, "rollouts_task1.jsonl"))
    assert any(r["t"] == 1 for r in rows)                                     # turn 1 groups exist
    assert all(x["p_end"] is None for r in rows if r["t"] == 1 for x in r["samples"])
    assert not any(p[1] == 1 for p in u1["task1_stats"]["aux_points"])         # no turn-1 supervision
    fin = json.load(open(os.path.join(out, "final.json")))
    assert fin["validated"] and fin["selection"]["J"]["0"] == 0.0
    # no SFT, ref = u0 = the init adapter copy
    assert not os.path.exists(os.path.join(out, "sft.jsonl"))
    st0 = json.load(open(os.path.join(out, "ckpt", "u00000", "state.json")))
    assert st0["init"]["copied"] and st0["policy_sha"] == st0["init"]["policy_sha"]
    meta = T.read_jsonl(os.path.join(out, "run_meta.jsonl"))
    assert [m["kind"] for m in meta][:2] == ["start", "init"] and meta[0]["spec_version"] == "v18"


def test_v18_task1_points_are_1_to_n_and_one_message_conversation(monkeypatch):
    orig = T.FakeEnv.human_turns
    monkeypatch.setattr(T.FakeEnv, "human_turns", lambda self, c: 1 if c == "c00" else orig(self, c))
    d = tempfile.mkdtemp()
    sp = make_splits(d)
    out = os.path.join(d, "run")
    T.main(F.v18_args(d, sp, out, "--task1-convs", "15", updates=1))
    rows = T.read_jsonl(os.path.join(out, "rollouts_task1.jsonl"))
    assert [r["t"] for r in rows if r["conversation_id"] == "c00"] == [1]      # n = 1 still has its t = 1 group
    env = T.FakeEnv(None)
    for cid in {r["conversation_id"] for r in rows}:
        n = 1 if cid == "c00" else env.human_turns(cid)
        assert sorted(r["t"] for r in rows if r["conversation_id"] == cid) == list(range(1, n + 1))
    assert not fails(report(out, sp))


def test_scales_refused_when_stop_frozen(monkeypatch):
    d = tempfile.mkdtemp()
    sp = make_splits(d)
    out = os.path.join(d, "run")
    # every plan the same P_end -> r_stop has no spread anywhere -> frozen -> training refused (§4.1 item 4)
    monkeypatch.setattr(T.FakeLearner, "end_prob", lambda self, x: 0.4)
    with pytest.raises(SystemExit) as e:
        T.main(F.v18_args(d, sp, out, updates=1))
    assert "stop" in str(e.value) and os.path.exists(os.path.join(out, "adv_scales.refused.json"))
    assert not os.path.exists(os.path.join(out, "adv_scales.json"))
    assert not os.path.exists(os.path.join(out, "ckpt", "u00001"))


def test_scales_r_len_freeze_allowed(monkeypatch):
    # r_len without spread anywhere -> frozen; the run goes on (§4.1: only r_len may freeze), its z is 0
    monkeypatch.setattr(RR, "r_len_of", lambda dist, label, hw, band=2.0 / 3.0: (1.0, None))
    d, sp, out = fresh(updates=1)
    sc = json.load(open(os.path.join(out, "adv_scales.json")))
    assert sc["frozen"] == ["len"] and sc["spread_groups"]["len"] == 0
    u1 = T.read_jsonl(os.path.join(out, "updates.jsonl"))[0]
    for g in u1["task1_stats"]["advantages"]:
        for rec in (g[3] if g[2] == "used" else []):
            assert rec[4]["len"] == 0.0
    assert not fails(report(out, sp))
    assert "len" not in V18.MUST_NOT_FREEZE and set(V18.MUST_NOT_FREEZE) == {"stop", "act", "turn"}


def test_kappa2_low_confidence_flag():
    groups = [[{"turn": 0.5}, {"turn": 0.9}], [{"turn": 0.1}, {"turn": 0.3}], [{"turn": 0.2}, {"turn": 0.2}]]
    m = RA.measure_scales(groups, ["turn"], {"turn": 2}, 0.01)
    assert m["turn"]["n_spread_groups"] == 2 and not m["turn"]["frozen"]
    assert T.V18_KAPPA_LOW_CONF_GROUPS == 2


def test_resume_reads_same_scales_and_refuses_changed_batch(monkeypatch):
    d, sp, out = fresh(updates=1)
    sc = json.load(open(os.path.join(out, "adv_scales.json")))
    tr = T.Trainer(T.parse_args(F.v18_args(d, sp, out, "--resume", updates=2)))
    tr.build()
    tr.load_checkpoint()
    assert tr.scales_v18(2, [], [], [], []) == sc                              # u1 exists: read back, not re-measured
    bad = dict(sc, batch_sha256="0" * 64)
    T.write_json_atomic(os.path.join(out, "adv_scales.json"), bad)
    with pytest.raises(SystemExit):
        tr.scales_v18(2, [], [], [], [])


def test_abort_in_update1_renames_and_remeasures(monkeypatch):
    d = tempfile.mkdtemp()
    sp = make_splits(d)
    out = os.path.join(d, "run")
    calls = {"n": 0}
    orig = T.FakeLearner.update

    def aborting(self, samples, cfg, seed, aux=None, aux_orders=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RA.MismatchAbort(0.5)
        return orig(self, samples, cfg, seed, aux=aux, aux_orders=aux_orders)
    monkeypatch.setattr(T.FakeLearner, "update", aborting)
    # the dry-run trainer passes no mismatch_abort; the fake raises it itself on the first call
    with pytest.raises(SystemExit):
        T.main(F.v18_args(d, sp, out, updates=1))
    first = json.load(open(os.path.join(out, "adv_scales.json")))
    assert os.path.exists(os.path.join(out, "ABORTED_u00001.json"))
    T.main(F.v18_args(d, sp, out, "--resume", updates=1))
    now = json.load(open(os.path.join(out, "adv_scales.json")))
    old = os.path.join(out, "adv_scales.aborted_%s.json" % first["batch_sha256"][:12])
    assert os.path.exists(old) and now["batch_sha256"] != first["batch_sha256"] and now["renamed"]["reason"] == "aborted"
    rep = report(out, sp)
    assert not fails(rep), fails(rep)


def test_crash_resume_recomputes_selection_from_validation(monkeypatch):
    d = tempfile.mkdtemp()
    sp = make_splits(d)
    out = os.path.join(d, "run")
    orig = T.Trainer.finalize_v18
    monkeypatch.setattr(T.Trainer, "finalize_v18", lambda self, u, r: (_ for _ in ()).throw(SystemExit("crash")))
    with pytest.raises(SystemExit):
        T.main(F.v18_args(d, sp, out, updates=2))
    assert not os.path.exists(os.path.join(out, "final.json"))
    monkeypatch.setattr(T.Trainer, "finalize_v18", orig)
    T.main(F.v18_args(d, sp, out, "--resume", updates=2))
    fin = json.load(open(os.path.join(out, "final.json")))
    assert fin["last_update"] == 2 and fin["validated"]
    assert len(T.read_jsonl(os.path.join(out, "updates.jsonl"))) == 2            # no update re-run
    assert not fails(report(out, sp))


# ================================================================== §7.3 selection and stopping
def M(stop, act, len_, nll=0.5, ent=1.0, inv=0):
    return {"stop_score": stop, "act_score": act, "len_score": len_, "nll": nll, "act_entropy": ent, "n_invalid": inv}


W = {"stop": 1.0, "act": 1.0, "len": 0.5}


def test_selection_rule():
    m = {0: M(0.8, 0.7, 0.6), 1: M(0.82, 0.71, 0.6), 2: M(0.85, 0.72, 0.62), 3: M(0.9, 0.6, 0.6)}
    s = V18.selection(m, {1: 0.0, 2: 0.0, 3: 0.0}, 1.0, 3, W)
    assert s["candidates"] == [1, 2] and s["order"] == [2, 1]                 # u3 fails the act guard (-0.1 < -0.02)
    assert abs(s["J"]["2"] - (0.05 + 0.02 + 0.5 * 0.02)) < 1e-12
    m[2] = M(0.85, 0.72, 0.62, nll=0.56)                                     # nll guard: > nll0 + 0.05
    assert V18.selection(m, {1: 0.0, 2: 0.0, 3: 0.0}, 1.0, 3, W)["candidates"] == [1]
    m[2] = M(0.85, 0.72, 0.62, ent=0.49)                                     # entropy collapse
    assert 2 not in V18.selection(m, {1: 0.0, 2: 0.0, 3: 0.0}, 1.0, 3, W)["candidates"]
    m[2] = M(0.85, 0.72, 0.62, inv=2)                                        # n_invalid > u0 + 1
    assert 2 not in V18.selection(m, {1: 0.0, 2: 0.0, 3: 0.0}, 1.0, 3, W)["candidates"]
    m[2] = M(0.85, 0.72, 0.62)
    # audit B NIT 1: u is excluded only when the drift STOP rule triggered at u (u-1 and u both below -margin)
    assert 2 in V18.selection(m, {1: 0.0, 2: -1.5, 3: 0.0}, 1.0, 3, W)["candidates"]
    assert 2 not in V18.selection(m, {1: -1.2, 2: -1.5, 3: 0.0}, 1.0, 3, W)["candidates"]
    assert V18.drift_stop_at({1: -1.2, 2: -1.5}, 2, 1.0) and not V18.drift_stop_at({1: -1.2}, 1, 1.0)
    # equal J -> the later update first
    m = {0: M(0.8, 0.7, 0.6), 1: M(0.9, 0.7, 0.6), 2: M(0.9, 0.7, 0.6)}
    assert V18.selection(m, {1: 0, 2: 0}, 1.0, 2, W)["order"] == [2, 1]


def test_stop_rule():
    assert V18.stop_rule({1: 0, 2: 0}, {0: 0.0, 1: -0.1, 2: -0.2}, 1.0, 5, 2) == (2, "val_decline")
    assert V18.stop_rule({1: 0, 2: 0}, {0: 0.0, 1: 0.1, 2: 0.05}, 1.0, 5, 2) == (None, None)
    assert V18.stop_rule({1: -1.5, 2: -2.0}, {0: 0.0, 1: 0.1, 2: 0.2}, 1.0, 5, 2) == (2, "length_drift")
    assert V18.stop_rule({1: 0, 2: 0, 3: 0}, {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3}, 1.0, 3, 3) == (3, "max_updates")
    assert V18.stop_rule({1: 0, 2: 0, 3: 0}, {0: 0.0, 1: 0.3, 2: 0.2, 3: 0.1}, 1.0, 5, 3) == (3, "val_decline")


def test_final_falls_back_to_u0_when_task2_check_fails(monkeypatch):
    orig = T.Trainer.validate_v18

    def short(self, u, task2):
        s = orig(self, u, task2)
        if task2 and u > 0:
            s = dict(s, task2_drift=-500.0)          # below -margin (the dry runs use margin 100)
        return s
    monkeypatch.setattr(T.Trainer, "validate_v18", short)
    d, sp, out = fresh(updates=2)
    fin = json.load(open(os.path.join(out, "final.json")))
    assert fin["final_update"] == 0 and not fin["grpo_adopted"]
    assert len(fin["selection"]["task2_check"]) == min(2, len(fin["selection"]["order"]))


# ================================================================== §6 selector + reranker
class StubHL:
    sha256 = "stub"

    def __init__(self, scores):
        self.s = scores

    def scores(self, texts):
        return [self.s.get(t) if (t or "").strip() else None for t in texts]


def test_borda_top_k_and_rerank_choice():
    info = {"candidates": [0, 1, 2, 3], "score": [1, 1, 2, 3], "len_rank": [0, 1, 0, 1]}
    assert SS.borda_order(info) == [0, 1, 2, 3] and SS.borda_top_k(info, 2) == [0, 1]
    info2 = {"candidates": [0, 1, 2, 3], "score": [0, 2, 2, 3], "len_rank": [0, 1, 0, 1]}
    assert SS.borda_top_k(info2, 2) == [0, 2, 1]                     # every candidate tied with the 2nd included
    ch, _ = SS.rerank_choice(info2, {0: 0.1, 1: 0.9, 2: 0.9}, 2)
    assert ch == 1                                                   # tie on s and Borda score -> the candidate order
    info3 = {"candidates": [0, 1, 2], "score": [0, 1, 1], "len_rank": [0, 0, 1]}
    ch, top = SS.rerank_choice(info3, {0: 0.1, 1: 0.9, 2: 0.9}, 3)
    assert top == [0, 1, 2] and ch == 1
    ch, _ = SS.rerank_choice(info, {0: 0.1, 1: None}, 2)
    assert ch == 0                                                   # a blank (unscored) candidate ranks last


def test_select_rerank_fields_and_fallback():
    cands = ["aa bb", "aa bb cc dd", "zz", ""]
    idx, info = SS.select_rerank(cands, [0, 1, 2], 2, [], None, StubHL({"aa bb": 0.0, "aa bb cc dd": 1.0, "zz": 5.0}), 2)
    assert info["borda_index"] == 0 and info["topk"] == [0, 2] and idx == 2 and info["rerank_changed"]
    assert len(info["rerank_scores"]) == len(info["candidates"]) and info["rerank_scores"][1] is None
    ok, good, why = VV.selection_ok({"selection": info, "selected_index": idx})
    assert ok and good, why
    # no guard survivor (eligible None): the non-empty candidates
    idx, info = SS.select_rerank(cands, None, 2, [], None, StubHL({"aa bb": 0.0, "aa bb cc dd": 1.0, "zz": 5.0}), 2)
    assert 3 not in info["candidates"]
    # a tampered selection is caught
    bad = {"selection": dict(info), "selected_index": (idx + 1) % 3}
    assert not VV.selection_ok(bad)[1]


def test_human_likeness_scorer_features(tmp_path):
    import numpy as np
    p = F.make_reranker(str(tmp_path / "r.json"))

    class Emb:
        def embed(self, texts):
            return np.asarray([[len(t), t.count("a"), 1.0] for t in texts], dtype=float)
    hl = SS.HumanLikenessScorer(p, embedder=Emb())
    s = hl.scores(["abc", "", "Hello there!"])
    X = hl.features(["abc"])
    assert s[1] is None and abs(s[0] - float(X[0] @ np.asarray(hl.w))) < 1e-12
    assert SS.style_features("hello? Yes!") == [1.0, 1.0, 1.0, 1.0, 1 / 8]
    d = json.load(open(p))
    d["features"]["style"] = ["x"]
    json.dump(d, open(p, "w"))
    with pytest.raises(ValueError):
        SS.HumanLikenessScorer(p, embedder=Emb())


# ================================================================== §6.1-6.3 train_reranker (dry run)
def _corpus(tmp, ids_n):
    recs = []
    for cid, n in ids_n.items():
        msgs = []
        for t in range(1, n + 1):
            msgs.append({"participant_name": "User", "text": ("i need data on %s please, maybe %d" % (cid, t)).lower()})
            msgs.append({"participant_name": "Agent", "text": "Here is something %d." % t})
        recs.append({"conversation_id": cid, "record_id": "r" + cid, "chat_messages": msgs, "scenario": {}})
    p = os.path.join(tmp, "corpus.jsonl")
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    return p


def _rerank_world(tmp):
    tr = ["c%02d" % i for i in range(5)]
    va = ["v%02d" % i for i in range(2)]
    sp = os.path.join(tmp, "splits.json")
    json.dump({"folds": [{"fold": 0, "train": tr, "train_all": tr, "validation": va, "validation_all": va,
                          "test": ["x0"], "test_all": ["x0"], "forbidden_for_training": va + ["x0"]}]}, open(sp, "w"))
    corpus = _corpus(tmp, {**{c: 3 for c in tr}, **{c: 2 for c in va}, "x0": 2})
    init, sha = F.make_init_adapter(os.path.join(tmp, "init"))
    return sp, corpus, init, sha


def fake_embed(texts):
    import numpy as np
    out = []
    for t in texts:
        v = [((hash_(t, k) % 1000) / 1000.0) for k in range(12)]
        v[0] = 1.0 if t[:1].islower() else 0.0            # a signal the reranker can learn: people write lowercase
        out.append(v)
    return np.asarray(out)


def hash_(t, k):
    import hashlib
    return int(hashlib.sha256(("%d|%s" % (k, t)).encode()).hexdigest()[:8], 16)


def test_train_reranker_generate_fit_gate(tmp_path):
    import train_reranker as R
    sp, corpus, init, sha = _rerank_world(str(tmp_path))
    ct, cv, out = str(tmp_path / "ct.jsonl"), str(tmp_path / "cv.jsonl"), str(tmp_path / "rr.json")
    R.main(["generate", "--dry-run", "--fold", "0", "--split", "train_all", "--seeds", "0", "1", "--init-adapter", init,
            "--init-policy-sha", sha, "--splits", sp, "--corpus", corpus, "--out", ct])
    R.main(["generate", "--dry-run", "--fold", "0", "--split", "validation_all", "--seeds", "0", "--init-adapter", init,
            "--init-policy-sha", sha, "--splits", sp, "--corpus", corpus, "--out", cv])
    with pytest.raises(SystemExit):                                   # spec seeds
        R.parse(["generate", "--dry-run", "--fold", "0", "--split", "train_all", "--seeds", "0", "--init-adapter", init,
                 "--out", "x"])
    rows = [json.loads(l) for l in open(ct)]
    pairs, st = R.build_pairs(rows, {"c%02d" % i for i in range(5)}, set())
    # slot 0 is the same text under both seeds: kept once (dedupe); guard-rejected / blank removed
    assert st["n_duplicate"] > 0 and st["n_pairs"] == len(pairs) and st["n_conversations"] == 5
    for (cid, t) in {(p[0], p[1]) for p in pairs}:
        cs = [p[3] for p in pairs if p[0] == cid and p[1] == t]
        assert len(cs) == len({R.norm_text(c) for c in cs})
    with pytest.raises(SystemExit):                                   # a row outside the allowed split
        R.build_pairs(rows, {"c00"}, set())
    res = R.main(["fit", "--fold", "0", "--cands-train", ct, "--cands-val", cv, "--splits", sp, "--out", out,
                  "--n-boot", "200"], embed=fake_embed)
    j = json.load(open(out))
    assert j["loco"]["chosen_C"] in R.CS and len(j["loco"]["folds"]) == 5
    for fo in j["loco"]["folds"]:                                     # per-fold PCA, never on the held-out conversation
        assert fo["held_out"] not in fo["pca_fit_cids"] and fo["held_out"] not in fo["train_cids"]
    g = j["gate"]
    assert g["passed"] == (g["loco_acc"] >= 0.60 and g["val_acc"] >= 0.55)
    assert g["loco_ci"]["n_clusters"] == 5 and g["val_ci"]["n_clusters"] == 2 and g["loco_ci"]["lo"] <= g["loco_ci"]["hi"]
    assert j["train_cids"] == sorted({p[0] for p in pairs}) and "platt" in j and res["n_pairs"] == len(pairs)
    assert not VV.opened_bench(j["opened_files"])
    # the runtime scorer reads it
    import numpy as np

    class Emb:
        def embed(self, texts):
            return fake_embed(texts)
    hl = SS.HumanLikenessScorer(out, embedder=Emb())
    assert len(hl.scores(["i want data", "Option 3 for turn 1 ."])) == 2


def test_cluster_ci_and_acc():
    import train_reranker as R
    assert R.acc_of([1.0, -1.0, 0.0]) == 0.5
    ci = R.cluster_ci({"a": [1, 1], "b": [0, 0]}, n_boot=500)
    assert 0.0 <= ci["lo"] <= 0.5 <= ci["hi"] <= 1.0


# ================================================================== §1.1 label_acts (fake chat)
def _label_world(tmp):
    sp = os.path.join(tmp, "splits.json")
    json.dump({"folds": [{"fold": 0, "train": ["c0", "c1"], "train_all": ["c0", "c1"], "validation": ["v0"],
                          "validation_all": ["v0"], "test": ["x0"], "test_all": ["x0"],
                          "forbidden_for_training": ["v0", "x0"]}]}, open(sp, "w"))
    corpus = _corpus(tmp, {"c0": 3, "c1": 2, "v0": 2, "x0": 2})
    cb = os.path.join(tmp, "codebook.txt")
    open(cb, "w").write("This is an unrelated codebook text about QUERY REFINE INSPECT CHALLENGE DECIDE CLOSE labels.")
    return sp, corpus, cb


def test_label_acts_votes_retries_and_cache(tmp_path):
    import label_acts as L
    sp, corpus, cb = _label_world(str(tmp_path))
    seen = []

    def chat(system, user, seed):
        seen.append(seed)
        if seed == 1:
            return "", "length"                                       # empty -> retried with seed 101
        if seed % 100 == 2:
            return "not json at all", "stop"                          # unparseable, and every retry fails
        return '{"move": "Inquire", "act": "ask_more"}', "stop"
    out = str(tmp_path / "lab.jsonl")
    argv = ["--fold", "0", "--split", "train_all", "--splits", sp, "--corpus", corpus, "--codebook", cb, "--out", out]
    with pytest.raises(SystemExit) as e:                              # 1 of 3 votes missing everywhere -> 33% > 5%
        L.main(argv, chat=chat)
    assert "failure rate" in str(e.value) and not os.path.exists(out) and os.path.exists(out + ".failed.json")

    def chat2(system, user, seed):
        if seed == 1:
            return "", "length"
        return '{"move": "Inquire", "act": "ask_more"}', "stop"
    out = str(tmp_path / "lab2.jsonl")
    argv = argv[:-1] + [out]
    m = L.main(argv, chat=chat2)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 5 and m["n_votes_missing"] == 0 and m["n_retries"] == 5                # one retry per message
    assert all(r["n_valid_votes"] == 3 and r["q7"] == {"Inquire": 1.0} and r["majority"] == "Inquire" for r in rows)
    assert all(v["seed"] in (0, 101, 2) for r in rows for v in r["votes"])
    assert m["ngram8_overlap"] == 0 and m["ngram8_checked"] and os.path.exists(out + ".review.md")
    assert m["complete_agreement"] == 1.0
    assert not VV.opened_bench(m["opened_files"], allow=(cb,))
    # cache: the same key is reused (no call), a different key is refused
    assert L.main(argv, chat=lambda *a: (_ for _ in ()).throw(AssertionError("called")))["labels_sha256"] == m["labels_sha256"]
    with pytest.raises(SystemExit):
        L.main(argv[:-2] + ["--model", "other", "--out", out], chat=chat2)
    # the labels load as train labels (cids in train_all)
    labels, meta, sha = V18.load_labels(out)
    assert not V18.check_label_cids(labels, {"c0", "c1"}, {"v0", "x0"})
    assert V18.check_label_cids(labels, {"c0"}, set()) == ["c1"]


def test_label_acts_insufficient_votes_and_ngram_refusal(tmp_path):
    import label_acts as L
    sp, corpus, cb = _label_world(str(tmp_path))
    acts = L.load_acts_module()
    sysp = L.build_system_prompt(acts)
    assert acts.coarse_block() in sysp and acts.fine_block() in sysp
    open(cb, "w").write("blah " + " ".join(sysp.split()[10:30]) + " blah")      # shares 8-grams with our prompt
    with pytest.raises(SystemExit) as e:
        L.main(["--fold", "0", "--split", "train_all", "--splits", sp, "--corpus", corpus, "--codebook", cb,
                "--out", str(tmp_path / "x.jsonl")], chat=lambda s, u, sd: ('{"move": "Note", "act": "confirm"}', "stop"))
    assert "8-grams" in str(e.value)
    with pytest.raises(SystemExit):                                   # the test side is never labelled
        L.parse(["--fold", "0", "--split", "test_all", "--out", "x"])
    assert L.parse_answer('{"move": "Inquire", "act": "zzz"}', acts) == ("Inquire", "unknown")   # the move is kept
    assert L.parse_answer("", acts) is None and L.parse_answer('{"x": 1}', acts) is None
    q7, n = L.soft_label([{"move": "Note"}, {"move": None}, {"move": "Note"}])
    assert q7 == {"Note": 1.0} and n == 2
    assert L.fleiss_kappa([[3, 0], [0, 3]], ["a", "b"]) == 1.0
    # a message with < 2 valid votes: no move information (r_act / r_len not applicable)
    assert RR.r_act_of(D(("Note", "confirm", 1.0, 3)), {"q7": {"Note": 1.0}, "n_valid_votes": 1}, 1, 3)[1] == "votes"


# ================================================================== turn 1 in task2_env (subprocess, the E1.6 tree)
T1_CODE = r'''
import json, os, sys, types
sys.path[:0] = [%r, %r]
os.environ.update({"SEPSIM_ACT_PRIOR": "nostopclobber", "SEPSIM_ACT_FULL": "1"})
import task2_env as TE
class Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)
class Planner:
    tok = Tok()
    def stop_mask(self, gen_ids):
        return TE.PlannerLM.stop_mask(self, gen_ids)
    def field_mask(self, gen_ids, field):
        return None
    def stop_target(self, gen_ids, mask, want):
        return TE.PlannerLM.stop_target(self, gen_ids, mask, want)
def plan(es):
    d = {"critique": "", "end_session": es, "act_distribution": [
        {"move": "Inquire", "act": "ask_more", "p": 0.7, "length_words": 12},
        {"move": "Complete", "act": "settle", "p": 0.3, "length_words": 3}], "length_words": 12}
    if es is None:
        d.pop("end_session")
    return json.dumps(d)
out = {}
for name, es in (("true", True), ("false", False), ("missing", None), ("bad", "maybe")):
    raw = plan(es)
    gen = {"raw": raw, "gen_ids": [ord(c) for c in raw], "prompt_ids": [1], "hit_max_new": False, "fit": {}}
    env = types.SimpleNamespace(arm="pend", recs={"c": {"scenario": {"persona": {}, "goal": {}}}}, system="s",
                                planner=Planner(), _plan_many=lambda items, gen=gen: [dict(gen) for _ in items])
    xs = TE.Task2Env.task1_sample(env, "c", 1, "u", False, 2, 1.0, 1.0, 0, t1=True)
    out[name] = [{k: x.get(k) for k in ("valid_t1", "value_mask_ok", "decision_valid")} | {
        "vm": x["planner_gen"].get("value_mask") is not None, "sm": x["planner_gen"]["stop_mask"]} for x in xs]
try:
    TE.Task2Env.task1_sample(env, "c", 1, "u", False, 1, 1.0, 1.0, 0)
    out["v17_raises"] = False
except ValueError:
    out["v17_raises"] = True
print("OUT " + json.dumps(out))
'''


def test_task1_sample_turn1_valid_t1_and_value_mask():
    code = T1_CODE % (E1R, HERE)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    line = [l for l in r.stdout.splitlines() if l.startswith("OUT ")]
    assert line, r.stdout + r.stderr
    out = json.loads(line[-1][4:])
    assert out["v17_raises"] is True                                       # without t1 turn 1 still raises (v17)
    for name in ("true", "false"):                                         # the ignored turn-1 value still located
        assert all(x["valid_t1"] and x["value_mask_ok"] and x["vm"] and x["sm"] is None
                   for x in out[name]), out
    for name in ("missing", "bad"):                                        # end_session missing / invalid -> invalid
        assert all(x["valid_t1"] is False and not x["vm"] for x in out[name]), out


def test_t1_value_mask_decode_check():
    import task2_env as TE

    class Tok:
        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(i) for i in ids)

    class P:
        tok = Tok()

        def stop_mask(self, gen_ids):
            return TE.PlannerLM.stop_mask(self, gen_ids)
    raw = '{"end_session": true, "x": 1}'
    g = {"gen_ids": [ord(c) for c in raw]}
    d = {"end_session_raw": True}
    assert TE.t1_value_mask(P(), g, d) is not None
    d = {"end_session_raw": "false"}                                        # the located value is not the parsed one
    assert TE.t1_value_mask(P(), g, d) is None and d["t1_value_mismatch"]
    assert V18.t1_status({"valid_t1": True, "value_mask_ok": False}, 1) == ("dropped", "t1_value_mismatch")
    assert V18.t1_status({"valid_t1": False}, 1) == ("invalid", None)
    assert V18.t1_status({"decision_valid": True, "mask_ok": False}, 3) == ("dropped", "stop_mask_mismatch")


# ================================================================== §3.2 clean_v18 in training / validation / verify
def test_clean_v18_groups_and_validation(monkeypatch):
    orig = T.FakeEnv.run_episode

    def run(self, cid, seed, replicate=0, **kw):
        e = orig(self, cid, seed, replicate=replicate, **kw)
        if replicate == 1:
            e["episode_counters"] = {"judge_error": 1}                     # a lost verdict: still clean_v18
            e["clean"], e["coverage_diag"] = False, None
        if replicate == 2:
            e["episode_counters"] = {"r0_len_truncated": 1}
            e["clean"] = e["clean_v18"] = False
        return e
    monkeypatch.setattr(T.FakeEnv, "run_episode", run)
    d, sp, out = fresh(updates=1)
    u1 = T.read_jsonl(os.path.join(out, "updates.jsonl"))[0]
    excl = u1["task2_stats"]["excluded"]
    assert excl and all(e[1] == 2 and e[2] == ["r0_len_truncated"] for e in excl)
    assert u1["task2_stats"]["coverage_diag_missing"] > 0
    used = [r[0] for a_ in u1["task2_stats"]["advantages"] if a_[1] == "used" for r in a_[2]]
    assert 1 in used and 2 not in used                                     # the judge failure stays in its group
    vrows = T.read_jsonl(os.path.join(out, "validation.jsonl"))
    # validation seed episodes are replicate 0 -> clean; the summary counts clean_v18 episodes
    s0 = [v for v in vrows if v.get("kind") == "summary" and v["update"] == 0][0]
    assert s0["n_episodes"] == len([r for r in vrows if r.get("kind") == "episode" and r["update"] == 0])
    rep = report(out, sp)
    assert not fails(rep), fails(rep)


def test_verify_clean_flags_and_judge_error_regression():
    rep = V.Report()
    r = ep(3, 5, counters={"judge_error": 1})
    r.update(clean=True, clean_v18=True, conversation_id="c", seed=0)
    V.check_pend([r], rep, v18=True)
    # v17 expected value now includes judge_error == 0 (the env's own rule): clean True with judge_error is a FAIL
    assert rep.checks["pend.clean_flag"]["fail"] == 1 and rep.checks["pend.clean_v18_flag"]["fail"] == 0
    rep = V.Report()
    r.update(clean=False, clean_v18=False)
    V.check_pend([r], rep, v18=True)
    assert rep.checks["pend.clean_flag"]["fail"] == 0 and rep.checks["pend.clean_v18_flag"]["fail"] == 1


# ================================================================== the verifier catches tampering
@pytest.mark.parametrize("what", ["label_cid", "r_act", "advantage", "tau", "scales", "selection", "final", "t1_aux",
                                  "status", "clean_v18", "rerank_gate"])
def test_verify_v18_catches(what):
    d, sp, out = fresh(updates=2)
    assert not fails(report(out, sp))
    p1, pu = os.path.join(out, "rollouts_task1.jsonl"), os.path.join(out, "updates.jsonl")
    rows, ups = T.read_jsonl(p1), T.read_jsonl(pu)
    if what == "label_cid":
        args = T.read_jsonl(os.path.join(out, "run_meta.jsonl"))[0]["config"]["args"]
        lab = T.read_jsonl(args["act_labels"])
        lab.append(dict(lab[0], conversation_id="c14", t=1))               # a validation id in the reward labels
        rewrite(args["act_labels"], lab)
        name = "rl.v18_labels"
    elif what == "r_act":
        for r in rows:
            for x in r["samples"]:
                if x.get("components") and x["components"].get("act") is not None:
                    x["components"]["act"] += 0.1
                    break
            else:
                continue
            break
        rewrite(p1, rows)
        name = "rl.v18_reward"
    elif what == "advantage":
        for g in ups[0]["task1_stats"]["advantages"]:
            if g[2] == "used":
                g[3][0][6] += 0.01
                break
        rewrite(pu, ups)
        name = "rl.v18_advantage"
    elif what == "tau":
        ups[1]["tau"]["task1"]["tau"] *= 1.5
        rewrite(pu, ups)
        name = "rl.v18_tau"
    elif what == "scales":
        sp_ = os.path.join(out, "adv_scales.json")
        sc = json.load(open(sp_))
        sc["scales"]["act"] *= 2
        json.dump(sc, open(sp_, "w"))
        name = "rl.v18_scales"
    elif what == "selection":
        fp = os.path.join(out, "final.json")
        fin = json.load(open(fp))
        fin["selection"]["order"] = list(reversed(fin["selection"]["order"]))
        if fin["selection"]["order"] == json.load(open(fp))["selection"]["order"]:
            fin["selection"]["J"]["1"] = 9.0
        json.dump(fin, open(fp, "w"))
        name = "rl.v18_selection"
    elif what == "final":
        fp = os.path.join(out, "final.json")
        fin = json.load(open(fp))
        fin["final_update"] = 1 if fin["final_update"] != 1 else 2
        json.dump(fin, open(fp, "w"))
        name = "rl.v18_selection"
    elif what == "t1_aux":
        ups[0]["task1_stats"]["aux_points"].append([ups[0]["task1_stats"]["convs"][0], 1])
        rewrite(pu, ups)
        name = "rl.v18_task1"
    elif what == "status":
        for r in rows:
            if r["t"] == 1:
                for x in r["samples"]:
                    if x["status"] == "valid":
                        x["status"] = "invalid"
                        break
                break
        rewrite(p1, rows)
        name = "rl.v18_task1"
    elif what == "clean_v18":
        p2 = os.path.join(out, "rollouts.jsonl")
        r2 = T.read_jsonl(p2)
        r2[0]["episode"]["episode_counters"] = {"r0_empty": 1}
        rewrite(p2, r2)
        name = "rl.v18_task2"
    else:
        args = T.read_jsonl(os.path.join(out, "run_meta.jsonl"))[0]["config"]["args"]
        F.make_reranker(args["reranker"], passed=False)                    # the gate result changed after the run
        name = "rl.v18_reranker"
    rep = report(out, sp)
    assert rep.checks[name]["fail"] >= 1, (what, rep.checks.get(name))


def test_verify_code_has_no_bench_paths():
    rep = V.Report()
    VV.check_code_no_bench(rep, HERE)
    assert rep.checks["rl.v18_labels"]["fail"] == 0 and rep.checks["rl.v18_labels"]["n"] == len(VV.CODE_NO_BENCH)
    assert VV.bench_paths_in("x = 'data/req_shards_v1.json'") and not VV.bench_paths_in("data/x.json")


# ================================================================== SPEC gating, v17 pin
def test_spec_gate_v18_and_v17_unchanged(tmp_path):
    d = str(tmp_path)
    sp = make_splits(d)
    base = F.v18_args(d, sp, os.path.join(d, "o"))
    no_abl = [x for i, x in enumerate(base) if x != "--ablation" and (i == 0 or base[i - 1] != "--ablation")]
    with pytest.raises(SystemExit):                       # the dry run's own --G / scenarios / margin need --ablation
        T.parse_args(no_abl)
    rr_fail = F.make_reranker(os.path.join(d, "rr_fail.json"), passed=False)
    for bad in (["--stop-credit", "1"], ["--w-cov", "0.5"], ["--sft-lr", "1e-5"], ["--selector", "borda_rerank"]):
        args_ = base + bad
        if bad[0] == "--selector":                                     # the gate failed -> borda_rerank refused
            args_ = F.v18_args(d, sp, os.path.join(d, "o2"), reranker=rr_fail) + bad
        with pytest.raises(SystemExit):
            T.parse_args(args_)
    a = T.parse_args(base)
    assert a.selector == "borda_rerank" and a.task1_positions == "all_t1" and a.stop_credit == 0 and a.w_len == 0.5
    a = T.parse_args(F.v18_args(d, sp, os.path.join(d, "o3"), reranker=rr_fail))
    assert a.selector == "borda"
    # v17: the default; v18 arguments refused; the recorded v17 config has no v18 key
    a17 = T.parse_args(["--dry-run", "--fold", "0", "--splits", sp, "--out", os.path.join(d, "v17")])
    assert a17.spec == "v17" and a17.selector == "borda" and a17.task1_positions == "all" and a17.stop_credit == 1
    assert a17.sft_lr == T.SPEC_V17["sft_lr"]
    with pytest.raises(SystemExit):
        T.parse_args(["--dry-run", "--fold", "0", "--splits", sp, "--out", "o", "--w-act", "1"])
    with pytest.raises(SystemExit):
        T.parse_args(["--dry-run", "--fold", "0", "--splits", sp, "--out", "o", "--task1-positions", "all_t1"])
    tr = T.Trainer(a17)
    assert not set(tr.config_record["args"]) & set(T.V18_ONLY_ARGS) and tr.config_record["spec_version"] == "v17"


def test_real_run_needs_label_approval(tmp_path, monkeypatch):
    d = str(tmp_path)
    sp = make_splits(d)
    a = T.parse_args(F.v18_args(d, sp, os.path.join(d, "o")))
    a.dry_run = False
    with pytest.raises(SystemExit) as e:
        T.Trainer(a)
    assert "not approved" in str(e.value)
    labels_sha = T.sha_file(a.act_labels)
    open(a.act_labels + ".APPROVED", "w").write("approved %s\n" % labels_sha)
    T.Trainer(a)                                                       # accepted (construction only)


def test_init_policy_sha_locked(tmp_path):
    d = str(tmp_path)
    sp = make_splits(d)
    args_ = F.v18_args(d, sp, os.path.join(d, "o"))
    i = args_.index("--init-policy-sha")
    args_[i + 1] = "0" * 64
    with pytest.raises(SystemExit) as e:
        T.main(args_)
    assert "policy sha" in str(e.value)


def test_labels_must_cover_train_all(tmp_path):
    d = str(tmp_path)
    sp = make_splits(d)
    lab = F.make_labels(sp, os.path.join(d, "l.jsonl"))
    rows = T.read_jsonl(lab)[1:]
    rewrite(lab, rows)
    m = json.load(open(lab + ".meta.json"))
    m["labels_sha256"] = T.sha_file(lab)                                    # a consistent meta: coverage is the issue
    json.dump(m, open(lab + ".meta.json", "w"))
    with pytest.raises(SystemExit) as e:
        T.main(F.v18_args(d, sp, os.path.join(d, "o"), labels=lab))
    assert "lack" in str(e.value)


def test_kappa_below_04_halves_w_act(tmp_path):
    d = str(tmp_path)
    sp = make_splits(d)
    lab = F.make_labels(sp, os.path.join(d, "l.jsonl"))
    m = json.load(open(lab + ".meta.json"))
    m["fleiss_kappa_coarse"] = 0.3
    json.dump(m, open(lab + ".meta.json", "w"))
    tr = T.Trainer(T.parse_args(F.v18_args(d, sp, os.path.join(d, "o"), labels=lab)))
    assert tr.w_eff["act"] == 0.5 and tr.w_nominal["act"] == 1.0 and "0.300" in tr.w_act_note


# ================================================================== §8 test evaluation (dry run)
def test_eval_test_rl_v18_final_only_and_boot():
    import eval_test_rl as E
    d, sp, out = fresh(updates=2)
    # the v17 comparison run (a v17 dry run with its own test evaluation)
    from test_rl_advantages import args as v17_args
    out17 = os.path.join(d, "pend_f0_v17")
    T.main(v17_args(sp, out17, updates=2))
    E.main(["--final", "--test-seeds", "0", "1", "2", "--test-updates", "0", "2", "--include-base"]
           + v17_args(sp, out17, updates=2))
    fin = json.load(open(os.path.join(out, "final.json")))
    fu = fin["final_update"]
    test = ["--final", "--test-seeds", "0", "1", "2", "--v17-run", out17]
    with pytest.raises(SystemExit):                                      # v18: the final policy only
        E.main(test + ["--test-updates", "0", str(fu)] + F.v18_args(d, sp, out, updates=2,
                                                                   init=os.path.join(d, "init_u0")))
    argv18 = F.v18_args(d, sp, out, updates=2, init=os.path.join(d, "init_u0"),
                        labels=os.path.join(d, "labels_train.jsonl"), labels_val=os.path.join(d, "labels_val.jsonl"),
                        reranker=os.path.join(d, "reranker.json"))
    E.main(test + ["--test-updates", str(fu)] + argv18)
    tm = T.read_jsonl(os.path.join(out, "test_meta.jsonl"))[-1]
    assert tm["updates"] == [fu] and tm["v17_reference"]["test_jsonl_sha256"] == T.sha_file(os.path.join(out17, "test.jsonl"))
    rows = T.read_jsonl(os.path.join(out, "test.jsonl"))
    assert {r["update"] for r in rows} == {fu} and [r for r in rows if r["kind"] == "summary"][0]["clean_key"] == "clean_v18"
    r = subprocess.run([sys.executable, os.path.join(HERE, "eval_test_boot.py"), out], capture_output=True, text=True,
                       cwd=HERE)
    assert r.returncode == 0 and "TEST CHECK PASSED" in r.stdout and "v18 (GRPO+R)" in r.stdout \
        and "v17 SFT (u0)" in r.stdout and "v17 GRPO (u2)" in r.stdout, r.stdout + r.stderr
    rep = report(out, sp)
    assert rep.checks["rl.v18_test"]["fail"] == 0 and rep.checks["rl.v18_test"]["n"] >= 2


# ================================================================== §8 bench generation gates (v18 manifest)
def test_bench_tf_generate_v18_gates_and_provenance(world, tmp_path):
    import hashlib
    import bench_tf_generate as B
    import test_bench_tf_generate as TB
    w = world
    rr = F.make_reranker(str(tmp_path / "rr.json"))
    for u in (0, 5):
        mp = os.path.join(w["run"], "ckpt", "u%05d" % u, "rl_manifest.json")
        m = json.load(open(mp))
        m.update(spec_version="v18", init_adapter="/runs/v17/ckpt/u00000", init_policy_sha=B.V18_INIT_POLICY_SHA,
                 selector="borda_rerank", reranker_sha256=hashlib.sha256(open(rr, "rb").read()).hexdigest(),
                 rerank_topk=2, labels_train_conversations=["r00"])
        json.dump(m, open(mp, "w"))
    with pytest.raises(SystemExit):                                       # v18: only the final update
        B.main(TB._argv(w, 0, extra=["--reranker", rr]))
    with pytest.raises(SystemExit):                                       # borda_rerank needs the run's reranker
        B.main(TB._argv(w, 5))
    other = F.make_reranker(str(tmp_path / "other.json"), passed=False)
    with pytest.raises(SystemExit):
        B.main(TB._argv(w, 5, extra=["--reranker", other]))
    out = str(tmp_path / "g5.jsonl")
    B.main(TB._argv(w, 5, out, extra=["--reranker", rr]))
    p = json.load(open(out + ".provenance.json"))
    assert p["settings"]["reranker_sha256"] == hashlib.sha256(open(rr, "rb").read()).hexdigest()
    assert p["settings"]["reranker_gate"]["passed"] is True and p["settings"]["selector"] == "borda_rerank"
    assert p["settings"]["intent_variant"] == "pend_v18_f2_u5" and "candidate order" in p["settings"]["samples_order"]
    # a test id in the label conversations is a leak
    mp = os.path.join(w["run"], "ckpt", "u00005", "rl_manifest.json")
    m = json.load(open(mp))
    m["labels_train_conversations"] = [w["test_ids"][0]]
    json.dump(m, open(mp, "w"))
    with pytest.raises(SystemExit):
        B.main(TB._argv(w, 5, str(tmp_path / "leak.jsonl"), extra=["--reranker", rr]))


# ================================================================== validation: a lost ledger verdict is not re-run
def test_validation_judge_failure_not_rerun_coverage_null(monkeypatch):
    orig = T.FakeEnv.run_episode
    calls = {}

    def run(self, cid, seed, replicate=0, **kw):
        e = orig(self, cid, seed, replicate=replicate, **kw)
        if seed == 1 and not kw.get("record_generation"):               # validation seed 1: a lost ledger verdict
            calls[(cid, seed)] = calls.get((cid, seed), 0) + 1
            e["episode_counters"] = {"judge_unparseable": 1}
            e["clean"], e["coverage_diag"] = False, None
        return e
    monkeypatch.setattr(T.FakeEnv, "run_episode", run)
    d, sp, out = fresh(updates=1)
    vr = T.read_jsonl(os.path.join(out, "validation.jsonl"))
    per = {}
    for r in vr:
        if r.get("kind") == "episode" and r["seed"] == 1:
            per[(r["update"], r["conversation_id"])] = per.get((r["update"], r["conversation_id"]), 0) + 1
    assert calls and per and all(v == 1 for v in per.values())          # never re-run (v17 would re-run it)
    s0 = [v for v in vr if v.get("kind") == "summary" and v["update"] == 0 and v["task2"]][0]
    assert s0["turn_stats"]["coverage_missing"] == sum(1 for k in per if k[0] == 0) and s0["n_unclean_episodes"] == 0
    assert not fails(report(out, sp))


def test_verify_eval_clean_uses_clean_v18(tmp_path):
    d, sp, out = fresh(updates=1)
    vrows = [r for r in T.read_jsonl(os.path.join(out, "validation.jsonl")) if r.get("kind") == "episode"][:2]
    vrows[0]["episode"]["clean"] = False                                 # a judge failure: v17-unclean, v18-clean
    vrows[1]["episode"]["clean_v18"] = False
    p = str(tmp_path / "eps.jsonl")
    rewrite(p, vrows)
    rep = V.verify([p], os.path.join(out, "run_meta.jsonl"), sp, 0, "validation", "pend", expected_sha="x")
    c = rep.checks["eval.clean"]
    assert c["n"] == 2 and c["fail"] == 1 and "clean_v18" in c["examples"][0]


def test_bench_boot_v18_synthetic(tmp_path):
    import bench_boot_v18 as BB

    def dump(path, lens, empty_at=()):
        rows = {}
        for r, ls in lens.items():
            for t, n in enumerate(ls, 1):
                rows["%s|%d" % (r, t)] = {"record_id": r, "turn_index": t, "utterance_length_in_whitespace_tokens": n,
                                          "turn_with_empty_output": (r, t) in empty_at, "is_first_turn": t == 1,
                                          "utterance_repeating_own_earlier_utterance_in_session_at_unigram_jaccard_0_6": False}
        json.dump({"method_id": os.path.basename(path), "rows": rows}, open(path, "w"))
        return path
    hum = {"r1": [10, 20], "r2": [5, 7, 9]}
    corpus = str(tmp_path / "c.jsonl")
    open(corpus, "w").write("".join(json.dumps({"record_id": r, "chat_messages": [
        {"participant_name": "User", "text": " ".join(["w"] * n)} for n in ls]}) + "\n" for r, ls in hum.items()))
    a = dump(str(tmp_path / "v18.json"), hum)                              # identical to the people: W1 0
    b = dump(str(tmp_path / "u0.json"), {"r1": [30, 30], "r2": [30, 30, 30]})
    c = dump(str(tmp_path / "u5.json"), {"r1": [12, 20], "r2": [5, 7, 9]}, empty_at={("r2", 3)})
    out = BB.main(["--dump-v18", a, "--dump-u0", b, "--dump-u5", c, "--corpus", corpus, "--n-boot", "200",
                   "--out", str(tmp_path / "o.json")])
    w = out["comparisons"]["v18 (GRPO+R) vs v17 SFT (u0)"]["length_w1"]
    assert w["v18"] == 0.0 and w["base"] > 0 and w["diff"] < 0 and w["p_v18_better"] == 1.0
    assert "utterance_repeating_own_earlier_utterance_in_session_at_unigram_jaccard_0_6" in \
        out["comparisons"]["v18 (GRPO+R) vs v17 GRPO (u5)"]


# ================================================================== audit A (labeller)
def test_label_parse_keeps_the_models_move():
    import label_acts as L
    acts = L.load_acts_module()
    sysp = L.build_system_prompt(acts)
    assert "or other" not in sysp and "only when none of the six moves fits" in sysp
    v = L.parse_vote('{"move": "Inquire", "act": "other"}', acts)
    assert v["move"] == "Inquire" and v["act"] == "unknown" and v["move_overridden_by_normalise"]
    v = L.parse_vote('{"move": "Inquire"}', acts)                         # act missing
    assert v["move"] == "Inquire" and v["act"] == "unknown"
    v = L.parse_vote('{"move": "navigate", "act": "ask_more"}', acts)      # an act of another move: the move wins
    assert v["move"] == "Navigate" and v["act"] == "unknown" and v["normalised"] == ["Inquire", "ask_more"]
    v = L.parse_vote('{"move": "Reveal", "act": "narrow"}', acts)
    assert (v["move"], v["act"]) == ("Reveal", "narrow") and not v["move_overridden_by_normalise"]
    v = L.parse_vote('{"move": "Other", "act": "whatever"}', acts)
    assert (v["move"], v["act"]) == ("Other", "other")
    v = L.parse_vote('{"move": "", "act": "ask_more"}', acts)              # no valid move: acts.normalise
    assert (v["move"], v["act"]) == ("Inquire", "ask_more")
    assert L.parse_answer('{"move": "Inquire", "act": "zzz"}', acts) == ("Inquire", "unknown")


def test_label_meta_counts_overrides_and_refusals(tmp_path):
    import label_acts as L
    sp, corpus, cb = _label_world(str(tmp_path))
    out = str(tmp_path / "lab.jsonl")
    argv = ["--fold", "0", "--split", "train_all", "--splits", sp, "--corpus", corpus, "--codebook", cb, "--out", out]
    m = L.main(argv, chat=lambda s, u, sd: ('{"move": "Inquire", "act": "other"}', "stop"))
    assert m["n_move_overridden"] == 15 and m["n_act_unknown"] == 15
    assert all(r["q7"] == {"Inquire": 1.0} for r in (json.loads(l) for l in open(out)))
    os.remove(out + ".meta.json")                                          # a label file without its meta: refused
    with pytest.raises(SystemExit) as e:
        L.main(argv, chat=lambda s, u, sd: ('{"move": "Note", "act": "confirm"}', "stop"))
    assert "without its" in str(e.value)
    calls = {"n": 0}

    def down(s, u, sd):
        calls["n"] += 1
        raise OSError("connection refused")
    with pytest.raises(SystemExit) as e:                                   # 5 transport errors in a row: stop early
        L.main(argv[:-1] + [str(tmp_path / "x.jsonl")], chat=down)
    assert "transport errors" in str(e.value) and calls["n"] == L.MAX_CONSECUTIVE_TRANSPORT_ERRORS
    assert not os.path.exists(str(tmp_path / "x.jsonl"))


def test_label_test_side_refused(tmp_path):
    import label_acts as L
    sp, corpus, cb = _label_world(str(tmp_path))
    d = json.load(open(sp))
    d["folds"][0]["test_all"].append("c0")                                 # a broken split: a test id in train_all
    json.dump(d, open(sp, "w"))
    with pytest.raises(AssertionError):
        L.main(["--fold", "0", "--split", "train_all", "--splits", sp, "--corpus", corpus, "--codebook", cb,
                "--out", str(tmp_path / "t.jsonl")], chat=lambda s, u, sd: ('{"move": "Note", "act": "confirm"}', "stop"))


def test_trainer_refuses_label_meta_of_another_file(tmp_path):
    d = str(tmp_path)
    sp = make_splits(d)
    lab = F.make_labels(sp, os.path.join(d, "l.jsonl"))
    rows = T.read_jsonl(lab)
    rows[0]["n_valid_votes"] = 2                                           # the file changed after its meta
    rewrite(lab, rows)
    with pytest.raises(SystemExit) as e:
        T.Trainer(T.parse_args(F.v18_args(d, sp, os.path.join(d, "o"), labels=lab)))
    assert "labels_sha256" in str(e.value)


def test_verify_checks_validation_label_meta():
    d, sp, out = fresh(updates=1)
    args = T.read_jsonl(os.path.join(out, "run_meta.jsonl"))[0]["config"]["args"]
    mp = args["act_labels_val"] + ".meta.json"
    m = json.load(open(mp))
    m.update(ngram8_overlap=3, opened_files=[VV.BENCH + "/data/req_shards_v1.json"])
    json.dump(m, open(mp, "w"))
    rep = report(out, sp)
    ex = rep.checks["rl.v18_labels"]["examples"]
    # (a dry run waives the 8-gram record; the opened benchmark file is a FAIL in any run)
    assert rep.checks["rl.v18_labels"]["fail"] >= 1 and any("validation labels meta" in x for x in ex)


# ================================================================== audit B
def test_resume_refuses_changed_labels_or_reranker_even_with_allow_code_change(monkeypatch):
    orig = T.Trainer.finalize_v18
    monkeypatch.setattr(T.Trainer, "finalize_v18", lambda self, u, r: (_ for _ in ()).throw(SystemExit("crash")))
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    with pytest.raises(SystemExit):                                      # an unfinished run (crash before final.json)
        T.main(F.v18_args(d, sp, out, updates=1))
    monkeypatch.setattr(T.Trainer, "finalize_v18", orig)
    args_ = F.v18_args(d, sp, out, "--resume", "--allow-code-change", updates=1,
                       init=os.path.join(d, "init_u0"), labels=os.path.join(d, "labels_train.jsonl"),
                       labels_val=os.path.join(d, "labels_val.jsonl"), reranker=os.path.join(d, "reranker.json"))
    rr = os.path.join(d, "reranker.json")
    j = json.load(open(rr))
    j["w"][0] += 0.1                                                     # another reranker (gate still passed)
    json.dump(j, open(rr, "w"))
    with pytest.raises(SystemExit) as e:
        T.main(args_)
    assert "reranker_sha256" in str(e.value) and "pinned" in str(e.value)
    F.make_reranker(rr)                                                  # back to the run's reranker: resume accepted
    lv = os.path.join(d, "labels_val.jsonl")
    rows = T.read_jsonl(lv)
    rows[0]["n_valid_votes"] = 2
    rewrite(lv, rows)
    with pytest.raises(SystemExit) as e:
        T.main(args_)
    assert "labels_val_sha256" in str(e.value)


def test_aborted_scales_hold_their_rows_and_verify_matches_them(monkeypatch):
    d = tempfile.mkdtemp()
    sp = make_splits(d)
    out = os.path.join(d, "run")
    calls = {"n": 0}
    orig = T.FakeLearner.update

    def aborting(self, samples, cfg, seed, aux=None, aux_orders=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RA.MismatchAbort(0.5)
        return orig(self, samples, cfg, seed, aux=aux, aux_orders=aux_orders)
    monkeypatch.setattr(T.FakeLearner, "update", aborting)
    for _ in range(2):                                                   # two aborted attempts of update 1
        with pytest.raises(SystemExit):
            T.main(F.v18_args(d, sp, out, *([] if _ == 0 else ["--resume"]), updates=1))
    T.main(F.v18_args(d, sp, out, "--resume", updates=1))
    import glob
    ab = sorted(glob.glob(os.path.join(out, "adv_scales.aborted_*.json")))
    assert len(ab) == 2 and all(json.load(open(p))["batch_ids"] for p in ab)
    assert not fails(report(out, sp))
    a0 = json.load(open(ab[0]))
    a0["batch_ids"] = a0["batch_ids"][1:]                                # rows that do not hash to its fingerprint
    json.dump(a0, open(ab[0], "w"))
    assert report(out, sp).checks["rl.v18_scales"]["fail"] >= 1


def test_verify_recomputes_weights_effective():
    d, sp, out = fresh(updates=1)
    p = os.path.join(out, "run_meta.jsonl")
    meta = T.read_jsonl(p)
    for m in meta:
        m["config"]["v18"]["weights_effective"]["act"] = 0.5          # claims a halved w_act, kappa 0.62 says 1.0
    rewrite(p, meta)
    rep = report(out, sp)
    assert any("weights_effective" in x for x in rep.checks["rl.v18_settings"]["examples"])


def test_test_rows_check_catches_foreign_policy_rows():
    import eval_test_rl as E
    d, sp, out = fresh(updates=2)
    from test_rl_advantages import args as v17_args
    out17 = os.path.join(d, "pend_f0_v17")
    T.main(v17_args(sp, out17, updates=2))
    E.main(["--final", "--test-seeds", "0", "1", "2", "--test-updates", "0", "2", "--include-base"]
           + v17_args(sp, out17, updates=2))
    fu = json.load(open(os.path.join(out, "final.json")))["final_update"]
    argv18 = F.v18_args(d, sp, out, updates=2, init=os.path.join(d, "init_u0"),
                        labels=os.path.join(d, "labels_train.jsonl"), labels_val=os.path.join(d, "labels_val.jsonl"),
                        reranker=os.path.join(d, "reranker.json"))
    E.main(["--final", "--test-seeds", "0", "1", "2", "--v17-run", out17, "--test-updates", str(fu)] + argv18)
    f0 = json.load(open(sp))["folds"][0]
    rows = T.read_jsonl(os.path.join(out, "test.jsonl"))
    assert VV.check_test_rows(out, rows, fu, set(f0["test"]), set(f0["test_all"]), [0, 1, 2], False) == []
    bad = [dict(r) for r in rows]
    t1 = next(r for r in bad if r["kind"] == "task1")
    t1["end_probs"] = t1["end_probs"][1:]                                # a missing decision point
    rewrite(os.path.join(out, "test.jsonl"), bad)
    r = subprocess.run([sys.executable, os.path.join(HERE, "eval_test_boot.py"), out], capture_output=True, text=True,
                       cwd=HERE)
    assert r.returncode == 1 and "probe points" in r.stdout, r.stdout + r.stderr
    assert report(out, sp).checks["rl.v18_test"]["fail"] >= 1


def test_bench_boot_says_2afc_is_point_value_only(tmp_path):
    import bench_boot_v18 as BB
    rows = {"r1|1": {"record_id": "r1", "turn_index": 1, "utterance_length_in_whitespace_tokens": 3,
                     "turn_with_empty_output": False}}
    for n in ("a", "b", "c"):
        json.dump({"method_id": n, "rows": rows}, open(str(tmp_path / (n + ".json")), "w"))
    corpus = str(tmp_path / "c.jsonl")
    open(corpus, "w").write(json.dumps({"record_id": "r1", "chat_messages": [{"participant_name": "User",
                                                                              "text": "a b c"}]}) + "\n")
    out = BB.main(["--dump-v18", str(tmp_path / "a.json"), "--dump-u0", str(tmp_path / "b.json"), "--dump-u5",
                   str(tmp_path / "c.json"), "--corpus", corpus, "--n-boot", "20", "--out", str(tmp_path / "o.json")])
    k = [x for x in out["not_bootstrapped"] if "two_alternative_forced_choice" in x]
    assert k and "POINT VALUE ONLY" in out["not_bootstrapped"][k[0]]


def test_label_acts_refuses_a_server_that_is_not_ours():
    import label_acts as L
    assert L.local_port_of("http://127.0.0.1:8029/v1") == 8029 and L.local_port_of("https://api.x.com/v1") is None
    seen = []
    assert L.server_is_ours(8029, user="me", pgrep=lambda c: seen.append(c) or 0)
    assert seen[0][:3] == ["pgrep", "-u", "me"] and "--port 8029" in seen[0][-1]
    assert not L.server_is_ours(8029, user="me", pgrep=lambda c: 1)


# ================================================================== 2026-10-02 verify false positives on the fold-2 run
def test_full_verify_skips_v17_reward_components_for_v18():
    d, sp, out = fresh(updates=1)
    rep = V.verify([], os.path.join(out, "run_meta.jsonl"), sp, 0, "train", "pend", rl_dir=out, expected_sha="x")
    c = rep.checks["rl.reward_components"]
    assert c["fail"] == 0 and any("covered by rl.v18" in n for n in c["notes"])


def _r0_rows(tok, counted, charged):
    rows = [{"r0_len_truncated_total": counted, "process_token": tok,
             "episode_counters": {"r0_len_truncated": 1 if i < charged else 0}} for i in range(3)]
    return rows


def _meta(out, rows):
    p = os.path.join(out, "run_meta.jsonl")
    meta = T.read_jsonl(p)
    base = meta[0]
    rewrite(p, [dict(base, **r) for r in rows])


def test_r0_attribution_tolerates_only_a_crashed_process():
    d, sp, out = fresh(updates=1)
    t0 = 1790867243
    tok = "3926060-%d-3771cc8a" % t0
    # the process's launch at t0+60 was followed by a --resume (no stop row between): crashed -> WARN
    _meta(out, [{"kind": "start", "time": t0 + 60}, {"kind": "resume", "time": t0 + 9000},
                {"kind": "stop", "time": t0 + 20000}])
    rep = V.Report()
    V.check_r0_attribution(_r0_rows(tok, 3, 2), rep, out)
    c = rep.checks["trunc.r0_attributed"]
    assert c["fail"] == 0 and c["warn"] and tok in c["notes"][0]
    # more uncharged incidents than rollout workers: FAIL
    rep = V.Report()
    V.check_r0_attribution(_r0_rows(tok, 3 + 10, 2), rep, out)
    assert rep.checks["trunc.r0_attributed"]["fail"] == 1
    # the process's launch is the last one (no later --resume): not a crash -> FAIL
    _meta(out, [{"kind": "start", "time": t0 + 60}, {"kind": "stop", "time": t0 + 20000}])
    rep = V.Report()
    V.check_r0_attribution(_r0_rows(tok, 3, 2), rep, out)
    assert rep.checks["trunc.r0_attributed"]["fail"] == 1
    # a stop row between the launch and the next one: ended cleanly -> FAIL
    _meta(out, [{"kind": "start", "time": t0 + 60}, {"kind": "stop", "time": t0 + 100}, {"kind": "resume", "time": t0 + 9000}])
    rep = V.Report()
    V.check_r0_attribution(_r0_rows(tok, 3, 2), rep, out)
    assert rep.checks["trunc.r0_attributed"]["fail"] == 1
    # without a v18 run dir (v17 / no rl_dir): unchanged, FAIL
    rep = V.Report()
    V.check_r0_attribution(_r0_rows(tok, 3, 2), rep)
    assert rep.checks["trunc.r0_attributed"]["fail"] == 1
