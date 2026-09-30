"""SPEC v17 (ops/SPEC_v17_sft_pend.md, user 2026-09-30): one or more dry-run tests per item of §10 and of the
implementation checklist item 41 (no GPU; the torch-only parts are in test_rl_algos_server.py and smoke_v17.py)."""
import json
import math
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_algos as RA  # noqa: E402
import threshold_control as TC  # noqa: E402
import train_planner_rl as T  # noqa: E402
import verify_pipeline as V  # noqa: E402
from test_rl_advantages import args, make_splits, strip  # noqa: E402


def fresh(*extra, updates=3, margin=100.0):
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    T.main(args(sp, out, *extra, updates=updates, margin=margin))
    return sp, out


def v17_report(out, sp):
    rep = V.Report()
    V.check_v17(out, rep, sp)
    return rep


def no_fail(rep):
    return all(c["fail"] == 0 for c in rep.checks.values())


def fails(rep):
    return {k: c for k, c in rep.checks.items() if c["fail"]}


def _rewrite(path, rows):
    with open(path, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def trainer_for(sp, out, *extra, updates=3):
    tr = T.Trainer(T.parse_args(args(sp, out, *extra, updates=updates)))
    tr.build()
    return tr


# ------------------------------------------------------------------ §1.1 SFT data
def test_sft_data_every_point_three_plans_and_counts(monkeypatch):
    calls = []
    orig = T.FakeEnv.task1_sample

    def rec(self, cid, t, prompt, real_final, G, temperature, top_p, seed):
        calls.append((cid, t, G, temperature, top_p, seed))
        return orig(self, cid, t, prompt, real_final, G, temperature, top_p, seed)
    monkeypatch.setattr(T.FakeEnv, "task1_sample", rec)
    capped_cid = "c03"
    orig_prompts = T.FakeEnv.task1_prompts

    def prompts(self, cid):
        ps = orig_prompts(self, cid)
        if cid == capped_cid:
            ps[1]["emitted_capped"] = True           # message 2 capped -> points t >= 3 skipped
        return ps
    monkeypatch.setattr(T.FakeEnv, "task1_prompts", prompts)
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    T.main(args(sp, out, updates=1))
    f0 = json.load(open(sp))["folds"][0]
    em = json.load(open(os.path.join(out, "sft_examples_meta.json")))
    ex = T.read_jsonl(os.path.join(out, "sft_examples.jsonl"))
    env = T.FakeEnv(None)
    assert em["conversations"] == sorted(f0["train_all"])
    for cid in f0["train_all"]:
        n = env.human_turns(cid)
        pts = sorted(p[1] for p in em["points"] if p[0] == cid)
        assert pts == list(range(2, n + 1))                                   # every decision point t = 2..n
        for p in em["points"]:
            if p[0] == cid:
                assert p[3] == ("skipped_capped_history" if (cid == capped_cid and p[1] >= 3) else "ok")
    sft_calls = [c for c in calls if c[2] == 1]
    ok_pts = [(p[0], p[1]) for p in em["points"] if p[3] == "ok"]
    assert len(sft_calls) == 3 * len(ok_pts)                                 # 1 greedy + 2 sampled per point (N1)
    for cid, t in ok_pts:
        cs = [c for c in sft_calls if c[0] == cid and c[1] == t]
        assert [c[3] for c in cs] == [0.0, 1.0, 1.0] and all(c[4] == 1.0 for c in cs)
        assert [c[5] for c in cs] == [T.seed_of(0, "sft", cid, t, k) for k in range(3)]
    assert all(e["conversation_id"] in f0["train_all"] and e["conversation_id"] not in f0["forbidden_for_training"]
               for e in ex)
    assert all(e["target_ids"] == (e["target_true"] if e["t"] == e["n"] else e["target_false"]) for e in ex)
    c = em["counts"]
    assert c["n_examples"] == len(ex) and c["n_plans"] == 3 * len(ok_pts)
    assert c["n_plans"] == c["n_examples"] + c["unparsed"] + c["hit_max_new"] + c["invalid_value"] + c["mask_not_found"]
    assert c["unparsed"] > 0 and c["mask_not_found"] > 0                     # both drop reasons exercised and counted
    assert c["n_points_skipped_capped"] == sum(1 for p in em["points"] if p[3] != "ok") > 0
    base = T.read_jsonl(os.path.join(out, "base_pend_train.jsonl"))
    assert sorted((b["conversation_id"], b["t"]) for b in base) == sorted(ok_pts)
    assert len({b["policy_sha"] for b in base}) == 1
    assert base[0]["policy_sha"] == json.load(open(os.path.join(out, "ckpt", "sft_e0", "state.json")))["policy_sha"]
    assert no_fail(v17_report(out, sp)), fails(v17_report(out, sp))


# ------------------------------------------------------------------ §1.3 SFT choice
def test_sft_choice_rule():
    row = lambda k, nll, ninv: {"epoch": k, "val": {"nll": nll, "n_invalid": ninv}}
    assert T.sft_choice([row(0, 0.5, 0), row(1, 0.3, 2), row(2, 0.4, 0)]) == 1               # lowest nll
    assert T.sft_choice([row(0, 0.3, 2), row(1, 0.3, 1), row(2, 0.3, 1)]) == 1               # then fewer invalid, earlier
    assert T.sft_choice([row(0, None, 0), row(1, 0.9, 5)]) == 1                               # no valid point ranks last
    assert T.sft_choice([row(0, 0.2, 0), row(1, 0.2, 0)]) == 0                                # tie -> the earlier


def test_sft_selection_u0_and_verify():
    sp, out = fresh(updates=1)
    rows = T.read_jsonl(os.path.join(out, "sft.jsonl"))
    eps = [r for r in rows if r["kind"] == "epoch"]
    sel = [r for r in rows if r["kind"] == "selection"][-1]
    assert [r["epoch"] for r in eps] == [0, 1, 2, 3]
    assert sel["chosen_epoch"] == T.sft_choice(eps)
    chosen = next(r for r in eps if r["epoch"] == sel["chosen_epoch"])
    st0 = json.load(open(os.path.join(out, "ckpt", "u00000", "state.json")))
    assert st0["policy_sha"] == chosen["policy_sha"] and st0["sft"]["chosen_epoch"] == sel["chosen_epoch"]
    assert all(r["val_ids"] == sorted(json.load(open(sp))["folds"][0]["validation_all"]) for r in eps)
    for r in eps:
        pts = [p for c in r["val_end_probs"].values() for p in c]
        assert abs(T.T1.task1_prob_metrics(pts)["nll"] - r["val"]["nll"]) < 1e-12
    assert eps[0]["train_loss"] is None and all(r["train_loss"] is not None for r in eps[1:])
    meta = T.read_jsonl(os.path.join(out, "run_meta.jsonl"))
    s = [m for m in meta if m["kind"] == "sft"][0]
    assert s["ref_policy_sha"] == st0["policy_sha"] == s["u0_policy_sha"] and s["sft_chosen_epoch"] == sel["chosen_epoch"]
    assert s["sft_examples_sha256"] == T.sha_file(os.path.join(out, "sft_examples.jsonl"))
    assert no_fail(v17_report(out, sp))
    # tampering: the logged choice is not the recomputed one
    p = os.path.join(out, "sft.jsonl")
    rows[-1]["chosen_epoch"] = (sel["chosen_epoch"] + 1) % 4
    _rewrite(p, rows)
    assert v17_report(out, sp).checks["rl.sft_choice"]["fail"] >= 1


def test_sft_epoch0_can_be_chosen(monkeypatch):
    """If no SFT epoch beats the start policy on validation_all NLL, u0 IS the start policy (recorded as such)."""
    monkeypatch.setattr(T.FakeLearner, "sft_step", lambda self, batch: (
        [self.theta.__setitem__(i, self.theta[i] - (0.4 if i == 3 else -0.4)) for i in range(5)] and None)
        or {"loss": 1.0, "grad_norm": 1.0, "n": len(batch)})                 # an SFT that makes things worse
    sp, out = fresh(updates=1)
    sel = [r for r in T.read_jsonl(os.path.join(out, "sft.jsonl")) if r["kind"] == "selection"][-1]
    assert sel["chosen_epoch"] == 0 and sel["u0_is_start_policy"] is True
    st0 = json.load(open(os.path.join(out, "ckpt", "u00000", "state.json")))
    assert st0["policy_sha"] == json.load(open(os.path.join(out, "ckpt", "sft_e0", "state.json")))["policy_sha"]
    assert no_fail(v17_report(out, sp))


# ------------------------------------------------------------------ SFT resume (§1.3, S7)
def test_sft_crash_resumes_from_cached_examples(monkeypatch):
    d = tempfile.mkdtemp()
    sp = make_splits(d)
    full, part = os.path.join(d, "full"), os.path.join(d, "part")
    T.main(args(sp, full, updates=2))
    orig = T.Trainer.sft_probe

    def crash(self, attempt, k, examples, steps):
        if k == 2:
            raise RuntimeError("simulated crash in SFT epoch 2")
        return orig(self, attempt, k, examples, steps)
    monkeypatch.setattr(T.Trainer, "sft_probe", crash)
    with pytest.raises(RuntimeError):
        T.main(args(sp, part, updates=2))
    monkeypatch.setattr(T.Trainer, "sft_probe", orig)
    assert not os.path.exists(os.path.join(part, "ckpt", "LATEST.json"))       # u0 / LATEST only after SFT
    assert not os.path.exists(os.path.join(part, "ckpt", "u00000"))
    # S7: a non-resume launch over the leftovers is refused
    with pytest.raises(SystemExit) as e:
        T.main(args(sp, part, updates=2))
    assert "SFT leftovers" in str(e.value)
    # --resume re-runs SFT from the cached examples (no new plan is generated)
    calls = []
    o_s = T.FakeEnv.task1_sample
    monkeypatch.setattr(T.FakeEnv, "task1_sample", lambda self, *a, **k: calls.append(a[4]) or o_s(self, *a, **k))
    T.main(args(sp, part, "--resume", updates=2))
    assert all(G != 1 for G in calls), "SFT plans regenerated on resume"
    rows = T.read_jsonl(os.path.join(part, "sft.jsonl"))
    assert [r["attempt"] for r in rows if r["kind"] == "attempt_start"] == [1, 2]
    for f in ("final.json",):
        assert strip([json.load(open(os.path.join(full, f)))]) == strip([json.load(open(os.path.join(part, f)))])
    assert json.load(open(os.path.join(full, "ckpt", "u00000", "state.json")))["policy_sha"] == \
        json.load(open(os.path.join(part, "ckpt", "u00000", "state.json")))["policy_sha"]
    assert no_fail(v17_report(part, sp)), fails(v17_report(part, sp))


def test_u0_exists_sft_never_reruns(monkeypatch):
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    with pytest.raises(SystemExit):
        T.main(args(sp, out, "--dry-run-crash-after-episodes", "14", updates=3))
    assert os.path.exists(os.path.join(out, "ckpt", "u00001"))
    monkeypatch.setattr(T.Trainer, "sft_stage", lambda self: (_ for _ in ()).throw(AssertionError("SFT re-run")))
    T.main(args(sp, out, "--resume", updates=3))
    assert json.load(open(os.path.join(out, "final.json")))["final_update"] == 3


def test_mismatching_sft_cache_is_refused(monkeypatch):
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    orig = T.Trainer.sft_probe
    monkeypatch.setattr(T.Trainer, "sft_probe", lambda self, a, k, e, s: (_ for _ in ()).throw(RuntimeError("x"))
                        if k == 1 else orig(self, a, k, e, s))
    with pytest.raises(RuntimeError):
        T.main(args(sp, out, updates=1))
    monkeypatch.setattr(T.Trainer, "sft_probe", orig)
    mp = os.path.join(out, "sft_examples_meta.json")
    m = json.load(open(mp))
    m["key"]["start_policy_sha"] = "0" * 64
    json.dump(m, open(mp, "w"))
    with pytest.raises(SystemExit) as e:
        T.main(args(sp, out, "--resume", updates=1))
    assert "not reused, not overwritten" in str(e.value)


# ------------------------------------------------------------------ §2 ref adapter (B4)
def test_ref_stub_and_saved_checkpoints_have_no_ref():
    sp, out = fresh(updates=2)
    tr = trainer_for(sp, out)                                                    # u0 exists -> ref loaded at build
    assert tr.learner.has_ref and tr.learner.ref_path.endswith(os.path.join("u00000", "adapter"))
    assert tr.learner.ref_theta == json.load(open(os.path.join(out, "ckpt", "u00000", "fake_learner.json")))["theta"]
    with pytest.raises(AssertionError):
        tr.learner.load_ref(tr.learner.ref_path)                                  # never twice
    fl = T.FakeLearner("grpo", {}, 1e-5)
    with pytest.raises(AssertionError):
        fl.prepare([{"prompt_ids": [1, 2], "gen_ids": [8, 1, 7]}])               # no update without the reference
    assert not [dp for dp, ds, _ in os.walk(os.path.join(out, "ckpt")) for x in ds if x == "ref"]
    rep = v17_report(out, sp)
    assert no_fail(rep)
    os.makedirs(os.path.join(out, "ckpt", "u00001", "adapter", "ref"))            # a checkpoint with the ref inside
    assert v17_report(out, sp).checks["rl.ref_adapter"]["fail"] == 1


class _P:
    def __init__(self, g):
        self.requires_grad = g


class _MockPeft:
    """Just enough of a PeftModel for load_ref_adapter / save_adapter (no torch)."""

    def __init__(self):
        self.params = {"m.lora_A.default.weight": _P(True), "m.lora_B.default.weight": _P(True), "m.base": _P(False)}
        self.peft_config = {"default": object()}
        self.active, self.saved = "default", []

    def named_parameters(self):
        return list(self.params.items())

    def parameters(self):
        return list(self.params.values())

    def load_adapter(self, path, adapter_name, is_trainable):
        assert is_trainable is False
        self.peft_config[adapter_name] = object()
        self.params["m.lora_A.%s.weight" % adapter_name] = _P(False)
        self.params["m.lora_B.%s.weight" % adapter_name] = _P(False)

    def set_adapter(self, name):                                               # PEFT: the active adapter is trainable
        self.active = name
        for n, p in self.params.items():
            if ".lora_" in n:
                p.requires_grad = (".%s." % name) in n

    def save_pretrained(self, path, selected_adapters=None):
        self.saved.append((path, selected_adapters))
        os.makedirs(path, exist_ok=True)


class _Opt:
    def __init__(self, params):
        self.param_groups = [{"name": "policy", "params": params}]


def test_load_ref_adapter_and_save_default_only_mock():
    m = _MockPeft()
    opt = _Opt([p for p in m.parameters() if p.requires_grad])
    assert RA.load_ref_adapter(m, "u0/adapter", opt) == 2
    assert m.active == "default" and not any(p.requires_grad for n, p in m.params.items() if ".ref." in n)
    with pytest.raises(AssertionError):
        RA.load_ref_adapter(m, "u0/adapter", opt)                               # already loaded
    m2 = _MockPeft()
    with pytest.raises(AssertionError):
        RA.load_ref_adapter(m2, "u0/adapter", _Opt([m2.params["m.base"]]))     # optimizer set != trainable set
    tl = RA.TorchLearner.__new__(RA.TorchLearner)
    tl.model = m
    d = tempfile.mkdtemp()
    tl.save_adapter(d)
    assert m.saved == [(os.path.join(d, "adapter"), ["default"])]
    with pytest.raises(ValueError):
        RA.setup_policy(None, None)                                             # B1: the LoRA init must be seeded
    with pytest.raises(ValueError):
        RA.setup_policy(None, 1, init_adapter="x")                              # v17: no init adapter


# ------------------------------------------------------------------ §3.2 Brier reward, drops, B7 groups, advantages (S2)
def _sample(status, rep, p=None, stop=False):
    x = {"t": 3, "replicate": rep, "real_final": True, "ended_planner": stop, "decision_valid": status != "invalid",
         "mask_ok": status == "valid", "prefix_ids": [8] if status == "valid" else None,
         "target_true": [1] if status == "valid" else None, "target_false": [0] if status == "valid" else None,
         "planner_gen": {"prompt_ids": [3, 3], "gen_ids": [8, 1 if stop else 9, 7],
                         "stop_mask": [0, 1, 0] if status == "valid" else None, "note_mask": None,
                         "temperature": 1.0, "top_p": 1.0, "seed": rep}}
    return x


def test_brier_reward_and_drop_rule():
    d = tempfile.mkdtemp()
    sp = make_splits(d)
    tr = T.Trainer(T.parse_args(args(sp, os.path.join(d, "run"))))
    tr.build()
    tr.learner.theta[3] = 0.7
    smp = [_sample("valid", 0, stop=True), _sample("invalid", 1), _sample("dropped", 2), _sample("valid", 3)]
    tr.score_task1(smp, True)
    p = 1 / (1 + math.exp(-0.7))
    assert smp[0]["status"] == "valid" and abs(smp[0]["p_end"] - p) < 1e-12 and abs(smp[0]["reward"] - (1 - (p - 1) ** 2)) < 1e-12
    assert smp[1]["status"] == "invalid" and smp[1]["reward"] == 0.0 and smp[1]["p_end"] is None
    assert smp[2]["status"] == "dropped" and smp[2]["dropped"] and smp[2]["reward"] is None \
        and smp[2]["drop_reason"] == "stop_mask_mismatch"
    tr.score_task1([_sample("valid", 0)], False)                               # y = 0 at a non-final point
    x = _sample("valid", 0)
    tr.score_task1([x], False)
    assert abs(x["reward"] - (1 - p ** 2)) < 1e-12


def test_task1_groups_b7_and_advantage_placement():
    d = tempfile.mkdtemp()
    sp = make_splits(d)
    tr = T.Trainer(T.parse_args(args(sp, os.path.join(d, "run"))))
    tr.build()
    tr.learner.theta[3] = 0.3

    def row(cid, smp):
        tr.score_task1(smp, True)
        return {"conversation_id": cid, "t": 3, "n_real": 3, "real_final": True, "policy_version": 0, "policy_sha": "s",
                "samples": smp}
    rows = [row("a", [_sample("valid", 0), _sample("dropped", 1), _sample("dropped", 2), _sample("dropped", 3)]),   # lt2
            row("b", [_sample("invalid", 0), _sample("invalid", 1), _sample("dropped", 2), _sample("invalid", 3)]),  # all invalid
            row("c", [_sample("valid", 0), _sample("valid", 1), _sample("dropped", 2), _sample("valid", 3)]),       # zero std
            row("e", [_sample("valid", 0), _sample("invalid", 1), _sample("dropped", 2), _sample("valid", 3)])]     # used
    samples, cnt, recs = tr.task1_samples(rows, 0, "s")
    assert cnt == {"groups_skipped_zero_std": 1, "n_groups_lt2": 1, "n_groups_all_invalid": 1, "n_groups_used": 1}
    assert [r[2] for r in recs] == ["lt2", "all_invalid", "zero_std", "used"]
    assert len(samples) == 3 and {s["replicate"] for s in samples} == {0, 1, 3}      # the dropped sample never enters
    R = [x["reward"] for x in rows[3]["samples"] if not x["dropped"]]
    A = [r - sum(R) / 3 for r in R]
    for s, a in zip(samples, A):
        tok = RA.token_advantages(s, 3)
        assert s["adv_stop"] == 0.0 and s["source"] == "task1"
        if s["replicate"] == 1:                                                     # invalid: every token
            assert s["adv"] == a and s["prefix_mask"] is None and tok == [a, a, a]
        else:                                                                       # valid: the prefix only
            assert s["adv"] == 0.0 and s["adv_prefix"] == a and s["prefix_mask"] == [1, 0, 0]
            assert tok == [a, 0.0, 0.0]                                             # never on the value token
    assert recs[3][3] == [[0, "valid", A[0], "prefix"], [1, "invalid", A[1], "all"], [3, "valid", A[2], "prefix"]]


def test_token_advantages_formula():
    s = {"adv": 0.5, "adv_prefix": 2.0, "adv_stop": -1.0, "note_mask": [0, 1, 0, 0], "prefix_mask": [1, 1, 0, 0],
         "stop_mask": [0, 0, 1, 0]}
    assert RA.token_advantages(s, 4) == [0.5 + 2.0, 0.0, 0.5 - 1.0, 0.5]
    assert RA.token_advantages({"adv": 1.0}, 2) == [1.0, 1.0]                  # note_mask None = all 0
    with pytest.raises(AssertionError):
        RA.token_advantages(dict(s, stop_mask=[0, 1, 0, 0]), 4)                 # prefix / stop overlap
    with pytest.raises(AssertionError):
        RA.token_advantages(dict(s, note_mask=[0, 1]), 4)                       # wrong length
    with pytest.raises(AssertionError):
        RA.token_advantages({"adv_prefix": 1.0}, 2)                             # a prefix advantage without its mask
    assert RA.prefix_mask_of([0, 0, 1, 1, 0]) == [1, 1, 0, 0, 0] and RA.prefix_mask_of(None) is None


# ------------------------------------------------------------------ §3.3 / §3.4 aux split, optimizer steps (S3, S4)
def test_aux_split():
    assert RA.aux_split(3, [2, 0, 1], 4) == [[2], [0], [1], []]
    assert RA.aux_split(9, list(range(9)), 4) == [[0, 1, 2], [3, 4], [5, 6], [7, 8]]
    assert RA.aux_split(0, [], 4) == [[], [], [], []]
    with pytest.raises(ValueError):
        RA.aux_split(3, [0, 0, 1], 4)


def _fake_samples(n):
    return [{"prompt_ids": [1 + i % 4, 2], "gen_ids": [8, i % 2, 7], "temperature": 1.0, "adv": 0.3 * (i % 3 - 1),
             "stop_mask": [0, 1, 0], "source": "task2"} for i in range(n)]


def _aux(n):
    return [{"prompt_ids": [3 if i % 2 else 1, 2], "prefix_ids": [8], "target_ids": [i % 2], "want_end": bool(i % 2),
             "gen_len": 3, "weight": 0.5} for i in range(n)]


@pytest.mark.parametrize("n,aux,want", [(10, 5, 8), (4, 0, 8), (3, 2, 6), (1, 1, 2), (0, 3, 1), (0, 0, 0)])
def test_optimizer_steps(n, aux, want):
    fl = T.FakeLearner("grpo", {}, 1e-5)
    fl.has_ref = True
    a_ = _aux(aux)
    orders = [list(range(aux)), list(reversed(range(aux)))] if aux else None
    st = fl.update(_fake_samples(n), {"lr": 1e-5, "kl_coef": 0.01}, seed=1, aux=a_ or None, aux_orders=orders)
    assert st["optimizer_steps"] == want
    if aux:
        # every supervision example once per epoch (one step for supervision only)
        assert sum(s["n_aux"] for s in st["steps"]) == aux * (2 if n else 1)
        assert st["aux_only"] is (n == 0) and st["aux_p_correct_before"] is not None
    if n:
        assert st["n_minibatches"] == min(4, n) and st["rl_grad_norm"] is not None and st["rl_grad_norm_max"] is not None


def test_aux_examples_one_per_point():
    sp, out = fresh(updates=2)
    for u in T.read_jsonl(os.path.join(out, "updates.jsonl")):
        ts = u["task1_stats"]
        rows = [r for r in T.read_jsonl(os.path.join(out, "rollouts_task1.jsonl")) if r["update"] == u["update"]]
        want = sorted([r["conversation_id"], r["t"]] for r in rows if any(x["status"] == "valid" for x in r["samples"]))
        assert ts["aux_points"] == want and u["learner_stats"]["aux_n"] == len(want)
        assert u["train_aggregate"]["aux_weight"] == 0.5
        assert u["learner_stats"]["optimizer_steps"] == 8 and len(u["learner_stats"]["steps"]) == 8
        assert u["learner_stats"]["adv_abs_mean_by_source"]["task1"] is not None
        assert sum(s["n_aux"] for s in u["learner_stats"]["steps"]) == 2 * len(want)


def test_task1_groups_every_point_verified():
    sp, out = fresh(updates=2)
    f0 = json.load(open(sp))["folds"][0]
    for u in T.read_jsonl(os.path.join(out, "updates.jsonl")):
        ts = u["task1_stats"]
        assert len(ts["convs"]) == 8 and set(ts["convs"]) <= set(f0["train_all"])
        env = T.FakeEnv(None)
        want = sorted([c, t] for c in ts["convs"] for t in range(2, env.human_turns(c) + 1))
        assert ts["groups"] == want and ts["skipped_capped"] == []
        assert "refill_stop" not in ts and "n_refill_groups" not in u["train_aggregate"]["task1_train"]
        assert {"brier_mean", "p_end_final_mean", "p_end_nonfinal_mean", "n_dropped_mask", "n_invalid",
                "n_groups_lt2"} <= set(ts)
    assert no_fail(v17_report(out, sp))


# ------------------------------------------------------------------ §3.5 length drift (S5)
def _short_episodes(monkeypatch, human=None):
    orig = T.FakeEnv.run_episode

    def short(self, cid, seed, *a, **k):
        ep = orig(self, cid, seed, *a, **k)
        ep["trace"] = ep["trace"][:1]
        ep["trace"][0].update(decision="continue", emitted=True, user="u1")
        ep["emitted_user_turns"], ep["decision_steps"], ep["end_kind"] = 1, 1, "planner_end"
        return ep
    monkeypatch.setattr(T.FakeEnv, "run_episode", short)
    if human is not None:
        monkeypatch.setattr(T.FakeEnv, "human_turns", lambda self, cid: human)


def test_length_drift_stop(monkeypatch):
    _short_episodes(monkeypatch)
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    T.main(args(sp, out, updates=5, margin=1.0))
    fin = json.load(open(os.path.join(out, "final.json")))
    assert fin["final_update"] == 2 and fin["stop_reason"] == "length_drift" and fin["validated"]
    ups = T.read_jsonl(os.path.join(out, "updates.jsonl"))
    assert [u["update"] for u in ups] == [1, 2] and all(u["train_aggregate"]["drift_stat"] < -1.0 for u in ups)
    meta = T.read_jsonl(os.path.join(out, "run_meta.jsonl"))
    assert [(m["kind"], m.get("stop_reason")) for m in meta if m["kind"] == "stop"] == [("stop", "length_drift")]
    assert json.load(open(os.path.join(out, "ckpt", "u00002", "state.json")))["stop_reason"] == "length_drift"
    summ = [(v["update"], v["task2"]) for v in T.read_jsonl(os.path.join(out, "validation.jsonl")) if v["kind"] == "summary"]
    assert summ == [(0, True), (1, False), (2, True)]
    assert no_fail(v17_report(out, sp)), fails(v17_report(out, sp))
    # B2: a stopped run is not resumable
    with pytest.raises(SystemExit) as e:
        T.main(args(sp, out, "--resume", updates=5, margin=1.0))
    assert "finished" in str(e.value)
    # the verifier recomputes the rule: a logged stop reason that does not fit fails
    fp = os.path.join(out, "final.json")
    f = json.load(open(fp))
    f["stop_reason"] = "max_updates"
    json.dump(f, open(fp, "w"))
    assert v17_report(out, sp).checks["rl.v17_stop"]["fail"] >= 1


def test_drift_uses_min_human_turns_t_max(monkeypatch):
    """S5: the human side is min(human_turns, t_max): with 13-message people and 1-turn episodes the drift is 1 - 10."""
    _short_episodes(monkeypatch, human=13)
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    T.main(args(sp, out, updates=5, margin=100.0))
    ups = T.read_jsonl(os.path.join(out, "updates.jsonl"))
    assert all(u["train_aggregate"]["drift_stat"] == 1 - 10 for u in ups)       # not 1 - 13
    assert json.load(open(os.path.join(out, "final.json")))["stop_reason"] == "max_updates"
    assert no_fail(v17_report(out, sp)), fails(v17_report(out, sp))


def test_drift_decision_rule():
    assert T.drift_decision({1: -2.0, 2: -0.5, 3: -2.0, 4: -1.5}, 1.0, 5) == (4, "length_drift")
    assert T.drift_decision({1: -2.0, 2: -1.0}, 1.0, 5) == (None, None)          # -1.0 is not < -1.0
    assert T.drift_decision({1: -2.0, 2: -0.5, 3: 0.0, 4: -2.0, 5: -2.0}, 1.0, 5) == (5, "length_drift")
    assert T.drift_decision({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}, 1.0, 5) == (5, "max_updates")
    assert T.drift_decision({}, 1.0, 5) == (None, None)


# ------------------------------------------------------------------ B2: updates fixed
def test_updates_cannot_change_on_resume():
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    with pytest.raises(SystemExit):
        T.main(args(sp, out, "--dry-run-crash-after-episodes", "14", updates=3))
    with pytest.raises(SystemExit) as e:
        T.main(args(sp, out, "--resume", updates=4))
    assert "updates" in str(e.value)
    with pytest.raises(SystemExit):
        T.parse_args(["--dry-run", "--fold", "0", "--out", "o", "--updates", "4"])   # another count needs --ablation


# ------------------------------------------------------------------ §4 validation schedule (S6)
def test_final_validation_resumes_and_task1_only_summary_does_not_count(monkeypatch):
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    orig = T.Trainer.validate

    def crash(self, u, task2):
        if u == 2 and task2:
            raise RuntimeError("simulated crash in the final validation")
        return orig(self, u, task2)
    monkeypatch.setattr(T.Trainer, "validate", crash)
    with pytest.raises(RuntimeError):
        T.main(args(sp, out, updates=2))
    monkeypatch.setattr(T.Trainer, "validate", orig)
    fin = json.load(open(os.path.join(out, "final.json")))
    assert fin["final_update"] == 2 and fin["validated"] is False
    # a stray Task-1-only summary of the final update must not stand in for the final validation
    p = os.path.join(out, "validation.jsonl")
    rows = T.read_jsonl(p)
    t1only = dict([r for r in rows if r["kind"] == "summary" and r["update"] == 1][0], update=2)
    _rewrite(p, rows + [t1only])
    calls = []
    monkeypatch.setattr(T.Trainer, "one_update", lambda self, u: calls.append(u))
    T.main(args(sp, out, "--resume", updates=2))
    assert calls == []                                                            # no training after final.json
    summ = [(v["update"], v["task2"]) for v in T.read_jsonl(p) if v["kind"] == "summary"]
    assert (2, True) in summ and json.load(open(os.path.join(out, "final.json")))["validated"] is True


def test_validation_task1_on_validation_all_and_verify():
    sp, out = fresh(updates=3)
    f0 = json.load(open(sp))["folds"][0]
    val = T.read_jsonl(os.path.join(out, "validation.jsonl"))
    assert {r["conversation_id"] for r in val if r["kind"] == "task1"} == set(f0["validation_all"]) != set(f0["validation"])
    st0 = json.load(open(os.path.join(out, "ckpt", "u00000", "state.json")))
    assert st0["task1_base"]["update"] == 0 and "nll" in st0["task1_base"]
    rep = v17_report(out, sp)
    assert no_fail(rep)
    # tampering: a Task 2 episode at u1 (only u0 and the final update)
    p = os.path.join(out, "validation.jsonl")
    ep = dict([r for r in val if r["kind"] == "episode" and r["update"] == 0][0], update=1)
    _rewrite(p, val + [ep])
    assert v17_report(out, sp).checks["rl.v17_validation"]["fail"] >= 1


# ------------------------------------------------------------------ §9 the verifier catches tampering
@pytest.mark.parametrize("what", ["brier", "dropped_in_adv", "adv_where", "aux", "steps", "drift", "kl", "sft_example",
                                  "base_pend", "skip_counts", "t1_samples"])
def test_verify_v17_catches(what):
    sp, out = fresh(updates=2)
    assert no_fail(v17_report(out, sp))
    pu, pt = os.path.join(out, "updates.jsonl"), os.path.join(out, "rollouts_task1.jsonl")
    ups, t1 = T.read_jsonl(pu), T.read_jsonl(pt)
    check = {"brier": "rl.v17_task1_reward", "dropped_in_adv": "rl.v17_advantage", "adv_where": "rl.v17_advantage",
             "aux": "rl.v17_aux", "steps": "rl.v17_steps", "drift": "rl.v17_drift", "kl": "rl.v17_update",
             "sft_example": "rl.sft_data", "base_pend": "rl.sft_data", "skip_counts": "rl.v17_task1",
             "t1_samples": "rl.v17_task1"}[what]
    if what == "brier":
        x = next(x for r in t1 for x in r["samples"] if x["status"] == "valid")
        x["reward"] += 0.05
        _rewrite(pt, t1)
    elif what in ("dropped_in_adv", "adv_where"):
        ts = ups[0]["task1_stats"]
        g = next(a for a in ts["advantages"] if a[2] == "used")
        if what == "adv_where":
            g[3][0][3] = "all" if g[3][0][3] == "prefix" else "prefix"
        else:
            g[3].append([9, "dropped", 0.0, "all"])
        _rewrite(pu, ups)
    elif what == "aux":
        ups[0]["learner_stats"]["aux_n"] += 1
        _rewrite(pu, ups)
    elif what == "steps":
        ups[1]["learner_stats"]["optimizer_steps"] = 7
        _rewrite(pu, ups)
    elif what == "drift":
        ups[0]["train_aggregate"]["drift_stat"] -= 0.5
        _rewrite(pu, ups)
    elif what == "kl":
        ups[0]["cfg_used"]["kl_coef"] = 0.04
        _rewrite(pu, ups)
    elif what == "sft_example":
        p = os.path.join(out, "sft_examples.jsonl")
        ex = T.read_jsonl(p)
        ex[0]["conversation_id"] = json.load(open(sp))["folds"][0]["validation"][0]
        _rewrite(p, ex)
    elif what == "base_pend":
        p = os.path.join(out, "base_pend_train.jsonl")
        _rewrite(p, T.read_jsonl(p)[1:])
    elif what == "skip_counts":
        ups[0]["task1_stats"]["groups_skipped_zero_std"] += 1
        _rewrite(pu, ups)
    elif what == "t1_samples":
        t1[0]["samples"] = t1[0]["samples"][:3]
        _rewrite(pt, t1)
    rep = v17_report(out, sp)
    assert rep.checks[check]["fail"] >= 1, (what, rep.checks[check])


def test_verify_v17_ditto_and_settings():
    sp, out = fresh(updates=1)
    mp = os.path.join(out, "run_meta.jsonl")
    meta = T.read_jsonl(mp)
    env = {"arm": "pend", "speaker_path": "/tmp2/mzjiang_usersim/models/Ditto-8B", "speaker_endconv_is_none": True,
           "speaker_class": ["fit_prompts.FitDitto", "ditto_e16.DittoSpeaker", "sepsim.models.Speaker", "builtins.object"],
           "arm_env": {"SEPSIM_ENDMASK_RETRY": "0", "SEPSIM_ENDGATE": "0", "SEPSIM_KEEPEND": "0", "SEPSIM_ENDSCORE": "0"},
           "v2fix": {"SEPSIM_END_PROBE": "0"}, "tree": "/tmp2/mzjiang_usersim/grpo_planner/trees/e1r_cf19400"}
    for m in meta:
        m["config"]["env"] = env
    _rewrite(mp, meta)
    rep = v17_report(out, sp)
    assert rep.checks["rl.v17_ditto"]["fail"] == 0 and rep.checks["rl.v17_ditto"]["n"] >= 9
    for m in meta:
        m["config"]["env"] = dict(env, speaker_endconv_is_none=False, arm_env=dict(env["arm_env"], SEPSIM_ENDGATE="1"))
    _rewrite(mp, meta)
    assert v17_report(out, sp).checks["rl.v17_ditto"]["fail"] == 2
    for m in meta:
        m["config"]["env"] = env
        m["config"]["args"]["kl"] = 0.04
        del m["config"]["args"]["ablation"]
    _rewrite(mp, meta)
    assert v17_report(out, sp).checks["rl.v17_settings"]["fail"] >= 1


def test_full_verify_dispatches_to_v17():
    """B8: verify() on a v17 run takes the v17 branch (the v16 checks do not run), its Task 1 rows are checked against
    the logged 0/1 agreement (the Brier reward is check_v17's), validation rows may use validation_all, no best.json.
    (The dry-run episodes have no prompts / fits, so only these checks are asserted here.)"""
    sp, out = fresh(updates=2)
    rep = V.verify([], os.path.join(out, "run_meta.jsonl"), sp, 0, "train", arm="pend", expected_sha="x", rl_dir=out)
    for name in ("rl.task1_groups", "leak.task1_train_only", "leak.validation_ids", "rl.best", "rl.v17_task1_reward",
                 "rl.v17_advantage", "rl.v17_validation", "rl.sft_choice"):
        assert rep.checks[name]["fail"] == 0 and rep.checks[name]["n"] > 0, (name, rep.checks[name])
    assert "rl.task1_refill" not in rep.checks and "rl.selection" not in rep.checks and "rl.d2_trigger" not in rep.checks


def test_env_describe_speaker_fields():
    """S12: task2_env.describe() names the Speaker's path, class (MRO) and that it has no end token."""
    import inspect
    import task2_env as TE
    src = inspect.getsource(TE.Task2Env.describe)
    for k in ("speaker_path", "speaker_class", "speaker_endconv_is_none"):
        assert '"%s"' % k in src


# ------------------------------------------------------------------ §6 threshold control
def test_threshold_control(tmp_path):
    # train: base P_end systematically too low (true rate at the final points 0.8, the policy says 0.4)
    train = [{"real_final": True, "p_end": 0.4, "valid": True}] * 8 + [{"real_final": False, "p_end": 0.1, "valid": True}] * 8 \
        + [{"real_final": True, "p_end": 0.0, "valid": False}]
    b = TC.fit_offset(train)
    zs = [(TC.logit(p["p_end"]), 1.0 if p["real_final"] else 0.0) for p in train if p["valid"]]
    assert abs(sum(TC.sigmoid(z + b) - y for z, y in zs)) < 1e-9                # first-order optimum
    assert TC.nll(train, b) < TC.nll(train, 0.0)
    assert TC.fit_offset([{"real_final": True, "p_end": 0.3}]) == 50.0          # one label only: at the bound
    ptrain = tmp_path / "base_pend_train.jsonl"
    ptrain.write_text("".join(json.dumps(r) + "\n" for r in train))
    # test file in eval_test_rl's format (update 0 = base of a v16 run)
    rows = []
    for i, n in enumerate((3, 4)):
        cid = "t%d" % i
        turns = [{"t": t, "real_final": t == n, "ended_planner": False, "speaker_blank": False} for t in range(1, n + 1)]
        eps = [{"t": t, "real_final": t == n, "p_end": 0.4 if t == n else 0.1, "valid": t != 2 or i == 0,
                "decision_valid": True, "greedy_end": False} for t in range(2, n + 1)]
        for e in eps:
            if not e["valid"]:
                e["p_end"] = 0.0
        rows.append({"kind": "task1", "update": 0, "policy_sha": "s", "conversation_id": cid,
                     "task1": {"turns": turns, "k1_speaker_blank": False}, "end_probs": eps})
    rows.append({"kind": "summary", "update": 0, "policy_sha": "s", "task1_ids": ["t0", "t1"]})
    ptest = tmp_path / "test.jsonl"
    ptest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = TC.main(["--train", str(ptrain), "--test", str(ptest), "--test-update", "0", "--json-out", str(tmp_path / "o.json")])
    assert res["base_at_0.5"]["term_f1"] == 0.0 and res["threshold_control"]["term_f1"] == 1.0   # the shift ends at n
    assert res["threshold_control"]["n_invalid"] == 1 and res["threshold_control"]["nll"] < res["base_at_0.5"]["nll"]
    shifted = TC.shifted(rows[1]["end_probs"], b)
    assert shifted[0]["p_end"] == 0.0                                          # an unscored point is not shifted
    assert json.load(open(tmp_path / "o.json"))["offset_b"] == b


def test_crash_between_checkpoint_and_final_json_is_recomputed(monkeypatch):
    """S6: the final state is recomputed from the records on resume (a crash after the last checkpoint, before
    final.json): no further update, final.json written, then the final validation."""
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    orig = T.Trainer.finalize
    monkeypatch.setattr(T.Trainer, "finalize", lambda self, u, reason: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        T.main(args(sp, out, updates=2))
    assert not os.path.exists(os.path.join(out, "final.json")) and os.path.exists(os.path.join(out, "ckpt", "u00002"))
    monkeypatch.setattr(T.Trainer, "finalize", orig)
    calls = []
    o_up = T.Trainer.one_update
    monkeypatch.setattr(T.Trainer, "one_update", lambda self, u: calls.append(u) or o_up(self, u))
    T.main(args(sp, out, "--resume", updates=2))
    assert calls == []
    fin = json.load(open(os.path.join(out, "final.json")))
    assert fin["final_update"] == 2 and fin["stop_reason"] == "max_updates" and fin["validated"]
    assert no_fail(v17_report(out, sp)), fails(v17_report(out, sp))
