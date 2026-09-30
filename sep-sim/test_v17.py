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
    n_adapters = len([d for d in os.listdir(os.path.join(out, "ckpt")) if os.path.isdir(os.path.join(out, "ckpt", d, "adapter"))])
    assert n_adapters >= 3 + 4 and rep.checks["rl.ref_adapter"]["n"] >= n_adapters       # C-N6: u0..u2 + sft_e0..e3
    os.makedirs(os.path.join(out, "ckpt", "u00001", "adapter", "ref"))            # a checkpoint with the ref inside
    assert v17_report(out, sp).checks["rl.ref_adapter"]["fail"] == 1


class _P:
    def __init__(self, g, device="cuda:1"):
        self.requires_grad, self.device = g, device


class _MockPeft:
    """Just enough of a PeftModel for load_ref_adapter / save_adapter (no torch). The policy lives on cuda:1; a
    load_adapter without torch_device would put the ref on PEFT's default "cuda" = cuda:0 (fix round 1, B-1)."""

    def __init__(self, with_inference_mode=True):
        self.params = {"m.lora_A.default.weight": _P(True), "m.lora_B.default.weight": _P(True), "m.base": _P(False)}
        self.peft_config = {"default": object()}
        self.active, self.saved, self.calls = "default", [], []
        if not with_inference_mode:
            self.set_adapter = self._set_adapter_old

    def named_parameters(self):
        return list(self.params.items())

    def parameters(self):
        return iter(list(self.params.values()))

    def load_adapter(self, path, adapter_name, is_trainable, torch_device=None):
        assert is_trainable is False
        self.calls.append(("load_adapter", torch_device))
        self.peft_config[adapter_name] = object()
        self.params["m.lora_A.%s.weight" % adapter_name] = _P(False, torch_device or "cuda:0")
        self.params["m.lora_B.%s.weight" % adapter_name] = _P(False, torch_device or "cuda:0")

    def set_adapter(self, name, inference_mode=False):                          # PEFT: the active adapter is trainable
        self.calls.append(("set_adapter", name, inference_mode))
        self.active = name
        for n, p in self.params.items():
            if ".lora_" in n:
                p.requires_grad = ((".%s." % name) in n) and not inference_mode

    def _set_adapter_old(self, name):
        self.calls.append(("set_adapter", name, None))
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
    # B-1: the ref is loaded onto the policy's device (never PEFT's default cuda:0)
    assert ("load_adapter", "cuda:1") in m.calls
    assert {p.device for n, p in m.params.items() if ".ref." in n} == {"cuda:1"}
    # B-N5: the ref forward switches with inference_mode=True when PEFT supports it; back to "default" plainly
    RA.set_active_adapter(m, RA.REF_ADAPTER)
    assert m.calls[-1] == ("set_adapter", "ref", True) and not any(p.requires_grad for p in m.params.values())
    RA.set_active_adapter(m, "default")
    assert m.calls[-1] == ("set_adapter", "default", False)
    assert [n for n, p in m.params.items() if p.requires_grad] == ["m.lora_A.default.weight", "m.lora_B.default.weight"]
    old = _MockPeft(with_inference_mode=False)
    RA.set_active_adapter(old, RA.REF_ADAPTER)
    assert old.calls[-1] == ("set_adapter", "ref", None)
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
    assert cnt == {"groups_skipped_zero_std": 1, "n_groups_lt2": 1, "n_groups_all_invalid": 1, "n_groups_used": 1,
                   "n_groups_used_final": 1, "n_groups_used_nonfinal": 0, "n_groups_used_adv_gt_1e3": 1}    # D-N1
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


def test_env_describe_speaker_fields(monkeypatch):
    """S12 (fix round 1, C-N5): describe() of a stub env reports the Speaker's real path, class MRO and endconv."""
    import types
    import task2_env as TE
    monkeypatch.setenv("SEPSIM_ACT_PRIOR", "nostopclobber")

    class Speaker:
        pass

    class DittoSpeaker(Speaker):
        def __init__(self):
            self.path, self.endconv, self.max_new = "/tmp2/mzjiang_usersim/models/Ditto-8B", None, 512
    env = types.SimpleNamespace(
        arm="pend", planner=types.SimpleNamespace(path="Q4", adapter=None, budget=100, max_new=1536),
        system="sys", judge=None, r0=types.SimpleNamespace(model="gpt-oss-120b", gpt5_dialect=False), r0_effort="low",
        ledger_judge=types.SimpleNamespace(model="gpt-oss-120b", gpt5=False, floor=4000), judge_effort="low",
        t1_sampling=None, ip=True, fewshot=None, selector="borda", speaker=DittoSpeaker(), task1_only=False,
        planner_batcher=None, speaker_batcher=None)
    d = TE.Task2Env.describe(env)
    assert d["speaker_path"].endswith("Ditto-8B") and d["speaker_endconv_is_none"] is True
    assert d["speaker_class"][0].endswith(".DittoSpeaker") and d["speaker_class"][-1] == "builtins.object"
    env.speaker.endconv = 42                                                       # a Speaker with an end token
    assert TE.Task2Env.describe(env)["speaker_endconv_is_none"] is False
    del env.speaker.endconv
    assert TE.Task2Env.describe(env)["speaker_endconv_is_none"] is False


# ------------------------------------------------------------------ §6 threshold control
def test_threshold_control_default_is_v17_base():
    import inspect
    src = inspect.getsource(TC.main)                                           # the parser is built inside main
    assert 'ap.add_argument("--test-update", default="base"' in src                # fix round 3 (A N1)


def test_short_conversations_note_when_none():
    sp, out = fresh(updates=1)
    rep = v17_report(out, sp)
    assert rep.status("rl.sft_short_conversations") == "PASS" or rep.checks["rl.sft_short_conversations"]["notes"] == \
        ["0 train_all conversations with n_by_conv < 2"]
    assert not rep.checks["rl.sft_short_conversations"]["warn"]


def test_threshold_control(tmp_path):
    # train: base P_end systematically too low (true rate at the final points 0.8, the policy says 0.4)
    train = [{"conversation_id": "c01", "real_final": True, "p_end": 0.4, "valid": True}] * 8 \
        + [{"conversation_id": "c02", "real_final": False, "p_end": 0.1, "valid": True}] * 8 \
        + [{"conversation_id": "c03", "real_final": True, "p_end": 0.0, "valid": False}]
    sp = str(tmp_path / "splits.json")
    json.dump({"folds": [{"fold": 0, "train": ["c01", "c02"], "train_all": ["c01", "c02", "c03"], "validation": [],
                          "test": ["t0"], "test_all": ["t0", "t1"], "forbidden_for_training": ["t0", "t1"]}]}, open(sp, "w"))
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
    res = TC.main(["--train", str(ptrain), "--test", str(ptest), "--test-update", "0", "--json-out", str(tmp_path / "o.json"),
                   "--splits", sp, "--fold", "0"])
    assert res["base_at_0.5"]["term_f1"] == 0.0 and res["threshold_control"]["term_f1"] == 1.0   # the shift ends at n
    assert res["threshold_control"]["n_invalid"] == 1 and res["threshold_control"]["nll"] < res["base_at_0.5"]["nll"]
    shifted = TC.shifted(rows[1]["end_probs"], b)
    assert shifted[0]["p_end"] == 0.0                                          # an unscored point is not shifted
    assert json.load(open(tmp_path / "o.json"))["offset_b"] == b
    # C-N10: a train point outside train_all, or a test conversation outside test_all, is refused
    bad = tmp_path / "bad_train.jsonl"
    bad.write_text("".join(json.dumps(dict(r, conversation_id="t0")) + "\n" for r in train))
    with pytest.raises(SystemExit):
        TC.main(["--train", str(bad), "--test", str(ptest), "--splits", sp, "--fold", "0"])
    s2 = json.load(open(sp))
    s2["folds"][0]["test_all"] = ["t0"]
    json.dump(s2, open(sp, "w"))
    with pytest.raises(SystemExit):
        TC.main(["--train", str(ptrain), "--test", str(ptest), "--splits", sp, "--fold", "0"])


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


# ================================================================== fix round 1 (audits A-D of 328a2d4, 2026-09-30)
def test_bs1_crash_before_sft_meta_row_is_repaired(monkeypatch):
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    orig = T.append_jsonl

    def crash(path, row):
        if path.endswith("run_meta.jsonl") and row.get("kind") == "sft":
            raise RuntimeError("crash after u0, before the sft meta row")
        return orig(path, row)
    monkeypatch.setattr(T, "append_jsonl", crash)
    with pytest.raises(RuntimeError):
        T.main(args(sp, out, updates=2))
    monkeypatch.setattr(T, "append_jsonl", orig)
    assert os.path.exists(os.path.join(out, "ckpt", "LATEST.json"))
    assert v17_report(out, sp).checks["rl.ref_adapter"]["fail"] >= 1          # no sft row naming the ref yet
    T.main(args(sp, out, "--resume", updates=2))
    rows = [m for m in T.read_jsonl(os.path.join(out, "run_meta.jsonl")) if m["kind"] == "sft"]
    assert len(rows) == 1 and rows[0]["repaired_on_resume"] is True
    assert no_fail(v17_report(out, sp)), fails(v17_report(out, sp))


def test_a2_crash_between_u0_and_latest_resumes(monkeypatch):
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    orig = T.Trainer.save_checkpoint

    def crash(self, u, row):
        orig(self, u, row)
        if u == 0:
            os.remove(os.path.join(self.ckpt_root, "LATEST.json"))          # u0 renamed, LATEST not yet written
            raise RuntimeError("crash between the u0 rename and LATEST")
    monkeypatch.setattr(T.Trainer, "save_checkpoint", crash)
    with pytest.raises(RuntimeError):
        T.main(args(sp, out, updates=2))
    monkeypatch.setattr(T.Trainer, "save_checkpoint", orig)
    assert os.path.exists(os.path.join(out, "ckpt", "u00000")) and not os.path.exists(os.path.join(out, "ckpt", "LATEST.json"))
    T.main(args(sp, out, "--resume", updates=2))                              # the ref is not loaded at build: SFT completes
    assert json.load(open(os.path.join(out, "final.json")))["validated"]
    assert no_fail(v17_report(out, sp)), fails(v17_report(out, sp))


def test_a3_stop_row_before_final_json_and_repair(monkeypatch):
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    orig = T.write_json_atomic

    def crash(path, obj):
        if path.endswith("final.json"):
            raise RuntimeError("crash before final.json")
        return orig(path, obj)
    monkeypatch.setattr(T, "write_json_atomic", crash)
    with pytest.raises(RuntimeError):
        T.main(args(sp, out, updates=2))
    monkeypatch.setattr(T, "write_json_atomic", orig)
    assert [m["kind"] for m in T.read_jsonl(os.path.join(out, "run_meta.jsonl"))][-1] == "stop"   # stop row first
    T.main(args(sp, out, "--resume", updates=2))
    assert sum(m["kind"] == "stop" for m in T.read_jsonl(os.path.join(out, "run_meta.jsonl"))) == 1
    assert no_fail(v17_report(out, sp)), fails(v17_report(out, sp))
    # final.json exists (not yet validated) without a stop row: the resume adds it
    d2 = tempfile.mkdtemp()
    sp2, out2 = make_splits(d2), os.path.join(d2, "run")
    o_val = T.Trainer.validate
    monkeypatch.setattr(T.Trainer, "validate", lambda self, u, task2: (_ for _ in ()).throw(RuntimeError("x"))
                        if (u == 2 and task2) else o_val(self, u, task2))
    with pytest.raises(RuntimeError):
        T.main(args(sp2, out2, updates=2))
    monkeypatch.setattr(T.Trainer, "validate", o_val)
    mp = os.path.join(out2, "run_meta.jsonl")
    _rewrite(mp, [m for m in T.read_jsonl(mp) if m["kind"] != "stop"])
    T.main(args(sp2, out2, "--resume", updates=2))
    assert sum(m["kind"] == "stop" for m in T.read_jsonl(mp)) == 1
    assert no_fail(v17_report(out2, sp2)), fails(v17_report(out2, sp2))


def test_a4_manifest_and_new_stat_fields():
    sp, out = fresh(updates=2)
    man = json.load(open(os.path.join(out, "ckpt", "u00001", "rl_manifest.json")))
    assert man["sft_args"] == {"sft_lr": 5e-5, "sft_epochs_max": 3, "sft_samples_per_point": 2}
    u = T.read_jsonl(os.path.join(out, "updates.jsonl"))[0]
    th = u["train_aggregate"]["task1_train"]
    assert th["n_groups_used"] == th["n_groups_used_final"] + th["n_groups_used_nonfinal"] >= th["n_groups_used_adv_gt_1e3"]
    assert th["n_value_roundtrip_fail"] == 0                                   # D-N7 (fake plans always round-trip)
    assert "value_roundtrip_fail" in json.load(open(os.path.join(out, "sft_examples_meta.json")))["counts"]
    assert "aux_p_correct_before_final" in u["train_aggregate"]["aux_stats"]   # D-N6
    assert "aux_p_correct_end" not in json.dumps(u)
    s = [v for v in T.read_jsonl(os.path.join(out, "validation.jsonl")) if v["kind"] == "summary" and v["task2"]][0]
    ts = s["turn_stats"]                                                        # D-N5: human turns capped at t_max
    assert ts["human_turns_capped_at"] == 10 and ts["human_turns_mean"] <= ts["human_turns_mean_uncapped"]


def test_dn5_turn_stats_capped():
    eps = [{"human_turns": 13, "emitted_user_turns": 10, "coverage": 1.0, "end_kind": "t_max"},
           {"human_turns": 3, "emitted_user_turns": 4, "coverage": 0.5, "end_kind": "planner_end"}]
    ts = T.turn_stats_of(eps, 10)
    assert ts["human_turns_mean"] == 6.5 and ts["human_turns_mean_uncapped"] == 8.0 and ts["abs_diff_mean"] == 0.5
    assert ts["turn_w1"] == T.turn_w1([10, 4], [10, 3])


def test_a5_no_v16_aux_field_in_task1_samples():
    tr_env = T.FakeEnv(T.FakeLearner("grpo", {}, 1e-5))
    out = tr_env.task1_sample("c01", 3, "p", True, 4, 1.0, 1.0, 7)
    assert all("aux" not in x for x in out) and all("value_roundtrip_ok" in x for x in out)
    import task2_env as TE
    import inspect
    assert 'out[0]["aux"]' not in inspect.getsource(TE.Task2Env.task1_sample)
    assert not hasattr(RA, "ref_param_names")


@pytest.mark.parametrize("what", ["steps_len", "n_aux", "ag_kl", "final_drift", "sft_seed", "brier_mean", "kl_q_ph",
                                  "n_real", "summary_sha", "no_final_t2", "not_validated", "kl_q_ph_missing",
                                  "p_h_missing"])
def test_verify_round1_catches(what):
    sp, out = fresh(updates=2)
    assert no_fail(v17_report(out, sp)), fails(v17_report(out, sp))
    pu, pt = os.path.join(out, "updates.jsonl"), os.path.join(out, "rollouts_task1.jsonl")
    ups, t1 = T.read_jsonl(pu), T.read_jsonl(pt)
    fp, pv = os.path.join(out, "final.json"), os.path.join(out, "validation.jsonl")
    pm = os.path.join(out, "sft_examples_meta.json")
    check = {"steps_len": "rl.v17_steps", "n_aux": "rl.v17_steps", "ag_kl": "rl.v17_update", "final_drift": "rl.v17_stop",
             "sft_seed": "rl.sft_data", "brier_mean": "rl.v17_task1", "kl_q_ph": "rl.v17_drift", "n_real": "rl.v17_task1",
             "summary_sha": "rl.v17_validation", "no_final_t2": "rl.v17_validation", "not_validated": "rl.v17_stop",
             "kl_q_ph_missing": "rl.v17_drift", "p_h_missing": "rl.v17_drift"}[what]
    if what == "steps_len":
        ups[0]["learner_stats"]["steps"] = ups[0]["learner_stats"]["steps"][:-1]
        _rewrite(pu, ups)
    elif what == "n_aux":
        ups[0]["learner_stats"]["steps"][0]["n_aux"] += 1
        _rewrite(pu, ups)
    elif what == "ag_kl":
        ups[1]["train_aggregate"]["kl_coef"] = 0.04
        _rewrite(pu, ups)
    elif what == "final_drift":
        f = json.load(open(fp))
        f["drift_stats"]["1"] += 0.5
        json.dump(f, open(fp, "w"))
    elif what == "sft_seed":
        p = os.path.join(out, "sft_examples.jsonl")
        ex = T.read_jsonl(p)
        ex[0]["seed"] += 1
        _rewrite(p, ex)
        m = json.load(open(pm))
        m["examples_sha256"] = T.sha_file(p)                                   # even with a consistent sha
        json.dump(m, open(pm, "w"))
    elif what == "brier_mean":
        ups[0]["task1_stats"]["brier_mean"] += 0.01
        _rewrite(pu, ups)
    elif what == "kl_q_ph":
        ups[0]["train_aggregate"]["kl_q_ph"] += 0.01
        _rewrite(pu, ups)
    elif what == "n_real":
        for r in t1:
            if r["update"] == 1:
                r["n_real"] += 1
                break
        _rewrite(pt, t1)
    elif what == "kl_q_ph_missing":                                            # fix round 2: FAIL, not skip
        del ups[0]["train_aggregate"]["kl_q_ph"]
        _rewrite(pu, ups)
    elif what == "p_h_missing":
        mp = os.path.join(out, "run_meta.jsonl")
        meta = T.read_jsonl(mp)
        for m in meta:
            del m["config"]["p_h"]
        _rewrite(mp, meta)
    elif what == "summary_sha":
        val = T.read_jsonl(pv)
        next(v for v in val if v["kind"] == "summary" and v["update"] == 1)["policy_sha"] = "0" * 64
        _rewrite(pv, val)
    elif what == "no_final_t2":
        val = T.read_jsonl(pv)
        _rewrite(pv, [v for v in val if not (v["kind"] == "summary" and v["update"] == 2 and v["task2"])])
    elif what == "not_validated":
        f = json.load(open(fp))
        f["validated"] = False
        json.dump(f, open(fp, "w"))
    rep = v17_report(out, sp)
    assert rep.checks[check]["fail"] >= 1, (what, rep.checks[check])


def test_bs3_verify_requires_kl_from_u2_on_real_runs():
    """B-S3: on a real (non-dry) run the logged KL to the ref must be > 0 from u2 on (dry runs log 0)."""
    sp, out = fresh(updates=2)
    mp, pu = os.path.join(out, "run_meta.jsonl"), os.path.join(out, "updates.jsonl")
    meta = T.read_jsonl(mp)
    for m in meta:
        m["config"]["args"].update(dry_run=False, planner_backend="hf")       # pretend real (no vLLM adapter checks)
    _rewrite(mp, meta)
    ups = T.read_jsonl(pu)
    ups[1]["learner_stats"]["kl"] = 1e-4
    _rewrite(pu, ups)
    assert v17_report(out, sp).checks["rl.v17_update"]["fail"] == 0
    ups[1]["learner_stats"]["kl"] = 0.0
    _rewrite(pu, ups)
    assert v17_report(out, sp).checks["rl.v17_update"]["fail"] == 1


def test_a7_eval_gates_test_seeds_and_final_sha(tmp_path):
    import eval_test_rl as E
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    spec = ["--dry-run", "--fold", "0", "--splits", sp, "--out", out, "--scenarios-per-update", "3"]
    T.main(spec)                                                               # spec values, no --ablation
    fin = json.load(open(os.path.join(out, "final.json")))
    ups = ["--test-updates", "0", str(fin["final_update"])]
    with pytest.raises(SystemExit) as e:
        E.main(["--final", "--test-seeds", "0", "1"] + ups + spec)
    assert "0..7" in str(e.value)
    f2 = dict(fin, policy_sha="0" * 64)
    json.dump(f2, open(os.path.join(out, "final.json"), "w"))
    with pytest.raises(SystemExit) as e:
        E.main(["--final", "--test-seeds"] + [str(i) for i in range(8)] + ups + spec)
    assert "policy sha" in str(e.value)
    assert not os.path.exists(os.path.join(out, "test.jsonl"))


# ================================================================== fix round 2 (audits A, B of 9768cc6)
def _t1_stats_of(rows):
    """The trainer's Task 1 statistics (one_update) of these rows, for rewriting a tampered record consistently."""
    allx = [x for r in rows for x in r["samples"]]
    sc = [x["reward"] for x in allx if not x.get("dropped")]
    pf = [x["p_end"] for r in rows if r["real_final"] for x in r["samples"] if x.get("p_end") is not None]
    pm = [x["p_end"] for r in rows if not r["real_final"] for x in r["samples"] if x.get("p_end") is not None]
    stds = [RA.pstd([x["reward"] for x in r["samples"] if not x.get("dropped")]) for r in rows
            if sum(1 for x in r["samples"] if not x.get("dropped")) >= 2]
    return {"brier_mean": (sum(sc) / len(sc)) if sc else None, "p_end_final_mean": (sum(pf) / len(pf)) if pf else None,
            "p_end_nonfinal_mean": (sum(pm) / len(pm)) if pm else None,
            "n_dropped_mask": sum(1 for x in allx if x.get("dropped")),
            "n_invalid": sum(1 for x in allx if x.get("status") == "invalid"),
            "reward_std_mean": (sum(stds) / len(stds)) if stds else None}


def _drop_task1_point(out, u, skipped):
    """Remove update u's Task 1 group at t = 2 of a conversation with n >= 3 CONSISTENTLY (rows, groups, advantage
    records, skip counters, recomputed statistics, the aux example and its two per-epoch uses); with skipped=True the
    point is recorded as skipped for a capped message instead. -> (cid, t)."""
    pu, pt = os.path.join(out, "updates.jsonl"), os.path.join(out, "rollouts_task1.jsonl")
    ups, t1 = T.read_jsonl(pu), T.read_jsonl(pt)
    row = next(r for r in ups if r["update"] == u)
    ts, st = row["task1_stats"], row["learner_stats"]
    cid, t = next([c, t_] for c, t_ in ts["groups"] if t_ == 2 and any(r["conversation_id"] == c and r["n_real"] >= 3
                                                                        for r in t1 if r["update"] == u))
    ts["groups"] = [g for g in ts["groups"] if g != [cid, t]]
    rec = next(a_ for a_ in ts["advantages"] if [a_[0], a_[1]] == [cid, t])
    ts["advantages"] = [a_ for a_ in ts["advantages"] if a_ is not rec]
    key = {"lt2": "n_groups_lt2", "all_invalid": "n_groups_all_invalid", "zero_std": "groups_skipped_zero_std"}.get(rec[2])
    if key:
        ts[key] -= 1
    if [cid, t] in ts["aux_points"]:
        ts["aux_points"] = [p for p in ts["aux_points"] if p != [cid, t]]
        st["aux_n"] -= 1
        half = len(st["steps"]) // 2
        for part in (st["steps"][:half], st["steps"][half:]):
            next(x for x in part if x["n_aux"] > 0)["n_aux"] -= 1
    t1 = [r for r in t1 if not (r["update"] == u and r["conversation_id"] == cid and r["t"] == t)]
    ts.update(_t1_stats_of([r for r in t1 if r["update"] == u and [r["conversation_id"], r["t"]] in ts["groups"]]))
    if skipped:
        ts["skipped_capped"] = sorted(ts["skipped_capped"] + [[cid, t]])
    _rewrite(pu, ups)
    _rewrite(pt, t1)
    return cid, t


def _only_failure(rep, name, text):
    bad = fails(rep)
    assert set(bad) == {name} and bad[name]["fail"] == 1 and text in bad[name]["examples"][0], bad


def test_verify_task1_missing_point_fails_only_completeness():
    sp, out = fresh(updates=2)
    _drop_task1_point(out, 1, skipped=False)
    _only_failure(v17_report(out, sp), "rl.v17_task1", "decision points")


def test_verify_task1_skip_not_suffix_fails_only_suffix():
    sp, out = fresh(updates=2)
    _drop_task1_point(out, 1, skipped=True)
    _only_failure(v17_report(out, sp), "rl.v17_task1", "do not form a suffix")


def test_verify_sft_skip_not_suffix_fails_only_suffix():
    sp, out = fresh(updates=2)
    pm, pe, pb = [os.path.join(out, f) for f in ("sft_examples_meta.json", "sft_examples.jsonl", "base_pend_train.jsonl")]
    m = json.load(open(pm))
    p0 = next(p for p in m["points"] if p[2] >= 3 and p[1] == 2 and p[3] == "ok")
    cid = p0[0]
    p0[3], p0[4] = "skipped_capped_history", 0
    ex = [e for e in T.read_jsonl(pe) if not (e["conversation_id"] == cid and e["t"] == 2)]
    _rewrite(pe, ex)
    _rewrite(pb, [b for b in T.read_jsonl(pb) if not (b["conversation_id"] == cid and b["t"] == 2)])
    m["counts"]["n_examples"] = len(ex)
    m["counts"]["n_points"] = sum(1 for p in m["points"] if p[3] == "ok")
    m["examples_sha256"], m["base_pend_sha256"] = T.sha_file(pe), T.sha_file(pb)
    json.dump(m, open(pm, "w"))
    _only_failure(v17_report(out, sp), "rl.sft_data", "do not form a suffix")


def test_one_message_conversation(monkeypatch):
    """A R2-1: a train_all conversation with a single message has no decision point: no SFT point, no Task 1 row;
    n_by_conv records its n = 1 and verify accepts the gap only because n < 2."""
    orig = T.FakeEnv.human_turns
    monkeypatch.setattr(T.FakeEnv, "human_turns", lambda self, cid: 1 if cid == "x_noshard" else orig(self, cid))
    d = tempfile.mkdtemp()
    sp, out = make_splits(d), os.path.join(d, "run")
    T.main(args(sp, out, "--task1-convs", "15", updates=2))                  # every train_all conversation in Task 1
    m = json.load(open(os.path.join(out, "sft_examples_meta.json")))
    assert m["n_by_conv"]["x_noshard"] == 1 and not [p for p in m["points"] if p[0] == "x_noshard"]
    assert sorted(m["n_by_conv"]) == sorted(json.load(open(sp))["folds"][0]["train_all"])
    ups = T.read_jsonl(os.path.join(out, "updates.jsonl"))
    assert all("x_noshard" in u["task1_stats"]["convs"] for u in ups)
    rep = v17_report(out, sp)                                                  # fix round 3 (C N1): listed, not failed
    assert rep.status("rl.sft_short_conversations") == "WARN" and "x_noshard" in rep.checks["rl.sft_short_conversations"]["notes"][0]
    assert not [r for r in T.read_jsonl(os.path.join(out, "rollouts_task1.jsonl")) if r["conversation_id"] == "x_noshard"]
    assert no_fail(v17_report(out, sp)), fails(v17_report(out, sp))
    # a conversation without points whose recorded n is >= 2 FAILs
    pm = os.path.join(out, "sft_examples_meta.json")
    m["n_by_conv"]["x_noshard"] = 3
    json.dump(m, open(pm, "w"))
    rep = v17_report(out, sp)
    assert rep.checks["rl.sft_data"]["fail"] >= 1 and rep.checks["rl.v17_task1"]["fail"] >= 1
    # n_by_conv that does not list exactly train_all FAILs
    del m["n_by_conv"]["x_noshard"]
    json.dump(m, open(pm, "w"))
    assert v17_report(out, sp).checks["rl.sft_data"]["fail"] >= 1
