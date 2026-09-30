#!/usr/bin/env python3
"""Server test of the torch path of rl_algos (GRPO first; PPO smoke at the end), SPEC v17. Needs one GPU.

  BASE=/tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/<hash>/ GPU=0 python test_rl_algos_server.py

Checks
  1  new LoRA: r=16, alpha=32, dropout 0.0, q/k/v/o; trainable params fp32, base bf16, requires_grad only on LoRA;
     v17 B1: the LoRA init is seeded (the same seed gives the same init sha)
  12 v17 B4: the start adapter loaded as the frozen "ref" (load_ref): trainable set, optimizer params and policy sha
     unchanged, every ref parameter frozen; a ref forward leaves them unchanged; save() writes "default" only (no ref/,
     the adapter file identical to the one saved before the ref was loaded)
  2  at init KL == 0: policy log-probs == reference (the "ref" adapter) log-probs, k3 == 0
  3  eval-mode and train_nodropout-mode log-probs identical (checkpointing active only in train mode)
  4  GRPO update: max |ratio - 1| at the start of the update <= 1e-4 (TorchLearner asserts it), loss finite
  5  gradients only on the policy LoRA params (and they are non-zero); no grad on base weights or ref
  6  one update changes the policy log-probs and makes KL > 0; the reference does not move
  7  save -> perturb (another update) -> load reproduces adapter sha, optimizer state and RNG state
  8  ids recorded at generation are used exactly (logits slice length == len(gen_ids))
  10 stop credit: sequence advantage 0, stop advantage on ONE token -> that token's log-prob rises; wrong mask refused
  13 v17 S2 prefix advantage: adv_prefix on the tokens before the value (prefix_mask) raises their log-prob; the value
     token (stop_mask, adv_stop 0) gets no advantage; an overlapping prefix / stop mask is refused
  14 v17 §3.4 / S4: epochs 2 x minibatches 4 = 8 optimizer steps with the aux split (each example once per epoch),
     per-step statistics; aux-only update = 1 step (flag aux_only)
  11 auxiliary stop supervision: an aux-only update raises p(target tokens | prompt + prefix)
  15 v17 §1.2: value_nll_loss / the SFT AdamW step raise p(target); p_end_batch == end_prob; fp32 value logits agree
     with the bf16 path; the ref is on --GPU and no other GPU holds memory (fix round 1, D-N3 / B-1)
  9  PPO smoke: value head gets gradients, loss finite (optional algorithm)
"""
import glob
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rl_algos as RA  # noqa: E402

BASE_GLOB = "/tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/"
GPU = int(os.environ.get("GPU", "0"))
MAX_NEW = int(os.environ.get("MAX_NEW", "48"))
SEED = 12345


def log(*a):
    print(*a, flush=True)


def make_samples(planner, n_groups=2, G=2):
    """Real sampled generations recorded exactly like Task2Env (PlannerLM.generate)."""
    system = "You plan the next message of a person searching for research datasets. Reply with JSON."
    users = ["They want climate datasets with daily resolution for Europe.",
             "They want a speech corpus of child language with transcripts."]
    samples, rewards = [], []
    for gi in range(n_groups):
        for g in range(G):
            out = planner.generate(system, users[gi % len(users)], temperature=1.0, top_p=1.0, seed=1000 * gi + g)
            assert len(out["gen_ids"]) > 0
            samples.append({"prompt_ids": out["prompt_ids"], "gen_ids": out["gen_ids"], "temperature": 1.0,
                            "policy_version": 0, "group": gi})
            rewards.append(float(g))                      # distinct rewards -> non-zero advantages
    advs = []
    for gi in range(n_groups):
        rs = [r for s, r in zip(samples, rewards) if s["group"] == gi]
        advs += RA.group_advantages(rs)
    for s, a, r in zip(samples, advs, rewards):
        s["adv"], s["ret"] = a, r
    return samples


def fresh(samples):
    return [{k: v for k, v in s.items() if k in ("prompt_ids", "gen_ids", "temperature", "policy_version",
                                                   "adv", "ret", "group")} for s in samples]


def logps(model, samples, mode="eval", learner=None):
    """Policy log-probs; with learner: the reference ("ref" adapter) log-probs through learner._logp."""
    import torch
    RA._set_mode(model, mode)
    out = []
    with torch.no_grad():
        for s in samples:
            if learner is not None:
                lp = learner._logp(s, grad=False, reference=True)[0]
            else:
                lp, _ = RA.token_logprobs(model, s["prompt_ids"], s["gen_ids"], s["temperature"])
            assert lp.shape[0] == len(s["gen_ids"])                                   # check 8
            out.append(lp.float().cpu())
    model.eval()
    return out


def maxdiff(a, b):
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def share_ref(new, old):
    """A second TorchLearner on the same model: the ref adapter is already loaded (by `old`)."""
    new.has_ref, new.ref_path, new._trainable0 = old.has_ref, old.ref_path, old._trainable0
    return new


def main():
    import torch
    from task2_env import PlannerLM
    BASE = os.environ.get("BASE") or sorted(glob.glob(BASE_GLOB))[-1]
    log("base", BASE, "gpu", GPU)
    torch.cuda.set_device(GPU)                    # fix round 1 (B-1): no context on another GPU
    planner = PlannerLM(BASE, gpu=GPU, dtype="bfloat16", max_new=MAX_NEW)
    model = RA.setup_policy(planner, SEED)
    # ---- 1 LoRA layout and dtypes, seeded init (B1)
    pc = model.peft_config["default"]
    assert (pc.r, pc.lora_alpha, pc.lora_dropout) == (16, 32, 0.0), pc
    assert set(pc.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}, pc.target_modules
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n, _ in trainable), [n for n, _ in trainable if "lora_" not in n][:5]
    assert all(p.dtype == torch.float32 for _, p in trainable)
    assert any(p.dtype == torch.bfloat16 for n, p in model.named_parameters() if not p.requires_grad)
    RA.assert_no_dropout(model)
    init_sha = RA.tensor_sha(RA.lora_state(model))
    torch.manual_seed(SEED)
    gen_a = [torch.empty(8).uniform_() for _ in range(2)]
    torch.manual_seed(SEED)
    gen_b = [torch.empty(8).uniform_() for _ in range(2)]
    assert all(torch.equal(x, y) for x, y in zip(gen_a, gen_b))
    try:
        RA.setup_policy(planner, None)
        raise SystemExit("an unseeded LoRA init was accepted")
    except ValueError:
        pass
    log("1 ok: %d trainable LoRA tensors, fp32; base bf16; init sha %s (seed %d)" % (len(trainable), init_sha[:12], SEED))

    samples = make_samples(planner)
    log("samples:", [(len(s["prompt_ids"]), len(s["gen_ids"])) for s in samples])
    # ---- 12 the ref adapter (v17 B4): the start adapter, frozen
    learner = RA.TorchLearner(model, "grpo", {"minibatches": 2, "epochs": 1}, lr=1e-4, seed=0)
    tmp0 = tempfile.mkdtemp()
    tmp1 = tempfile.mkdtemp()
    try:
        learner.save_adapter(tmp0)
        f0 = open(os.path.join(tmp0, "adapter", "adapter_model.safetensors"), "rb").read()
        names0 = learner.trainable_names()
        opt0 = [id(p) for g in learner.optimizer.param_groups for p in g["params"]]
        learner.load_ref(os.path.join(tmp0, "adapter"))
        assert learner.trainable_names() == names0, "ref loading changed the trainables"
        assert [id(p) for g in learner.optimizer.param_groups for p in g["params"]] == opt0
        assert learner.policy_sha() == init_sha
        refs = [p for n, p in model.named_parameters() if ".ref." in n]
        assert refs and not any(p.requires_grad for p in refs)
        _ = learner._logp(fresh(samples)[0], grad=False, reference=True)
        assert learner.trainable_names() == names0 and not any(p.requires_grad for p in refs)
        try:
            RA.load_ref_adapter(model, os.path.join(tmp0, "adapter"))
            raise SystemExit("a second ref adapter was accepted")
        except AssertionError:
            pass
        learner.save_adapter(tmp1)
        assert not os.path.exists(os.path.join(tmp1, "adapter", "ref")), "the ref adapter was saved"
        assert open(os.path.join(tmp1, "adapter", "adapter_model.safetensors"), "rb").read() == f0, \
            "the saved policy adapter changed after loading the ref"
    finally:
        shutil.rmtree(tmp0, ignore_errors=True)
        shutil.rmtree(tmp1, ignore_errors=True)
    log("12 ok: ref loaded frozen (%d tensors), trainables / optimizer / sha unchanged, save writes default only" % len(refs))
    # ---- 2 KL == 0 at init (policy == ref)
    pol, ref = logps(model, samples), logps(model, samples, learner=learner)
    d = maxdiff(pol, ref)
    k3 = max(float((torch.exp(r - p) - (r - p) - 1).abs().max()) for p, r in zip(pol, ref))
    assert d <= 1e-5 and k3 <= 1e-8, (d, k3)
    log("2 ok: init |logp - ref| max %.3g, k3 max %.3g" % (d, k3))
    # ---- 3 eval vs train_nodropout
    tr = logps(model, samples, mode="train_nodropout")
    d3 = maxdiff(pol, tr)
    log("3 eval vs train_nodropout max |dlogp| = %.3g" % d3)
    assert d3 <= 1e-4, d3

    # ---- 5 gradients only on the policy LoRA (manual backward of one sample)
    RA._set_mode(model, "train_nodropout")
    s = samples[0]
    lp, _ = RA.token_logprobs(model, s["prompt_ids"], s["gen_ids"], 1.0)
    (-(lp.sum()) * 1.0).backward()
    with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
    assert with_grad and all("lora_" in n and ".ref." not in n for n in with_grad), \
        [n for n in with_grad if "lora_" not in n or ".ref." in n][:5]
    nz = sum(float(p.grad.abs().sum()) > 0 for n, p in model.named_parameters() if p.grad is not None and "lora_B" in n)
    assert nz > 0, "no non-zero gradient on lora_B"
    model.zero_grad(set_to_none=True)
    model.eval()
    log("5 ok: %d params with grad, all policy LoRA; %d lora_B with non-zero grad" % (len(with_grad), nz))

    # ---- 4 + 6 GRPO update
    sha0 = learner.policy_sha()
    st = learner.update(fresh(samples), {"lr": 1e-4, "kl_coef": 0.04}, seed=1)
    log("update stats", {k: st[k] for k in ("loss", "kl", "ratio_mean", "clip_frac", "grad_norm", "n_tokens",
                                            "ratio_init_maxdev", "optimizer_steps")})
    assert st["ratio_init_maxdev"] is not None and st["ratio_init_maxdev"] <= 1e-4, st["ratio_init_maxdev"]
    assert math.isfinite(st["loss"]) and math.isfinite(st["grad_norm"]) and st["optimizer_steps"] == 2
    log("4 ok: ratio at update start max |r-1| = %.3g; loss %.4g finite" % (st["ratio_init_maxdev"], st["loss"]))
    pol1, ref1 = logps(model, samples), logps(model, samples, learner=learner)
    d6 = maxdiff(pol, pol1)
    k3_1 = sum(float((torch.exp(r - p) - (r - p) - 1).sum()) for p, r in zip(pol1, ref1))
    assert learner.policy_sha() != sha0 and d6 > 1e-4 and k3_1 > 0, (d6, k3_1)
    assert maxdiff(ref, ref1) <= 1e-5, "reference moved"
    log("6 ok: policy logp moved by %.3g, KL(k3 sum) now %.3g, reference unchanged" % (d6, k3_1))
    # a second update also starts on-policy (ratio re-anchored to the new theta_old)
    st2 = learner.update(fresh(samples), {"lr": 1e-4, "kl_coef": 0.04}, seed=2)
    assert st2["ratio_init_maxdev"] <= 1e-4, st2["ratio_init_maxdev"]
    log("4b ok: second update start max |r-1| = %.3g" % st2["ratio_init_maxdev"])

    # ---- 7 save / perturb / load
    tmp = tempfile.mkdtemp()
    try:
        learner.save(tmp)
        assert not os.path.exists(os.path.join(tmp, "adapter", "ref"))
        torch.save(RA.rng_state(), os.path.join(tmp, "rng.pt"))
        sha_saved = learner.policy_sha()
        opt_saved = {k: [t.clone() if torch.is_tensor(t) else t for t in v.values()]
                     for k, v in learner.optimizer.state_dict()["state"].items()}
        r_saved = torch.rand(4, device="cuda:%d" % GPU)
        learner.update(fresh(samples), {"lr": 1e-4, "kl_coef": 0.04}, seed=3)       # perturb weights + optimizer
        assert learner.policy_sha() != sha_saved
        learner.load(tmp)
        RA.set_rng_state(torch.load(os.path.join(tmp, "rng.pt"), weights_only=False))
        assert learner.policy_sha() == sha_saved, "adapter not restored"
        opt_now = learner.optimizer.state_dict()["state"]
        for k, vals in opt_saved.items():
            for a, b in zip(vals, opt_now[k].values()):
                if torch.is_tensor(a):
                    assert torch.equal(a.to(b.device), b), "optimizer state not restored"
                else:
                    assert a == b
        r_again = torch.rand(4, device="cuda:%d" % GPU)
        assert torch.equal(r_saved, r_again), "CUDA RNG state not restored"
        assert all(p.dtype == torch.float32 for n, p in model.named_parameters() if p.requires_grad)
        log("7 ok: adapter sha, optimizer state and RNG restored")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- 10 stop credit: sequence advantage 0, stop advantage +1 on ONE token -> that token's log-prob rises
    s0 = dict(fresh(samples)[0])
    # the most uncertain generated token: a near-certain JSON token (logp ~ 0) carries no gradient
    lp0 = logps(model, [s0])[0]
    k = int(lp0.argmin())
    assert float(lp0[k]) < -0.05, "no uncertain token in the sample (min logp %.3g)" % float(lp0[k])
    s0.update(adv=0.0, adv_stop=1.0, stop_mask=[1 if i == k else 0 for i in range(len(s0["gen_ids"]))])
    before = logps(model, [s0])[0]
    lr10 = share_ref(RA.TorchLearner(model, "grpo", {"minibatches": 1, "epochs": 1}, lr=1e-4, seed=0), learner)
    lr10.update([dict(s0)], {"lr": 1e-4, "kl_coef": 0.0}, seed=5)
    after = logps(model, [s0])[0]
    dk = float(after[k] - before[k])
    assert dk > 0, "masked token log-prob did not rise (%.3g)" % dk
    bad = dict(s0, stop_mask=[1, 0])
    try:
        lr10.update([bad], {"lr": 1e-4, "kl_coef": 0.0}, seed=6) if len(s0["gen_ids"]) != 2 else None
        raise SystemExit("a stop_mask of the wrong length was accepted")
    except AssertionError:
        pass
    log("10 ok: stop-credit token (t=%d, logp %.3g) log-prob +%.3g; wrong-length mask refused" % (k, float(before[k]), dk))

    # ---- 13 prefix advantage (v17 S2): adv_prefix on the tokens before the value; none on the value token
    s2 = dict(fresh(samples)[2])
    lp2 = logps(model, [s2])[0]
    n2 = len(s2["gen_ids"])
    kv = max(1, n2 // 2)                                   # the "value" token
    sm = [1 if i == kv else 0 for i in range(n2)]
    pm = RA.prefix_mask_of(sm)
    assert pm == [1] * kv + [0] * (n2 - kv)
    s2.update(adv=0.0, adv_prefix=1.0, adv_stop=0.0, prefix_mask=pm, stop_mask=sm)
    tok = RA.token_advantages(s2, n2)
    assert tok[kv] == 0.0 and all(v == 1.0 for v in tok[:kv]) and all(v == 0.0 for v in tok[kv:])
    lr13 = share_ref(RA.TorchLearner(model, "grpo", {"minibatches": 1, "epochs": 1}, lr=1e-4, seed=0), learner)
    lr13.update([dict(s2)], {"lr": 1e-4, "kl_coef": 0.0}, seed=8)
    lp2b = logps(model, [s2])[0]
    rise = float(lp2b[:kv].sum() - lp2[:kv].sum())
    assert rise > 0, "prefix log-prob did not rise (%.3g)" % rise
    try:
        lr13.update([dict(s2, prefix_mask=[1] * n2)], {"lr": 1e-4, "kl_coef": 0.0}, seed=9)
        raise SystemExit("an overlapping prefix / stop mask was accepted")
    except AssertionError:
        pass
    log("13 ok: prefix tokens' summed log-prob +%.3g; value token carries no advantage; overlap refused" % rise)

    # ---- 14 epochs 2 x minibatches 4 with the aux split = 8 steps; aux-only = 1 step
    four = fresh(make_samples(planner, n_groups=2, G=2))
    auxs = []
    for s_ in four[:3]:
        c = max(1, len(s_["gen_ids"]) // 2)
        auxs.append({"prompt_ids": s_["prompt_ids"], "prefix_ids": s_["gen_ids"][:c], "target_ids": s_["gen_ids"][c:c + 1],
                     "want_end": True, "weight": 0.5, "gen_len": len(s_["gen_ids"])})
    lr14 = share_ref(RA.TorchLearner(model, "grpo", {}, lr=1e-5, seed=0), learner)
    assert (lr14.acfg["epochs"], lr14.acfg["minibatches"]) == (2, 4)
    st14 = lr14.update(four, {"lr": 1e-5, "kl_coef": 0.01}, seed=10, aux=auxs, aux_orders=[[2, 0, 1], [1, 2, 0]])
    assert st14["optimizer_steps"] == 8 and len(st14["steps"]) == 8, st14["optimizer_steps"]
    assert [x["n_aux"] for x in st14["steps"]] == [1, 1, 1, 0, 1, 1, 1, 0]
    assert st14["aux_n"] == 3 and st14["rl_grad_norm_max"] >= st14["rl_grad_norm"] > 0 and st14["kl_step_max"] is not None
    st14b = lr14.update([], {"lr": 1e-5, "kl_coef": 0.01}, seed=11, aux=auxs)
    assert st14b["optimizer_steps"] == 1 and st14b["aux_only"] is True
    log("14 ok: 8 steps (aux %s), rl_grad_norm mean %.3g max %.3g, aux_grad_norm %.3g; aux-only = 1 step" % (
        [x["n_aux"] for x in st14["steps"]], st14["rl_grad_norm"], st14["rl_grad_norm_max"], st14["aux_grad_norm"]))

    # ---- 11 auxiliary stop supervision: an aux-only update raises p(target tokens | prompt + prefix)
    s1 = fresh(samples)[1]
    lp1 = logps(model, [s1])[0]
    cut = int(lp1.argmin())                      # an uncertain token (a certain one has no gradient)
    assert float(lp1[cut]) < -0.05, float(lp1[cut])
    x = {"prompt_ids": s1["prompt_ids"], "prefix_ids": s1["gen_ids"][:cut], "target_ids": s1["gen_ids"][cut:cut + 1],
         "want_end": True, "weight": 1.0, "gen_len": len(s1["gen_ids"])}
    lp_b = RA.token_logprobs(model, list(x["prompt_ids"]) + list(x["prefix_ids"]), x["target_ids"], 1.0)[0].sum().item()
    lr11 = share_ref(RA.TorchLearner(model, "grpo", {}, lr=1e-4, seed=0), learner)
    st11 = lr11.update([], {"lr": 1e-4, "kl_coef": 0.0}, seed=7, aux=[x])
    model.eval()
    with torch.no_grad():
        lp_a = RA.token_logprobs(model, list(x["prompt_ids"]) + list(x["prefix_ids"]), x["target_ids"], 1.0)[0].sum().item()
    assert st11["aux_n"] == 1 and lp_a > lp_b, (lp_b, lp_a, st11)
    log("11 ok: aux-only update raised target logp %.4g -> %.4g" % (lp_b, lp_a))

    # ---- 15 SFT step (value_nll_loss, own AdamW) and p_end_batch
    y = dict(x, weight=1.0)
    with torch.no_grad():
        q_b = RA.token_logprobs(model, list(y["prompt_ids"]) + list(y["prefix_ids"]), y["target_ids"], 1.0)[0].sum().item()
    learner.sft_begin(5e-5)
    assert learner.sft_opt is not learner.optimizer
    sst = learner.sft_step([y])
    learner.sft_end()
    with torch.no_grad():
        q_a = RA.token_logprobs(model, list(y["prompt_ids"]) + list(y["prefix_ids"]), y["target_ids"], 1.0)[0].sum().item()
    assert math.isfinite(sst["loss"]) and q_a > q_b, (q_b, q_a)
    items = [{"prompt_ids": y["prompt_ids"], "prefix_ids": y["prefix_ids"], "target_true": y["target_ids"],
              "target_false": s1["gen_ids"][cut + 1:cut + 2] or [0]}]
    assert abs(learner.p_end_batch(items)[0] - learner.end_prob(items[0])) < 1e-9
    # fix round 1 (D-N3): end_prob's value logits are fp32 from the lm_head input; they agree with the bf16 path
    with torch.no_grad():
        ctx_ = list(y["prompt_ids"]) + list(y["prefix_ids"])
        d32 = float((RA.token_logprobs_fp32(model, ctx_, y["target_ids"])
                     - RA.token_logprobs(model, ctx_, y["target_ids"], 1.0)[0]).abs().max())
    assert d32 < 0.05, d32
    # fix round 1 (B-1): the ref adapter lives on the policy's GPU, nothing on any other GPU
    assert {str(p.device) for n, p in model.named_parameters() if ".ref." in n} == {"cuda:%d" % GPU}
    assert all(torch.cuda.memory_allocated(i) == 0 for i in range(torch.cuda.device_count()) if i != GPU)
    log("15 ok: SFT step raised target logp %.4g -> %.4g; p_end_batch == end_prob" % (q_b, q_a))

    # ---- 9 PPO smoke (optional algorithm)
    ppo = share_ref(RA.TorchLearner(model, "ppo", {}, lr=1e-5, seed=0), learner)
    stp = ppo.update(fresh(samples), {"lr": 1e-5, "kl_coef": 0.0}, seed=4)
    assert math.isfinite(stp["loss"]) and stp["value_mse"] is not None and stp["ratio_init_maxdev"] <= 1e-4
    log("9 ok: PPO loss %.4g value_mse %.4g" % (stp["loss"], stp["value_mse"]))
    log("ALL SERVER CHECKS PASSED")


if __name__ == "__main__":
    main()
