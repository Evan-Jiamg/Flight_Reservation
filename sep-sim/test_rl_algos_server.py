#!/usr/bin/env python3
"""Server test of the torch path of rl_algos (GRPO first; PPO smoke at the end). Needs one GPU.

  BASE=/tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/<hash>/ GPU=0 python test_rl_algos_server.py

Checks
  1  new LoRA: r=16, alpha=32, dropout 0.0, q/k/v/o; trainable params fp32, base bf16, requires_grad only on LoRA
  2  at init KL == 0: policy log-probs == reference (disable_adapter) log-probs, k3 == 0
  3  eval-mode and train_nodropout-mode log-probs identical (checkpointing active only in train mode)
  4  GRPO update: max |ratio - 1| at the start of the update <= 1e-4 (TorchLearner asserts it), loss finite
  5  gradients only on LoRA params (and they are non-zero); no grad on base weights
  6  one update changes the policy log-probs and makes KL > 0
  7  save -> perturb (another update) -> load reproduces adapter sha, optimizer state and RNG state
  8  ids recorded at generation are used exactly (logits slice length == len(gen_ids))
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


def logps(model, samples, mode="eval", reference=False):
    import torch
    RA._set_mode(model, mode)
    out = []
    with torch.no_grad():
        for s in samples:
            if reference:
                with model.disable_adapter():
                    lp, _ = RA.token_logprobs(model, s["prompt_ids"], s["gen_ids"], s["temperature"])
            else:
                lp, _ = RA.token_logprobs(model, s["prompt_ids"], s["gen_ids"], s["temperature"])
            assert lp.shape[0] == len(s["gen_ids"])                                   # check 8
            out.append(lp.float().cpu())
    model.eval()
    return out


def maxdiff(a, b):
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def main():
    import torch
    from task2_env import PlannerLM
    BASE = os.environ.get("BASE") or sorted(glob.glob(BASE_GLOB))[-1]
    log("base", BASE, "gpu", GPU)
    planner = PlannerLM(BASE, gpu=GPU, dtype="bfloat16", max_new=MAX_NEW)
    model = RA.setup_policy(planner)
    # ---- 1 LoRA layout and dtypes
    pc = model.peft_config["default"]
    assert (pc.r, pc.lora_alpha, pc.lora_dropout) == (16, 32, 0.0), pc
    assert set(pc.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}, pc.target_modules
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n, _ in trainable), [n for n, _ in trainable if "lora_" not in n][:5]
    assert all(p.dtype == torch.float32 for _, p in trainable)
    assert any(p.dtype == torch.bfloat16 for n, p in model.named_parameters() if not p.requires_grad)
    RA.assert_no_dropout(model)
    log("1 ok: %d trainable LoRA tensors, fp32; base bf16" % len(trainable))

    samples = make_samples(planner)
    log("samples:", [(len(s["prompt_ids"]), len(s["gen_ids"])) for s in samples])
    # ---- 2 KL == 0 at init
    pol, ref = logps(model, samples), logps(model, samples, reference=True)
    d = maxdiff(pol, ref)
    k3 = max(float((torch.exp(r - p) - (r - p) - 1).abs().max()) for p, r in zip(pol, ref))
    assert d <= 1e-5 and k3 <= 1e-8, (d, k3)
    log("2 ok: init |logp - ref| max %.3g, k3 max %.3g" % (d, k3))
    # ---- 3 eval vs train_nodropout
    tr = logps(model, samples, mode="train_nodropout")
    d3 = maxdiff(pol, tr)
    log("3 eval vs train_nodropout max |dlogp| = %.3g" % d3)
    assert d3 <= 1e-4, d3

    # ---- 5 gradients only on LoRA (manual backward of one sample)
    RA._set_mode(model, "train_nodropout")
    s = samples[0]
    lp, _ = RA.token_logprobs(model, s["prompt_ids"], s["gen_ids"], 1.0)
    (-(lp.sum()) * 1.0).backward()
    with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
    assert with_grad and all("lora_" in n for n in with_grad), [n for n in with_grad if "lora_" not in n][:5]
    nz = sum(float(p.grad.abs().sum()) > 0 for n, p in model.named_parameters() if p.grad is not None and "lora_B" in n)
    assert nz > 0, "no non-zero gradient on lora_B"
    model.zero_grad(set_to_none=True)
    model.eval()
    log("5 ok: %d params with grad, all LoRA; %d lora_B with non-zero grad" % (len(with_grad), nz))

    # ---- 4 + 6 GRPO update
    learner = RA.TorchLearner(model, "grpo", {"minibatches": 2, "epochs": 1}, lr=1e-4, seed=0)
    sha0 = learner.policy_sha()
    st = learner.update(fresh(samples), {"lr": 1e-4, "kl_coef": 0.04}, seed=1)
    log("update stats", {k: st[k] for k in ("loss", "kl", "ratio_mean", "clip_frac", "grad_norm", "n_tokens",
                                            "ratio_init_maxdev", "optimizer_steps")})
    assert st["ratio_init_maxdev"] is not None and st["ratio_init_maxdev"] <= 1e-4, st["ratio_init_maxdev"]
    assert math.isfinite(st["loss"]) and math.isfinite(st["grad_norm"]) and st["optimizer_steps"] == 2
    log("4 ok: ratio at update start max |r-1| = %.3g; loss %.4g finite" % (st["ratio_init_maxdev"], st["loss"]))
    pol1, ref1 = logps(model, samples), logps(model, samples, reference=True)
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
    k = min(2, len(s0["gen_ids"]) - 1)
    s0.update(adv=0.0, adv_stop=1.0, stop_mask=[1 if i == k else 0 for i in range(len(s0["gen_ids"]))])
    before = logps(model, [s0])[0]
    lr10 = RA.TorchLearner(model, "grpo", {"minibatches": 1, "epochs": 1}, lr=1e-4, seed=0)
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
    log("10 ok: stop-credit token log-prob +%.3g; wrong-length mask refused" % dk)

    # ---- 9 PPO smoke (optional algorithm)
    ppo = RA.TorchLearner(model, "ppo", {}, lr=1e-5, seed=0)
    stp = ppo.update(fresh(samples), {"lr": 1e-5, "kl_coef": 0.0}, seed=4)
    assert math.isfinite(stp["loss"]) and stp["value_mse"] is not None and stp["ratio_init_maxdev"] <= 1e-4
    log("9 ok: PPO loss %.4g value_mse %.4g" % (stp["loss"], stp["value_mse"]))
    log("ALL SERVER CHECKS PASSED")


if __name__ == "__main__":
    main()
