"""RL objectives for the Planner: GRPO, RLOO, PPO.

Pure-python part (no torch; tested locally in test_rl_advantages.py):
  group_advantages      GRPO: A_i = (R_i - mean) / (std + eps) within a group of G episodes of the same
                        scenario; std is the population std; a group whose std <= min_std is skipped
                        (returns None) and counted by the caller.
  rloo_advantages       A_i = R_i - mean_{j != i} R_j (G >= 2).
  ppo_advantages        A = R - V (gamma = 1, reward only at the end, so the return of every Planner
                        step is the episode reward).
  normalize             batch normalisation (PPO option; statistics from the current TRAIN batch only).
  clipped_surrogate, k3 scalar reference versions of the token-level terms the torch loss uses.
  episode_samples       every Planner step of an episode -> one sample (prompt_ids, gen_ids exactly as
                        recorded at rollout time; never re-tokenised, never truncated).

Torch part (imported lazily, exercised by test_rl_algos_server.py on the server):
  setup_policy          new LoRA (r=16, alpha=32, q/k/v/o, dropout 0.0) on the Planner, fp32 trainable
                        params, bf16 base; an --init-adapter is merged into the base first, so the KL
                        reference (model.disable_adapter()) is exactly the starting policy.
  token_logprobs        per-token log-probs of gen_ids given prompt_ids, one sequence at a time,
                        logits only for the generated positions, scaled by the rollout temperature.
  TorchLearner          old log-probs (theta_old = the rollout policy) and reference log-probs computed
                        once before the first epoch, then epochs x minibatches of clipped updates with
                        gradient accumulation one sequence at a time. Records max |ratio - 1| at the
                        start of the update (must be ~0: on-policy, no dropout).

Mode note: HF only activates gradient checkpointing when module.training is True. To keep
checkpointing AND make every forward numerically identical to eval mode, the default forward mode is
"train_nodropout": model.train() with every dropout probability asserted to be 0 (LoRA dropout 0.0,
attention_dropout 0.0). All policy forwards (old, current, reference) use the same mode and the same
call, so the importance ratio at the first step is exactly 1. forward_mode "eval" is available
(no checkpointing -> more activation memory).
"""
from __future__ import annotations

import hashlib
import math
import random

ALGOS = ("grpo", "rloo", "ppo")
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "bias": "none"}
ALGO_DEFAULTS = {"clip_eps": 0.2, "adv_eps": 1e-6, "min_group_std": 1e-8, "epochs": 1, "minibatches": 1,
                 "max_grad_norm": 1.0, "vf_coef": 0.5, "value_hidden": 256, "value_lr": 1e-4,
                 "rloo_clipped": False, "ppo_adv_norm": True, "ratio_init_tol": 1e-4,
                 "forward_mode": "train_nodropout", "weight_decay": 0.0, "old_logp_grad_graph": False}
ALGO_BOUNDS = {"clip_eps": (0.0, 1.0), "adv_eps": (0.0, 1.0), "min_group_std": (0.0, 1.0), "epochs": (1, 10),
               "minibatches": (1, 256), "max_grad_norm": (0.0, 1000.0), "vf_coef": (0.0, 10.0),
               "value_hidden": (8, 8192), "value_lr": (1e-7, 1e-2), "ratio_init_tol": (0.0, 1.0),
               "weight_decay": (0.0, 1.0)}


def algo_cfg(**over):
    cfg = dict(ALGO_DEFAULTS)
    for k, v in over.items():
        if k not in ALGO_DEFAULTS:
            raise KeyError("unknown algo cfg key %r" % k)
        cfg[k] = v
    for k, (lo, hi) in ALGO_BOUNDS.items():
        if not (lo <= float(cfg[k]) <= hi):
            raise ValueError("algo cfg %s=%r outside declared bounds [%g, %g]" % (k, cfg[k], lo, hi))
    if cfg["forward_mode"] not in ("train_nodropout", "eval"):
        raise ValueError(cfg["forward_mode"])
    return cfg


# ------------------------------------------------------------------ pure-python advantage math
def mean(xs):
    return sum(xs) / len(xs)


def pstd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def group_advantages(rewards, eps=1e-6, min_std=1e-8):
    """GRPO. -> list of advantages, or None when the group has (near) zero spread (skipped)."""
    if len(rewards) < 2:
        raise ValueError("a GRPO group needs at least 2 episodes")
    m, s = mean(rewards), pstd(rewards)
    if s <= min_std:
        return None
    return [(r - m) / (s + eps) for r in rewards]


def split_group_advantages(rewards, stop_parts, eps=1e-6, min_std=1e-8):
    """GRPO with stop credit, normalised ONCE: A_i = (R_i - mean R) / std(R) is split into the part
    caused by the length term S (-> end_session tokens) and the rest (-> every token):
        A_stop_i = (S_i - mean S) / std(R),   A_seq_i = ((R_i - S_i) - mean(R - S)) / std(R),
    so A_stop + A_seq = the plain GRPO advantage and the reward weights keep their effect.
    -> (A_seq, A_stop) or (None, None) when the group has (near) zero spread."""
    if len(rewards) != len(stop_parts):
        raise ValueError("rewards / stop parts length mismatch")
    if len(rewards) < 2:
        raise ValueError("a GRPO group needs at least 2 episodes")
    s = pstd(rewards)
    if s <= min_std:
        return None, None
    rest = [r - q for r, q in zip(rewards, stop_parts)]
    ms, mr = mean(stop_parts), mean(rest)
    return [(x - mr) / (s + eps) for x in rest], [(x - ms) / (s + eps) for x in stop_parts]


def rloo_advantages(rewards):
    """Leave-one-out baseline: A_i = R_i - mean of the other G-1 rewards."""
    g = len(rewards)
    if g < 2:
        raise ValueError("RLOO needs at least 2 episodes per group")
    tot = sum(rewards)
    return [r - (tot - r) / (g - 1) for r in rewards]


def ppo_advantages(returns, values):
    if len(returns) != len(values):
        raise ValueError("returns/values length mismatch")
    return [r - v for r, v in zip(returns, values)]


def normalize(advs, eps=1e-6):
    if len(advs) < 2:
        return list(advs)
    m, s = mean(advs), pstd(advs)
    return [(a - m) / (s + eps) for a in advs]


def clipped_surrogate(ratio, adv, eps):
    """Scalar reference of the PPO/GRPO token objective (to be maximised)."""
    return min(ratio * adv, max(1.0 - eps, min(1.0 + eps, ratio)) * adv)


def k3(logp, ref_logp):
    """Scalar reference of the k3 KL estimator to the reference policy (>= 0)."""
    d = ref_logp - logp
    return math.exp(d) - d - 1.0


def episode_samples(episode, policy_version=None):
    """One sample per Planner generation in the episode, ids exactly as recorded."""
    out = []
    for step in episode["trace"]:
        g = step.get("planner_gen")
        if g is None:
            raise ValueError("step t=%s has no planner_gen (rollout needs record_generation=True)" % step.get("t"))
        if not g["gen_ids"]:
            raise ValueError("empty generation at t=%s" % step.get("t"))
        out.append({"conversation_id": episode["conversation_id"], "replicate": episode.get("replicate"),
                    "t": step["t"], "prompt_ids": list(g["prompt_ids"]), "gen_ids": list(g["gen_ids"]),
                    "temperature": float(g["temperature"]), "policy_version": policy_version,
                    "stop_mask": list(g["stop_mask"]) if g.get("stop_mask") is not None else None,
                    "note_mask": list(g["note_mask"]) if g.get("note_mask") is not None else None})
    return out


def advantages_for_groups(groups, algo, cfg):
    """groups: list of lists of rewards (one list per scenario group).
    -> (list of per-episode advantages or None per group, n_skipped). PPO returns None here (per-step
    advantages need the value head)."""
    out, skipped = [], 0
    for rs in groups:
        if algo == "grpo":
            a = group_advantages(rs, cfg["adv_eps"], cfg["min_group_std"])
        elif algo == "rloo":
            a = rloo_advantages(rs)
            if all(abs(x) <= cfg["min_group_std"] for x in a):
                a = None
        elif algo == "ppo":
            a = [None] * len(rs)
        else:
            raise ValueError(algo)
        if a is None:
            skipped += 1
        out.append(a)
    return out, skipped


def minibatches(n, k, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = max(1, min(k, n))
    return [idx[i::k] for i in range(k)]


# ------------------------------------------------------------------ torch part (lazy imports)
def assert_no_dropout(model):
    import torch
    bad = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.p != 0.0]
    cfg = getattr(model, "config", None)
    if cfg is not None and float(getattr(cfg, "attention_dropout", 0.0) or 0.0) != 0.0:
        bad.append("config.attention_dropout")
    if bad:
        raise AssertionError("dropout must be 0 for exact on-policy ratios: %s" % bad[:5])


def setup_policy(planner, init_adapter=None, gradient_checkpointing=True):
    """Put a NEW trainable LoRA on planner.model (a PlannerLM loaded WITHOUT adapter)."""
    import torch
    from peft import LoraConfig, get_peft_model
    model = planner.model
    if init_adapter:
        from peft import PeftModel
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "quantization_method", None):
            raise NotImplementedError("merging an init adapter into a quantized base is not supported")
        model = PeftModel.from_pretrained(model, init_adapter).merge_and_unload()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = True        # generation keeps its KV cache; training forwards pass use_cache=False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    assert_no_dropout(model)
    planner.model = model
    planner.model.eval()
    return model


def lora_state(model):
    from peft import get_peft_model_state_dict
    return get_peft_model_state_dict(model)


def tensor_sha(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().float().cpu().contiguous()
        h.update(k.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _set_mode(model, mode):
    if mode == "eval":
        model.eval()
    else:
        model.train()
        assert_no_dropout(model)


def token_logprobs(model, prompt_ids, gen_ids, temperature=1.0, want_hidden=False):
    """-> (logp [G] float32, hidden at the last prompt token [H] float32 detached, or None).
    Gradients flow iff grad mode is enabled by the caller."""
    import torch
    dev = next(model.parameters()).device
    P, G = len(prompt_ids), len(gen_ids)
    ids = torch.tensor([list(prompt_ids) + list(gen_ids)], dtype=torch.long, device=dev)
    kw = dict(input_ids=ids, use_cache=False, output_hidden_states=bool(want_hidden))
    try:
        out = model(logits_to_keep=G + 1, **kw)
        logits = out.logits[0, :G]
    except TypeError:
        out = model(**kw)
        logits = out.logits[0, P - 1:P + G - 1]
    if logits.shape[0] != G:
        raise AssertionError("logit slice %d != %d generated tokens" % (logits.shape[0], G))
    logits = logits.float()
    if temperature and temperature > 0:
        logits = logits / float(temperature)
    tgt = ids[0, P:P + G]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    hidden = out.hidden_states[-1][0, P - 1].detach().float() if want_hidden else None
    return logp, hidden


def make_value_head(hidden_size, width):
    import torch
    head = torch.nn.Sequential(torch.nn.Linear(hidden_size, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))
    return head.float()


class TorchLearner:
    """Holds the PEFT policy, optional value head and the optimizer. One sequence per forward."""

    def __init__(self, model, algo, acfg, lr, seed=0):
        import torch
        if algo not in ALGOS:
            raise ValueError(algo)
        self.model, self.algo, self.acfg = model, algo, algo_cfg(**acfg)
        dev = next(model.parameters()).device
        self.value_head = None
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr, "name": "policy"}]
        if algo == "ppo":
            torch.manual_seed(seed)
            self.value_head = make_value_head(model.config.hidden_size, int(self.acfg["value_hidden"])).to(dev)
            groups.append({"params": list(self.value_head.parameters()), "lr": self.acfg["value_lr"], "name": "value"})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=self.acfg["weight_decay"])

    def trainable_names(self):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        return names

    def policy_sha(self):
        return tensor_sha(lora_state(self.model))

    # -------------------------------------------------------- per-sample passes
    def _logp(self, s, grad, hidden=False, reference=False):
        import torch
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            if reference:
                with self.model.disable_adapter():
                    return token_logprobs(self.model, s["prompt_ids"], s["gen_ids"], s["temperature"], hidden)
            return token_logprobs(self.model, s["prompt_ids"], s["gen_ids"], s["temperature"], hidden)

    def prepare(self, samples):
        """theta_old and reference log-probs (+ old values for PPO), before the first epoch."""
        import torch
        _set_mode(self.model, self.acfg["forward_mode"])
        for s in samples:
            # old_logp_grad_graph: run theta_old through the very same autograd-enabled call as the
            # update pass (then detach), in case a kernel choice depends on grad mode
            lp, h = self._logp(s, grad=bool(self.acfg["old_logp_grad_graph"]), hidden=self.algo == "ppo")
            s["old_logp"] = lp.detach()
            del lp
            s["ref_logp"] = self._logp(s, grad=False, reference=True)[0].detach()
            if self.algo == "ppo":
                s["hidden"] = h
                with torch.no_grad():
                    s["old_value"] = float(self.value_head(h.unsqueeze(0))[0, 0])
        if self.algo == "ppo":
            adv = ppo_advantages([s["ret"] for s in samples], [s["old_value"] for s in samples])
            if self.acfg["ppo_adv_norm"]:
                adv = normalize(adv, self.acfg["adv_eps"])
            for s, a in zip(samples, adv):
                s["adv"] = a

    def update(self, samples, cfg, seed, aux=None):
        """cfg: controller cfg (lr, kl_coef). aux: optional stop-token supervision examples
        {prompt_ids, prefix_ids, target_ids, want_end, weight}. -> stats dict."""
        import torch
        if not samples and not aux:
            return {"n_samples": 0, "n_tokens": 0, "skipped_update": True}
        if not samples:
            samples = []
        for g in self.optimizer.param_groups:
            if g["name"] == "policy":
                g["lr"] = cfg["lr"]
        beta, eps = float(cfg["kl_coef"]), float(self.acfg["clip_eps"])
        clipped = self.algo in ("grpo", "ppo") or self.acfg["rloo_clipped"]
        self.prepare(samples)
        _set_mode(self.model, self.acfg["forward_mode"])
        st = {"n_samples": len(samples), "loss": 0.0, "kl": 0.0, "ratio_mean": 0.0, "clip_frac": 0.0,
              "n_tokens": 0, "grad_norm": [], "value_mse": 0.0, "ratio_init_maxdev": None, "optimizer_steps": 0}
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        tok_seen = 0
        for ep in range(int(self.acfg["epochs"])):
            for mb in minibatches(len(samples), int(self.acfg["minibatches"]), seed * 1000 + ep):
                n_tok = sum(len(samples[i]["gen_ids"]) for i in mb)
                self.optimizer.zero_grad(set_to_none=True)
                first = ep == 0 and st["optimizer_steps"] == 0
                maxdev = 0.0
                for i in mb:
                    s = samples[i]
                    logp, h = self._logp(s, grad=True, hidden=False)
                    old = s["old_logp"].to(logp.device)
                    ref = s["ref_logp"].to(logp.device)
                    ratio = torch.exp(logp - old)
                    # per-token advantage: the sequence advantage on every token, plus the stop
                    # advantage on the tokens of the end_session value only (stop credit assignment)
                    adv = torch.full_like(logp, float(s["adv"]))
                    if s.get("note_mask") and float(s["adv"]) != 0.0:
                        # D4: the profile_note tokens carry no sequence advantage (only the KL term)
                        nm = torch.tensor(s["note_mask"], dtype=logp.dtype, device=logp.device)
                        if nm.shape != logp.shape:
                            raise AssertionError("note_mask length %d != %d generated tokens" % (nm.shape[0], logp.shape[0]))
                        adv = adv * (1.0 - nm)
                    if s.get("adv_stop") and s.get("stop_mask"):
                        m = torch.tensor(s["stop_mask"], dtype=logp.dtype, device=logp.device)
                        if m.shape != logp.shape:
                            raise AssertionError("stop_mask length %d != %d generated tokens" % (m.shape[0], logp.shape[0]))
                        adv = adv + float(s["adv_stop"]) * m
                    if clipped:
                        surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
                    else:
                        surr = ratio * adv
                    d = ref - logp
                    kl = torch.exp(d) - d - 1.0
                    loss = (-surr + beta * kl).sum() / n_tok
                    if self.algo == "ppo":
                        v = self.value_head(s["hidden"].to(logp.device).unsqueeze(0))[0, 0]
                        vloss = self.acfg["vf_coef"] * (v - float(s["ret"])) ** 2 / len(mb)
                        loss = loss + vloss
                        st["value_mse"] += float((v.detach() - float(s["ret"])) ** 2)
                    loss.backward()
                    with torch.no_grad():
                        if first:
                            maxdev = max(maxdev, float((ratio - 1).abs().max()))
                        st["loss"] += float(loss.detach())
                        st["kl"] += float(kl.sum())
                        st["ratio_mean"] += float(ratio.sum())
                        st["clip_frac"] += float(((ratio - 1).abs() > eps).float().sum())
                        tok_seen += len(s["gen_ids"])
                if first:
                    st["ratio_init_maxdev"] = maxdev
                    if maxdev > self.acfg["ratio_init_tol"]:
                        raise AssertionError("off-policy start: max |ratio-1| = %.3g > %.3g"
                                             % (maxdev, self.acfg["ratio_init_tol"]))
                gn = torch.nn.utils.clip_grad_norm_(params, self.acfg["max_grad_norm"])
                st["grad_norm"].append(float(gn))
                self.optimizer.step()
                st["optimizer_steps"] += 1
        self.optimizer.zero_grad(set_to_none=True)
        if aux:
            # auxiliary stop-token supervision (after the on-policy GRPO steps, so it never touches the
            # ratio check): -log p(the human's end_session value | prompt + the policy's own prefix)
            _set_mode(self.model, self.acfg["forward_mode"])
            # normalised like the GRPO loss: by the number of GENERATED tokens of the generations these values
            # belong to (not by the 1-2 target tokens, which made the aux gradient ~1400x the RL gradient in the
            # v8 smoke); w_aux is then a relative weight, tuned by the LLM controller
            if any("gen_len" not in x for x in aux):
                raise ValueError("aux example without gen_len: cannot normalise like the GRPO loss")
            n_t = sum(int(x["gen_len"]) for x in aux)
            p_before = []
            for x in aux:
                lp, _ = token_logprobs(self.model, list(x["prompt_ids"]) + list(x["prefix_ids"]),
                                       x["target_ids"], 1.0)
                p_before.append(float(lp.detach().sum().exp()))
                loss = -float(x["weight"]) * lp.sum() / n_t
                loss.backward()
                st["aux_loss"] = st.get("aux_loss", 0.0) + float(loss.detach())
            gn = torch.nn.utils.clip_grad_norm_(params, self.acfg["max_grad_norm"])
            st["aux_grad_norm"] = float(gn)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            st["optimizer_steps"] += 1
            st["aux_n"] = len(aux)
            st["aux_p_correct_before"] = sum(p_before) / len(p_before)
            st["aux_p_correct_end"] = (sum(p for p, x in zip(p_before, aux) if x["want_end"]) /
                                       max(1, sum(1 for x in aux if x["want_end"])))
        self.model.eval()
        n_mb = max(1, st["optimizer_steps"])
        tok_div = max(1, tok_seen)          # 0 only for an aux-only update
        st.update(n_tokens=tok_seen, kl=st["kl"] / tok_div, ratio_mean=st["ratio_mean"] / tok_div,
                  clip_frac=st["clip_frac"] / tok_div, loss=st["loss"] / n_mb,
                  grad_norm=max(st["grad_norm"]) if st["grad_norm"] else 0.0,
                  value_mse=st["value_mse"] / max(1, len(samples) * int(self.acfg["epochs"])) if self.algo == "ppo" else None)
        for s in samples:          # free tensors
            for k in ("old_logp", "ref_logp", "hidden"):
                s.pop(k, None)
        return st

    # -------------------------------------------------------- checkpoint
    def save(self, d):
        import os
        import torch
        self.model.save_pretrained(os.path.join(d, "adapter"))
        torch.save(self.optimizer.state_dict(), os.path.join(d, "optimizer.pt"))
        if self.value_head is not None:
            torch.save(self.value_head.state_dict(), os.path.join(d, "value_head.pt"))

    def load(self, d):
        import os
        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(os.path.join(d, "adapter", "adapter_model.safetensors"))
        res = set_peft_model_state_dict(self.model, sd)
        unexpected = getattr(res, "unexpected_keys", None)
        if unexpected:
            raise AssertionError("unexpected adapter keys on resume: %s" % unexpected[:5])
        for p in self.model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
        dev = next(self.model.parameters()).device
        if self.value_head is not None:
            self.value_head.load_state_dict(torch.load(os.path.join(d, "value_head.pt"), map_location=dev))
        self.optimizer.load_state_dict(torch.load(os.path.join(d, "optimizer.pt"), map_location=dev))


def rng_state():
    import random as _r
    import torch
    st = {"python": _r.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    import random as _r
    import torch
    _r.setstate(st["python"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
