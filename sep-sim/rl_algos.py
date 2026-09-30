"""RL objectives for the Planner: GRPO, RLOO, PPO.

Pure-python part (no torch; tested locally in test_rl_advantages.py):
  group_advantages      GRPO: A_i = R_i - mean within a group of G episodes of the same scenario (v16, Dr.
                        GRPO: no division by the group std; grpo_std_norm=True restores (R_i - mean) / (std
                        + eps)); a group whose population std <= min_std is skipped (returns None) and
                        counted by the caller.
  rloo_advantages       A_i = R_i - mean_{j != i} R_j (G >= 2).
  ppo_advantages        A = R - V (gamma = 1, reward only at the end, so the return of every Planner
                        step is the episode reward).
  normalize             batch normalisation (PPO option; statistics from the current TRAIN batch only).
  clipped_surrogate, k3 scalar reference versions of the token-level terms the torch loss uses.
  episode_samples       every Planner step of an episode -> one sample (prompt_ids, gen_ids exactly as
                        recorded at rollout time; never re-tokenised, never truncated).
  token_advantages      SPEC v17 S2: the per-token advantage adv*(1-note) + adv_prefix*prefix_mask*(1-note)
                        + adv_stop*stop_mask (one definition for the torch learner and the dry-run fake).
  aux_split             SPEC v17 S4: the stop-supervision examples of one epoch, in a given order, cut into
                        the epoch's minibatches (each example once per epoch).

Torch part (imported lazily, exercised by test_rl_algos_server.py on the server):
  setup_policy          new LoRA (r=16, alpha=32, q/k/v/o, dropout 0.0) on the Planner, fp32 trainable
                        params, bf16 base; the LoRA init is seeded (v17 B1: the start policy is the same at
                        every launch); no init adapter (v17: the SFT warm-up happens inside the run).
  load_ref_adapter      v17 §2/B4: the SFT policy u0 as a second, frozen adapter "ref" (the KL reference).
  value_nll_loss        v17 §1.2/S4: -sum w log p(target | prompt + prefix) / norm_tokens, the loss of the SFT
                        warm-up AND of the auxiliary stop supervision.
  token_logprobs        per-token log-probs of gen_ids given prompt_ids, one sequence at a time,
                        logits only for the generated positions, scaled by the rollout temperature.
  TorchLearner          old log-probs (theta_old = the rollout policy) and reference log-probs (the "ref"
                        adapter) computed once before the first epoch, then epochs x minibatches of clipped
                        updates with gradient accumulation one sequence at a time, the stop supervision split
                        across the minibatches. Records max |ratio - 1| at the start of the update (must be
                        ~0: on-policy, no dropout).

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
# grpo_std_norm (v16, user 2026-09-28): False = Dr. GRPO advantages R - mean(R); a nearly uniform group no longer
# blows a chance difference up to a unit-size advantage
# epochs 2 x minibatches 4 (SPEC v17 §3.4, user 2026-09-30): 8 optimizer steps per update
ALGO_DEFAULTS = {"tis_cap": 2.0, "clip_eps": 0.2, "adv_eps": 1e-6, "min_group_std": 1e-8, "epochs": 2, "minibatches": 4,
                 "grpo_std_norm": False,
                 "max_grad_norm": 1.0, "vf_coef": 0.5, "value_hidden": 256, "value_lr": 1e-4,
                 "rloo_clipped": False, "ppo_adv_norm": True, "ratio_init_tol": 1e-4,
                 "forward_mode": "train_nodropout", "weight_decay": 0.0, "old_logp_grad_graph": False}
ALGO_BOUNDS = {"tis_cap": (1.0, 100.0), "clip_eps": (0.0, 1.0), "adv_eps": (0.0, 1.0), "min_group_std": (0.0, 1.0), "epochs": (1, 10),
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
    if not isinstance(cfg["grpo_std_norm"], bool):
        raise ValueError("grpo_std_norm must be true or false")
    return cfg


# ------------------------------------------------------------------ pure-python advantage math
def mean(xs):
    return sum(xs) / len(xs)


def pstd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def group_advantages(rewards, eps=1e-6, min_std=1e-8, std_norm=False):
    """GRPO. -> list of advantages, or None when the group is skipped: fewer than 2 members (v17 B7: e.g. a Task 1
    group left with one sample after the mask-mismatch drops; the caller counts it) or (near) zero spread.
    std_norm=False (v16 default): A = R - mean(R); True: (R - mean) / (std + eps)."""
    if len(rewards) < 2:
        return None
    m, s = mean(rewards), pstd(rewards)
    if s <= min_std:
        return None
    d = (s + eps) if std_norm else 1.0
    return [(r - m) / d for r in rewards]


def split_group_advantages(rewards, stop_parts, eps=1e-6, min_std=1e-8, std_norm=False):
    """GRPO with stop credit, normalised ONCE: A_i = (R_i - mean R) / d is split into the part
    caused by the length term S (-> end_session tokens) and the rest (-> every token):
        A_stop_i = (S_i - mean S) / d,   A_seq_i = ((R_i - S_i) - mean(R - S)) / d,
    so A_stop + A_seq = the plain GRPO advantage and the reward weights keep their effect; d = 1 (v16 default,
    Dr. GRPO) or std(R) + eps with std_norm=True.
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
    d = (s + eps) if std_norm else 1.0
    return [(x - mr) / d for x in rest], [(x - ms) / d for x in stop_parts]


class MismatchAbort(RuntimeError):
    def __init__(self, value):
        super().__init__("mean |log pi_learner - log pi_vllm| = %.4f" % value)
        self.value = value


def tis_weights(old_logp, behav_logp, cap):
    """Truncated importance sampling (pure python reference of the learner's tensor code): the rollout tokens were
    sampled by the vLLM engine (behaviour log-probs), the learner's policy is the HF model (old log-probs); each
    token's surrogate is weighted by min(exp(old - behav), cap)."""
    if len(old_logp) != len(behav_logp):
        raise ValueError("old / behaviour log-prob length mismatch")
    return [min(math.exp(o - b), cap) for o, b in zip(old_logp, behav_logp)]


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
                    "note_mask": list(g["note_mask"]) if g.get("note_mask") is not None else None,
                    # v17 S2: the tokens before the end_session value (Task 1 samples with a located value)
                    "prefix_mask": list(g["prefix_mask"]) if g.get("prefix_mask") is not None else None,
                    "behav_logp": list(g["gen_logprobs"]) if g.get("gen_logprobs") is not None else None,
                    "gen_adapter": g.get("gen_adapter")})
    return out


def prefix_mask_of(stop_mask):
    """v17 §3.2: 1 on every generated token before the first end_session value token, else 0 (None without a mask)."""
    if not stop_mask or 1 not in stop_mask:
        return None
    i = list(stop_mask).index(1)
    return [1] * i + [0] * (len(stop_mask) - i)


def token_advantages(s, n_tokens):
    """SPEC v17 S2 (user 2026-09-30): the advantage of every generated token of sample s,
        adv * (1 - note) + adv_prefix * prefix_mask * (1 - note) + adv_stop * stop_mask,
    note_mask None = all 0 (D4: profile_note tokens carry no sequence / prefix advantage, only the KL term). A mask of
    another length than the generation, or a prefix_mask overlapping the stop_mask, raises. -> list of floats."""
    def mask(key):
        m = s.get(key)
        if m is None:
            return None
        if len(m) != n_tokens:
            raise AssertionError("%s length %d != %d generated tokens" % (key, len(m), n_tokens))
        return [float(v) for v in m]
    note, pm, sm = mask("note_mask"), mask("prefix_mask"), mask("stop_mask")
    if pm is not None and sm is not None and any(a and b for a, b in zip(pm, sm)):
        raise AssertionError("prefix_mask and stop_mask overlap")
    a, ap, ast = float(s.get("adv") or 0.0), float(s.get("adv_prefix") or 0.0), float(s.get("adv_stop") or 0.0)
    if ap and pm is None:
        raise AssertionError("adv_prefix without a prefix_mask")
    out = []
    for i in range(n_tokens):
        keep = 1.0 - (note[i] if note is not None else 0.0)
        v = a * keep
        if pm is not None:
            v += ap * pm[i] * keep
        if sm is not None:
            v += ast * sm[i]
        out.append(v)
    return out


def aux_split(n_aux, order, k):
    """v17 S4: the stop-supervision examples of one epoch in `order` (a permutation of range(n_aux)), cut in order into
    k consecutive parts of near-equal size (the first n_aux % k parts one longer); with fewer examples than k some
    minibatches get none. Every example is used exactly once per epoch. -> list of k index lists."""
    if sorted(order) != list(range(n_aux)):
        raise ValueError("aux order is not a permutation of the %d examples" % n_aux)
    q, r = divmod(n_aux, k)
    out, i = [], 0
    for j in range(k):
        m = q + (1 if j < r else 0)
        out.append(list(order[i:i + m]))
        i += m
    return out


def advantages_for_groups(groups, algo, cfg):
    """groups: list of lists of rewards (one list per scenario group).
    -> (list of per-episode advantages or None per group, n_skipped). PPO returns None here (per-step
    advantages need the value head). v17 B7: a group with fewer than 2 members gets None and is NOT counted in
    n_skipped (the zero-spread count); the caller counts those groups itself."""
    out, skipped = [], 0
    for rs in groups:
        if len(rs) < 2:
            out.append(None)
            continue
        if algo == "grpo":
            a = group_advantages(rs, cfg["adv_eps"], cfg["min_group_std"], cfg["grpo_std_norm"])
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


def setup_policy(planner, lora_init_seed, gradient_checkpointing=True, init_adapter=None):
    """Put a NEW trainable LoRA (adapter "default") on planner.model (a PlannerLM loaded WITHOUT adapter).
    v17 B1 (user 2026-09-30): torch is seeded with lora_init_seed (the trainer passes seed_of(seed, "lora_init"))
    right before get_peft_model, so the start policy (LoRA B = 0, A drawn from the seed) and its sha are the same at
    every launch. v17: an init adapter is refused (the SFT warm-up runs inside the trainer; the KL reference is the
    SFT policy u0, loaded as the adapter "ref")."""
    if init_adapter:
        raise ValueError("v17: no --init-adapter (the SFT warm-up runs inside train_planner_rl.py)")
    if lora_init_seed is None:
        raise ValueError("v17 B1: the LoRA initialisation must be seeded")
    import torch
    from peft import LoraConfig, get_peft_model
    model = planner.model
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = True        # generation keeps its KV cache; training forwards pass use_cache=False
    torch.manual_seed(int(lora_init_seed))
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


REF_ADAPTER = "ref"


def ref_param_names(model):
    return [n for n, _ in model.named_parameters() if (".%s." % REF_ADAPTER) in n]


def load_ref_adapter(model, path, optimizer=None):
    """v17 §2 / B4 (user 2026-09-30): load the adapter at `path` (the SFT policy u0) as a second, FROZEN adapter named
    "ref" -- the KL reference. Called after the optimizer exists; the active adapter stays "default", the trainable
    parameter set and the optimizer's parameter set are asserted unchanged and every ref parameter has
    requires_grad False. Checkpoints save "default" only (TorchLearner.save). -> number of ref parameters."""
    if REF_ADAPTER in (getattr(model, "peft_config", None) or {}):
        raise AssertionError("the ref adapter is already loaded")
    names0 = [n for n, p in model.named_parameters() if p.requires_grad]
    opt0 = [id(p) for g in optimizer.param_groups for p in g["params"]] if optimizer is not None else None
    model.load_adapter(path, adapter_name=REF_ADAPTER, is_trainable=False)
    model.set_adapter("default")
    names1 = [n for n, p in model.named_parameters() if p.requires_grad]
    if names1 != names0:
        raise AssertionError("loading the ref adapter changed the trainable parameters (%d -> %d)" % (len(names0), len(names1)))
    refs = [p for n, p in model.named_parameters() if (".%s." % REF_ADAPTER) in n]
    if not refs:
        raise AssertionError("no parameter of the ref adapter found after loading %s" % path)
    if any(p.requires_grad for p in refs):
        raise AssertionError("a ref adapter parameter is trainable")
    if optimizer is not None:
        opt1 = [id(p) for g in optimizer.param_groups for p in g["params"]]
        pol = [id(p) for g in optimizer.param_groups if g.get("name") == "policy" for p in g["params"]]
        if opt1 != opt0 or set(pol) != {id(p) for p in model.parameters() if p.requires_grad}:
            raise AssertionError("the optimizer's parameter set is not the trainable set after loading the ref adapter")
    return len(refs)


def value_nll_loss(model, examples, norm_tokens, backward=True):
    """v17 §1.2 / S4 (user 2026-09-30): the loss of the SFT warm-up AND of the auxiliary stop supervision,
        -sum_x w_x log p(target_x | prompt_x + prefix_x) / norm_tokens,
    one sequence at a time (backward per example, so the graph of one example is freed before the next; no grad and
    no backward with backward=False). norm_tokens = the summed gen_len of the generations the values belong to (the
    GRPO loss's normalisation, user 2026-09-26). -> (loss value, [p(target) of each example in this forward])."""
    import torch
    if not norm_tokens or norm_tokens <= 0:
        raise ValueError("value_nll_loss needs a positive token count")
    tot, ps = 0.0, []
    for x in examples:
        with (torch.enable_grad() if backward else torch.no_grad()):
            lp, _ = token_logprobs(model, list(x["prompt_ids"]) + list(x["prefix_ids"]), x["target_ids"], 1.0)
            loss = -float(x["weight"]) * lp.sum() / float(norm_tokens)
            if backward:
                loss.backward()
        ps.append(float(lp.detach().sum().exp()))
        tot += float(loss.detach())
    return tot, ps


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


def step_stats(steps):
    """v17 §3.4: per-step mean and max of rl_grad_norm, aux_grad_norm (over the steps that carried supervision), kl and
    clip_frac, plus the steps themselves (shared by the torch learner and the dry-run fake)."""
    def mm(key, rows):
        xs = [float(r[key]) for r in rows if r.get(key) is not None]
        return (sum(xs) / len(xs), max(xs)) if xs else (None, None)
    out = {"steps": steps}
    for key, name in (("rl_grad_norm", "rl_grad_norm"), ("aux_grad_norm", "aux_grad_norm"), ("kl", "kl_step"),
                      ("clip_frac", "clip_frac_step")):
        m, x = mm(key, steps)
        out[name if key in ("rl_grad_norm", "aux_grad_norm") else name + "_mean"] = m
        out[name + "_max"] = x
    return out


class TorchLearner:
    """Holds the PEFT policy, optional value head and the optimizer. One sequence per forward.
    v17: the KL reference is the frozen adapter "ref" (load_ref: the SFT policy u0); the SFT warm-up has its own AdamW
    (sft_begin / sft_step), separate from the GRPO optimizer."""

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
        self.has_ref, self.ref_path, self._trainable0 = False, None, None
        self.sft_opt = None

    def trainable_names(self):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        return names

    def policy_sha(self):
        return tensor_sha(lora_state(self.model))

    # -------------------------------------------------------- KL reference (v17 §2, B4)
    def load_ref(self, path):
        """The SFT policy u0 (path = ckpt/u00000/adapter) as the frozen adapter "ref"; after the optimizer exists."""
        load_ref_adapter(self.model, path, self.optimizer)
        self.has_ref, self.ref_path = True, path
        self._trainable0 = self.trainable_names()

    def _check_adapters(self):
        if self.trainable_names() != self._trainable0:
            raise AssertionError("the trainable parameters changed around the ref forward")
        if any(p.requires_grad for n, p in self.model.named_parameters() if (".%s." % REF_ADAPTER) in n):
            raise AssertionError("a ref adapter parameter is trainable after switching back to the policy")

    # -------------------------------------------------------- per-sample passes
    def _logp(self, s, grad, hidden=False, reference=False):
        import torch
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            if reference:
                if not self.has_ref:
                    raise AssertionError("no KL reference loaded (v17: the SFT policy u0 as the adapter 'ref')")
                # v17 B4: switch the whole model to the frozen ref adapter for this forward only (set_adapter also
                # flips requires_grad, so the trainable set is checked after switching back); no per-forward
                # adapter_names=
                try:
                    self.model.set_adapter(REF_ADAPTER)
                    return token_logprobs(self.model, s["prompt_ids"], s["gen_ids"], s["temperature"], hidden)
                finally:
                    self.model.set_adapter("default")
                    self._check_adapters()
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

    def _grad_sq(self, params):
        return sum(float((p.grad.detach().float() ** 2).sum()) for p in params if p.grad is not None)

    def update(self, samples, cfg, seed, aux=None, aux_orders=None, mismatch_abort=None):
        """cfg: controller cfg (lr, kl_coef). aux: optional stop-token supervision examples
        {prompt_ids, prefix_ids, target_ids, want_end, weight, gen_len}; aux_orders: one permutation of the aux examples
        per epoch (the trainer draws them with seed_of(seed, "aux_mb", u, epoch)). mismatch_abort: raise MismatchAbort
        before any optimizer step when the mean |log pi_old - log pi_behaviour| of vLLM-sampled tokens exceeds it.
        v17 §3.4 (user 2026-09-30): epochs x min(minibatches, n) optimizer steps (8 with n >= 4 samples); per step the
        RL gradient of the minibatch plus the aux gradient of that minibatch's share of the supervision (S4: each
        example once per epoch, normalised by the share's summed gen_len), one clip, one step; an update with
        supervision only is one step (flag aux_only). -> stats (per-step mean and max of rl_grad_norm, aux_grad_norm,
        kl and clip_frac; optimizer_steps; aux_p_correct_before measured without grad before step 1)."""
        import torch
        samples = list(samples or [])
        aux = list(aux or [])
        if not samples and not aux:
            return {"n_samples": 0, "n_tokens": 0, "skipped_update": True, "optimizer_steps": 0}
        if aux and any("gen_len" not in x for x in aux):
            raise ValueError("aux example without gen_len: cannot normalise like the GRPO loss")
        n_ep = int(self.acfg["epochs"])
        k = min(int(self.acfg["minibatches"]), len(samples)) if samples else 1
        if aux and samples:
            if aux_orders is None or len(aux_orders) != n_ep:
                raise ValueError("the stop supervision needs one example order per epoch")
        for g in self.optimizer.param_groups:
            if g["name"] == "policy":
                g["lr"] = cfg["lr"]
        beta, eps = float(cfg["kl_coef"]), float(self.acfg["clip_eps"])
        clipped = self.algo in ("grpo", "ppo") or self.acfg["rloo_clipped"]
        self.prepare(samples)
        if mismatch_abort is not None:
            tot, n = 0.0, 0
            for s in samples:
                if s.get("behav_logp") is not None:
                    bl = torch.tensor(s["behav_logp"], dtype=s["old_logp"].dtype)
                    tot += float((s["old_logp"].cpu() - bl).abs().sum())
                    n += int(bl.numel())
            if n and tot / n > mismatch_abort:
                raise MismatchAbort(tot / n)
        _set_mode(self.model, self.acfg["forward_mode"])
        st = {"n_samples": len(samples), "loss": 0.0, "kl": 0.0, "ratio_mean": 0.0, "clip_frac": 0.0,
              "n_tokens": 0, "grad_norm": [], "value_mse": 0.0, "ratio_init_maxdev": None, "optimizer_steps": 0,
              "n_minibatches": k if samples else 0, "aux_only": bool(aux and not samples)}
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        tok_seen = 0
        steps = []
        if aux:
            # before step 1, without grad (v17 S4): how often the supervision's values are already the human's
            _, p_before = value_nll_loss(self.model, [dict(x, weight=1.0) for x in aux], 1, backward=False)
            st["aux_n"] = len(aux)
            st["aux_p_correct_before"] = sum(p_before) / len(p_before)
            st["aux_p_correct_end"] = (sum(p for p, x in zip(p_before, aux) if x["want_end"]) /
                                       max(1, sum(1 for x in aux if x["want_end"])))
            st["aux_loss"] = 0.0

        def aux_part(idx):
            """The aux gradient of these examples ADDED to the current .grad (the step's RL gradient); its own norm."""
            if not idx:
                return None
            ex = [aux[i] for i in idx]
            before = [p.grad.detach().clone() if p.grad is not None else None for p in params]
            loss, _ = value_nll_loss(self.model, ex, sum(int(x["gen_len"]) for x in ex), backward=True)
            st["aux_loss"] += loss
            sq = 0.0
            for p, b in zip(params, before):
                if p.grad is not None:
                    d = p.grad.detach().float() - (b.float() if b is not None else 0.0)
                    sq += float((d ** 2).sum())
            return sq ** 0.5

        for ep in (range(n_ep) if samples else ()):
            mbs = list(minibatches(len(samples), k, seed * 1000 + ep))
            parts = aux_split(len(aux), aux_orders[ep], k) if aux else [[] for _ in mbs]
            for k_mb, mb in enumerate(mbs):
                n_tok = sum(len(samples[i]["gen_ids"]) for i in mb)
                self.optimizer.zero_grad(set_to_none=True)
                first = st["optimizer_steps"] == 0
                maxdev, kl_s, clip_s, tok_s = 0.0, 0.0, 0.0, 0
                for i in mb:
                    s = samples[i]
                    logp, h = self._logp(s, grad=True, hidden=False)
                    old = s["old_logp"].to(logp.device)
                    ref = s["ref_logp"].to(logp.device)
                    ratio = torch.exp(logp - old)
                    # v17 S2: adv*(1-note) + adv_prefix*prefix_mask*(1-note) + adv_stop*stop_mask, per token
                    adv = torch.tensor(token_advantages(s, int(logp.shape[0])), dtype=logp.dtype, device=logp.device)
                    if clipped:
                        surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
                    else:
                        surr = ratio * adv
                    if s.get("behav_logp") is not None:
                        # rollout sampled by vLLM: truncated importance weight min(pi_old / pi_vllm, cap) per token
                        bl = torch.tensor(s["behav_logp"], dtype=logp.dtype, device=logp.device)
                        if bl.shape != logp.shape:
                            raise AssertionError("behaviour log-probs %d != %d generated tokens" % (bl.shape[0], logp.shape[0]))
                        with torch.no_grad():
                            dlp = old - bl
                            w = torch.clamp(torch.exp(dlp), max=float(self.acfg["tis_cap"]))
                            st["tis_w_sum"] = st.get("tis_w_sum", 0.0) + float(w.sum())
                            st["tis_capped"] = st.get("tis_capped", 0) + int((torch.exp(dlp) > float(self.acfg["tis_cap"])).sum())
                            st["tis_tokens"] = st.get("tis_tokens", 0) + int(w.numel())
                            st["behav_absdiff_sum"] = st.get("behav_absdiff_sum", 0.0) + float(dlp.abs().sum())
                        surr = surr * w
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
                        kl_s += float(kl.sum())
                        st["ratio_mean"] += float(ratio.sum())
                        clip_s += float(((ratio - 1).abs() > eps).float().sum())
                        tok_s += len(s["gen_ids"])
                if first:
                    # ratio_init only at the first step (later steps are off the rollout policy by construction)
                    st["ratio_init_maxdev"] = maxdev
                    if maxdev > self.acfg["ratio_init_tol"]:
                        raise AssertionError("off-policy start: max |ratio-1| = %.3g > %.3g"
                                             % (maxdev, self.acfg["ratio_init_tol"]))
                rl_gn = self._grad_sq(params) ** 0.5
                aux_gn = aux_part(parts[k_mb])            # same backward pass, same clip, same optimizer step
                gn = torch.nn.utils.clip_grad_norm_(params, self.acfg["max_grad_norm"])
                st["grad_norm"].append(float(gn))
                self.optimizer.step()
                st["optimizer_steps"] += 1
                st["kl"] += kl_s
                st["clip_frac"] += clip_s
                tok_seen += tok_s
                steps.append({"rl_grad_norm": rl_gn, "aux_grad_norm": aux_gn, "n_aux": len(parts[k_mb]),
                              "kl": kl_s / max(1, tok_s), "clip_frac": clip_s / max(1, tok_s), "grad_norm": float(gn)})
        self.optimizer.zero_grad(set_to_none=True)
        if aux and not samples:
            # v17 S3: an update with supervision only (no RL sample survived): ONE step on the whole supervision
            _set_mode(self.model, self.acfg["forward_mode"])
            aux_gn = aux_part(list(range(len(aux))))
            gn = torch.nn.utils.clip_grad_norm_(params, self.acfg["max_grad_norm"])
            st["grad_norm"].append(float(gn))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            st["optimizer_steps"] += 1
            steps.append({"rl_grad_norm": 0.0, "aux_grad_norm": aux_gn, "n_aux": len(aux), "kl": 0.0, "clip_frac": 0.0,
                          "grad_norm": float(gn)})
        self.model.eval()
        n_mb = max(1, st["optimizer_steps"])
        tok_div = max(1, tok_seen)          # 0 only for an aux-only update
        if st.get("tis_tokens"):
            n_tis = st.pop("tis_tokens")
            st["tis_w_mean"] = st.pop("tis_w_sum") / n_tis
            st["tis_capped_frac"] = st.pop("tis_capped") / n_tis
            st["behav_mismatch_mean"] = st.pop("behav_absdiff_sum") / n_tis     # mean |log pi_old - log pi_vllm|
            st["tis_tokens"] = n_tis // max(1, int(self.acfg["epochs"]))
        st.update(step_stats(steps))
        if aux:
            st["aux_loss"] = st["aux_loss"] / (n_ep if samples else 1)         # per pass over the supervision
        st.update(n_tokens=tok_seen, kl=st["kl"] / tok_div, ratio_mean=st["ratio_mean"] / tok_div,
                  clip_frac=st["clip_frac"] / tok_div, loss=st["loss"] / n_mb,
                  grad_norm=max(st["grad_norm"]) if st["grad_norm"] else 0.0,
                  value_mse=st["value_mse"] / max(1, len(samples) * int(self.acfg["epochs"])) if self.algo == "ppo" else None)
        for s in samples:          # free tensors
            for key in ("old_logp", "ref_logp", "hidden"):
                s.pop(key, None)
        return st

    # -------------------------------------------------------- SFT warm-up (v17 §1.2)
    def sft_begin(self, lr):
        """A separate AdamW for the SFT warm-up (never the GRPO optimizer), weight decay 0 (v17 N3)."""
        import torch
        self.sft_opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=float(lr),
                                         weight_decay=0.0)

    def sft_step(self, batch):
        """One SFT step: value_nll_loss over the batch (normalised by its summed gen_len), clip 1.0, the SFT AdamW
        step; forward mode train_nodropout (v17 N3). -> {loss, grad_norm, n}."""
        import torch
        if self.sft_opt is None:
            raise AssertionError("sft_begin first")
        params = [p for p in self.model.parameters() if p.requires_grad]
        _set_mode(self.model, "train_nodropout")
        self.sft_opt.zero_grad(set_to_none=True)
        loss, _ = value_nll_loss(self.model, batch, sum(int(x["gen_len"]) for x in batch), backward=True)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.sft_opt.step()
        self.sft_opt.zero_grad(set_to_none=True)
        self.model.eval()
        return {"loss": loss, "grad_norm": float(gn), "n": len(batch)}

    def sft_end(self):
        self.sft_opt = None

    # -------------------------------------------------------- checkpoint
    def save(self, d):
        import os
        import torch
        self.save_adapter(d)
        torch.save(self.optimizer.state_dict(), os.path.join(d, "optimizer.pt"))
        if self.value_head is not None:
            torch.save(self.value_head.state_dict(), os.path.join(d, "value_head.pt"))

    def save_adapter(self, d):
        """The policy adapter only (v17 B4: selected_adapters=["default"] -- the frozen "ref" is never saved)."""
        import os
        self.model.save_pretrained(os.path.join(d, "adapter"), selected_adapters=["default"])
        if os.path.isdir(os.path.join(d, "adapter", REF_ADAPTER)):
            raise AssertionError("the ref adapter was saved into %s" % d)

    def end_prob(self, x):
        """v16 validation Task 1 metric: P(end_session = true) given the prompt and the policy's own greedy prefix
        up to the value, normalised over the two values: exp(lp_true) / (exp(lp_true) + exp(lp_false)), each lp the
        summed log-prob of that value's tokens (temperature 1, no grad)."""
        import torch
        with torch.no_grad():
            ctx = list(x["prompt_ids"]) + list(x["prefix_ids"])
            lt = float(token_logprobs(self.model, ctx, x["target_true"], 1.0)[0].sum())
            lf = float(token_logprobs(self.model, ctx, x["target_false"], 1.0)[0].sum())
        m = max(lt, lf)
        return math.exp(lt - m) / (math.exp(lt - m) + math.exp(lf - m))

    def p_end_batch(self, items):
        """v17 §3.2 / S1: end_prob of each item (prompt_ids, prefix_ids = the sample's OWN prefix, target_true,
        target_false), no grad, eval mode. -> list of floats in [0, 1]."""
        self.model.eval()
        return [self.end_prob(x) for x in items]

    def load(self, d):
        import os
        import torch
        self.load_policy(d)
        dev = next(self.model.parameters()).device
        if self.value_head is not None:
            self.value_head.load_state_dict(torch.load(os.path.join(d, "value_head.pt"), map_location=dev))
        self.optimizer.load_state_dict(torch.load(os.path.join(d, "optimizer.pt"), map_location=dev))

    def load_policy(self, d):
        """The LoRA policy ("default" adapter) of a checkpoint only (no optimizer: old checkpoints keep only their
        adapter). v17: loads the chosen SFT epoch (ckpt/sft_e<k>) before u0 is saved, and u0 / the final update for the
        test evaluation; the frozen "ref" adapter is never touched."""
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
