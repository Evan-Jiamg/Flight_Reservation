"""Truncated importance sampling for vLLM-sampled rollouts (user decision 2026-09-26, cap C = 2). Run from sep-sim/."""


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


patch("rl_algos.py", [
    ('''ALGO_DEFAULTS = {"clip_eps": 0.2, "adv_eps": 1e-6, "min_group_std": 1e-8, "epochs": 1, "minibatches": 1,''',
     '''ALGO_DEFAULTS = {"tis_cap": 2.0, "clip_eps": 0.2, "adv_eps": 1e-6, "min_group_std": 1e-8, "epochs": 1, "minibatches": 1,'''),
    ('''ALGO_BOUNDS = {"clip_eps": (0.0, 1.0),''',
     '''ALGO_BOUNDS = {"tis_cap": (1.0, 100.0), "clip_eps": (0.0, 1.0),'''),
    ('''def rloo_advantages(rewards):''',
     '''def tis_weights(old_logp, behav_logp, cap):
    """Truncated importance sampling (pure python reference of the learner's tensor code): the rollout tokens were
    sampled by the vLLM engine (behaviour log-probs), the learner's policy is the HF model (old log-probs); each
    token's surrogate is weighted by min(exp(old - behav), cap)."""
    if len(old_logp) != len(behav_logp):
        raise ValueError("old / behaviour log-prob length mismatch")
    return [min(math.exp(o - b), cap) for o, b in zip(old_logp, behav_logp)]


def rloo_advantages(rewards):'''),
    ('''                    "note_mask": list(g["note_mask"]) if g.get("note_mask") is not None else None})''',
     '''                    "note_mask": list(g["note_mask"]) if g.get("note_mask") is not None else None,
                    "behav_logp": list(g["gen_logprobs"]) if g.get("gen_logprobs") is not None else None,
                    "gen_adapter": g.get("gen_adapter")})'''),
    ('''                    if clipped:
                        surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
                    else:
                        surr = ratio * adv''',
     '''                    if clipped:
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
                        surr = surr * w'''),
    ('''        st.update(n_tokens=tok_seen, kl=st["kl"] / tok_div, ratio_mean=st["ratio_mean"] / tok_div,''',
     '''        if st.get("tis_tokens"):
            n_tis = st.pop("tis_tokens")
            st["tis_w_mean"] = st.pop("tis_w_sum") / n_tis
            st["tis_capped_frac"] = st.pop("tis_capped") / n_tis
            st["behav_mismatch_mean"] = st.pop("behav_absdiff_sum") / n_tis     # mean |log pi_old - log pi_vllm|
            st["tis_tokens"] = n_tis // max(1, int(self.acfg["epochs"]))
        st.update(n_tokens=tok_seen, kl=st["kl"] / tok_div, ratio_mean=st["ratio_mean"] / tok_div,'''),
])
patch("task2_env.py", [
    ('''                        "fit": {**fit, "prompt_tokens": len(ids), "budget": self.budget, "batched": len(items),''',
     '''                        "fit": {**fit, "prompt_tokens": len(ids), "budget": self.budget, "batched": len(items), "backend": "hf",'''),
])
print("ok")
