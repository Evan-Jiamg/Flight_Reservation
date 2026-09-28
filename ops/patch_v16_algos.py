"""v16 items 3 (Dr. GRPO: no std normalisation) and 7 (TorchLearner.end_prob) in rl_algos.py."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''  group_advantages      GRPO: A_i = (R_i - mean) / (std + eps) within a group of G episodes of the same
                        scenario; std is the population std; a group whose std <= min_std is skipped
                        (returns None) and counted by the caller.''',
    '''  group_advantages      GRPO: A_i = R_i - mean within a group of G episodes of the same scenario (v16, Dr.
                        GRPO: no division by the group std; grpo_std_norm=True restores (R_i - mean) / (std
                        + eps)); a group whose population std <= min_std is skipped (returns None) and
                        counted by the caller.''')
rep('''ALGO_DEFAULTS = {"tis_cap": 2.0, "clip_eps": 0.2, "adv_eps": 1e-6, "min_group_std": 1e-8, "epochs": 1, "minibatches": 1,''',
    '''# grpo_std_norm (v16, user 2026-09-28): False = Dr. GRPO advantages R - mean(R); a nearly uniform group no longer
# blows a chance difference up to a unit-size advantage
ALGO_DEFAULTS = {"tis_cap": 2.0, "clip_eps": 0.2, "adv_eps": 1e-6, "min_group_std": 1e-8, "epochs": 1, "minibatches": 1,
                 "grpo_std_norm": False,''')
rep('''    if cfg["forward_mode"] not in ("train_nodropout", "eval"):
        raise ValueError(cfg["forward_mode"])
    return cfg''', '''    if cfg["forward_mode"] not in ("train_nodropout", "eval"):
        raise ValueError(cfg["forward_mode"])
    if not isinstance(cfg["grpo_std_norm"], bool):
        raise ValueError("grpo_std_norm must be true or false")
    return cfg''')
rep('''def group_advantages(rewards, eps=1e-6, min_std=1e-8):
    """GRPO. -> list of advantages, or None when the group has (near) zero spread (skipped)."""
    if len(rewards) < 2:
        raise ValueError("a GRPO group needs at least 2 episodes")
    m, s = mean(rewards), pstd(rewards)
    if s <= min_std:
        return None
    return [(r - m) / (s + eps) for r in rewards]''',
    '''def group_advantages(rewards, eps=1e-6, min_std=1e-8, std_norm=False):
    """GRPO. -> list of advantages, or None when the group has (near) zero spread (skipped).
    std_norm=False (v16 default): A = R - mean(R); True: (R - mean) / (std + eps)."""
    if len(rewards) < 2:
        raise ValueError("a GRPO group needs at least 2 episodes")
    m, s = mean(rewards), pstd(rewards)
    if s <= min_std:
        return None
    d = (s + eps) if std_norm else 1.0
    return [(r - m) / d for r in rewards]''')
rep('''def split_group_advantages(rewards, stop_parts, eps=1e-6, min_std=1e-8):
    """GRPO with stop credit, normalised ONCE: A_i = (R_i - mean R) / std(R) is split into the part
    caused by the length term S (-> end_session tokens) and the rest (-> every token):
        A_stop_i = (S_i - mean S) / std(R),   A_seq_i = ((R_i - S_i) - mean(R - S)) / std(R),
    so A_stop + A_seq = the plain GRPO advantage and the reward weights keep their effect.
    -> (A_seq, A_stop) or (None, None) when the group has (near) zero spread."""''',
    '''def split_group_advantages(rewards, stop_parts, eps=1e-6, min_std=1e-8, std_norm=False):
    """GRPO with stop credit, normalised ONCE: A_i = (R_i - mean R) / d is split into the part
    caused by the length term S (-> end_session tokens) and the rest (-> every token):
        A_stop_i = (S_i - mean S) / d,   A_seq_i = ((R_i - S_i) - mean(R - S)) / d,
    so A_stop + A_seq = the plain GRPO advantage and the reward weights keep their effect; d = 1 (v16 default,
    Dr. GRPO) or std(R) + eps with std_norm=True.
    -> (A_seq, A_stop) or (None, None) when the group has (near) zero spread."""''')
rep('''    rest = [r - q for r, q in zip(rewards, stop_parts)]
    ms, mr = mean(stop_parts), mean(rest)
    return [(x - mr) / (s + eps) for x in rest], [(x - ms) / (s + eps) for x in stop_parts]''',
    '''    rest = [r - q for r, q in zip(rewards, stop_parts)]
    ms, mr = mean(stop_parts), mean(rest)
    d = (s + eps) if std_norm else 1.0
    return [(x - mr) / d for x in rest], [(x - ms) / d for x in stop_parts]''')
rep('''        if algo == "grpo":
            a = group_advantages(rs, cfg["adv_eps"], cfg["min_group_std"])''',
    '''        if algo == "grpo":
            a = group_advantages(rs, cfg["adv_eps"], cfg["min_group_std"], cfg["grpo_std_norm"])''')
# item 7: probability of end_session = true (teacher forced, own greedy prefix)
rep('''    def load(self, d):
        import os
        import torch
        self.load_policy(d)''', '''    def end_prob(self, x):
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

    def load(self, d):
        import os
        import torch
        self.load_policy(d)''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
