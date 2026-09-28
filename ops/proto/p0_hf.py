# Phase 0 + prototype, HF side (read-only):
#  (1) how far did the policy move in the gate's one update: KL(u0 || u1) on tokens the u0 policy sampled in the gate's
#      Task 2 rollouts, and the change of the end_session value probability;
#  (2) HF generation speed on the same prompts the vLLM prototype timed in batches of 4 (u1 adapter, T=1.0);
#  (3) sampler/learner mismatch: HF (learner) log-probs of the tokens vLLM sampled vs vLLM's own log-probs.
import json
import math
import os
import sys
import time

import torch

G = "/tmp2/mzjiang_usersim/grpo_planner"
sys.path.insert(0, G + "/code_snapshots/pend_v10")
import rl_algos as RA  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

RUN = G + "/runs/pend_f2_v10/"
OUT = G + "/phase0/"
Q4 = os.environ["Q4"]
K = int(os.environ.get("KL_STEPS", "40"))
dev = "cuda:0"                                     # CUDA_VISIBLE_DEVICES selects the physical GPU

tok = AutoTokenizer.from_pretrained(Q4)
import fit_prompts as F  # noqa: E402
base = AutoModelForCausalLM.from_pretrained(Q4, **F.dtype_kwarg(torch.bfloat16)).to(dev)
model = PeftModel.from_pretrained(base, RUN + "ckpt/u00001/adapter", adapter_name="u1")
model.load_adapter(RUN + "ckpt/u00000/adapter", adapter_name="u0")
model.eval()
summary = {}

# ---------------------------------------------------------------- (1) KL u0 -> u1
steps = []
for l in open(RUN + "rollouts.jsonl", encoding="utf-8"):
    r = json.loads(l)
    for s in r["episode"]["trace"]:
        g = s.get("planner_gen")
        if g:
            steps.append(g)
pick = steps[:: max(1, len(steps) // K)][:K]
d_all, stop0, stop1 = [], [], []
with torch.no_grad():
    for g in pick:
        model.set_adapter("u0")
        lp0 = RA.token_logprobs(model, g["prompt_ids"], g["gen_ids"], 1.0)[0]
        model.set_adapter("u1")
        lp1 = RA.token_logprobs(model, g["prompt_ids"], g["gen_ids"], 1.0)[0]
        d_all.append((lp0 - lp1).cpu())
        m = g.get("stop_mask")
        if m:
            mm = torch.tensor(m, dtype=torch.bool)
            stop0.append(float(lp0.cpu()[mm].sum().exp()))
            stop1.append(float(lp1.cpu()[mm].sum().exp()))
d = torch.cat(d_all)
summary["kl_u0_u1"] = {"steps": len(pick), "tokens": int(d.numel()),
                       "k1_mean_logp0_minus_logp1": float(d.mean()),
                       "k3": float((torch.exp(-d) - 1 + d).mean()),
                       "abs_dlogp_mean": float(d.abs().mean()), "abs_dlogp_max": float(d.abs().max()),
                       "stop_value_prob_u0_mean": sum(stop0) / max(1, len(stop0)),
                       "stop_value_prob_u1_mean": sum(stop1) / max(1, len(stop1)),
                       "stop_value_abs_change_mean": sum(abs(a - b) for a, b in zip(stop0, stop1)) / max(1, len(stop0)),
                       "note": "per-token KL(u0||u1) on tokens sampled by u0; GRPO/PPO steps commonly sit around 1e-3..1e-2"}
print(json.dumps(summary["kl_u0_u1"], indent=1), flush=True)

# ---------------------------------------------------------------- (2) HF speed on the vLLM group-A prompts
rows = [json.loads(l) for l in open(OUT + "vllm_outputs.jsonl", encoding="utf-8")]
A = [r["prompt_ids"] for r in rows if r["group"] == "A"]
model.set_adapter("u1")
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
          else [model.generation_config.eos_token_id])
n_gen, t0 = 0, time.time()
with torch.no_grad():
    for i in range(0, len(A), 4):
        batch = A[i:i + 4]
        L = max(map(len, batch))
        ids = torch.tensor([[pad] * (L - len(p)) + p for p in batch], device=dev)
        att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in batch], device=dev)
        torch.manual_seed(1000 + i)
        out = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_p=1.0,
                             max_new_tokens=1536, pad_token_id=pad)
        for row in out[:, L:].tolist():
            cut = next((j for j, x in enumerate(row) if x in eos), None)
            n_gen += (cut + 1) if cut is not None else len(row)
t_hf = time.time() - t0
summary["hf_speed_batches_of_4"] = {"n": len(A), "s": round(t_hf, 1), "gen_tokens": n_gen,
                                    "tok_per_s": round(n_gen / t_hf, 1), "s_per_call": round(t_hf / len(A), 2)}
print(json.dumps(summary["hf_speed_batches_of_4"]), flush=True)

# ---------------------------------------------------------------- (3) vLLM sampled tokens re-scored by the learner (HF)
diffs, seq_diffs, eos_end = [], [], 0
with torch.no_grad():
    for r in rows:
        if not r["gen_ids"]:
            continue
        lp = RA.token_logprobs(model, r["prompt_ids"], r["gen_ids"], 1.0)[0].cpu().tolist()
        pairs = [(a, b) for a, b in zip(lp, r["vllm_logprobs"]) if b is not None]
        diffs += [abs(a - b) for a, b in pairs]
        seq_diffs.append(abs(sum(a for a, _ in pairs) - sum(b for _, b in pairs)))
        eos_end += r["gen_ids"][-1] in eos
diffs.sort()
q = lambda x: diffs[min(len(diffs) - 1, int(x * len(diffs)))]
summary["mismatch_vllm_vs_hf"] = {"tokens": len(diffs), "abs_dlogp_mean": sum(diffs) / len(diffs), "p50": q(0.5),
                                  "p99": q(0.99), "max": diffs[-1],
                                  "seq_sum_abs_diff_mean": sum(seq_diffs) / len(seq_diffs),
                                  "outputs_ending_with_eos_token": "%d/%d" % (eos_end, len(rows)),
                                  "reference": "HF batched-vs-unbatched mean |dlogp| measured earlier: 0.007"}
print(json.dumps(summary["mismatch_vllm_vs_hf"], indent=1), flush=True)
json.dump(summary, open(OUT + "hf_summary.json", "w"), indent=1)
