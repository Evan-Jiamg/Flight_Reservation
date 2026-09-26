# Speed prototype (read-only): Planner generation through vLLM on REAL recorded Planner prompts (token ids) of the
# v10 gate, with the u1 LoRA, at a memory share that fits next to gpt-oss on GPU0. Writes the outputs so the HF
# script can (a) time HF on the same prompts and (b) re-score the vLLM tokens (sampler/learner mismatch).
import json
import os
import time

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

RUN = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v10/"
OUT = "/tmp2/mzjiang_usersim/grpo_planner/phase0/"
Q4 = os.environ["Q4"]
UTIL = float(os.environ.get("VLLM_UTIL", "0.15"))
N = int(os.environ.get("N_PROMPTS", "32"))
MAX_NEW = 1536                                   # task2_env.PLANNER_MAX_NEW

prompts = []
for l in open(RUN + "rollouts.jsonl", encoding="utf-8"):
    e = json.loads(l)["episode"]
    for s in e["trace"]:
        g = s.get("planner_gen")
        if g:
            prompts.append(list(g["prompt_ids"]))
prompts = prompts[:N]
half = N // 2
A, B = prompts[:half], prompts[half:]           # A: batches of 4 (4 workers); B: all at once (upper bound)
print("prompts", len(prompts), "prompt tokens mean", sum(map(len, prompts)) / len(prompts), "max", max(map(len, prompts)))

t0 = time.time()
llm = LLM(model=Q4, enable_lora=True, max_lora_rank=16, max_loras=1, gpu_memory_utilization=UTIL,
          max_model_len=16384, enable_prefix_caching=True, dtype="bfloat16", seed=0)
t_load = time.time() - t0
import vllm  # noqa: E402
lora = LoRARequest("u1", 1, RUN + "ckpt/u00001/adapter")
tok = llm.get_tokenizer()
gen_cfg_eos = None
try:
    from transformers import GenerationConfig
    ec = GenerationConfig.from_pretrained(Q4).eos_token_id
    gen_cfg_eos = ec if isinstance(ec, list) else [ec]
except Exception as e:                            # recorded, not fatal
    gen_cfg_eos = "unavailable: %r" % e


def run(batch, seed0):
    sps = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=MAX_NEW, seed=seed0 + i, logprobs=0)
           for i in range(len(batch))]
    return llm.generate([{"prompt_token_ids": p} for p in batch], sps, lora_request=lora, use_tqdm=False)


res = []
t = time.time()
for i in range(0, len(A), 4):
    for p, o in zip(A[i:i + 4], run(A[i:i + 4], 1000 + i)):
        res.append((p, o, "A"))
t_A = time.time() - t
t = time.time()
for p, o in zip(B, run(B, 2000)):
    res.append((p, o, "B"))
t_B = time.time() - t

rows = []
for p, o, grp in res:
    c = o.outputs[0]
    ids = list(c.token_ids)
    lps = []
    for j, tid in enumerate(ids):
        d = c.logprobs[j] if c.logprobs else None
        lps.append(d[tid].logprob if (d and tid in d) else None)
    rows.append({"group": grp, "prompt_ids": p, "gen_ids": ids, "vllm_logprobs": lps, "finish_reason": c.finish_reason,
                 "stop_reason": c.stop_reason, "last_token": ids[-1] if ids else None})
with open(OUT + "vllm_outputs.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
nA = sum(len(r["gen_ids"]) for r in rows if r["group"] == "A")
nB = sum(len(r["gen_ids"]) for r in rows if r["group"] == "B")
summary = {"vllm_version": vllm.__version__, "util": UTIL, "load_s": round(t_load, 1),
           "A_batches_of_4": {"n": len(A), "s": round(t_A, 1), "gen_tokens": nA, "tok_per_s": round(nA / t_A, 1),
                              "s_per_call": round(t_A / len(A), 2)},
           "B_all_at_once": {"n": len(B), "s": round(t_B, 1), "gen_tokens": nB, "tok_per_s": round(nB / t_B, 1),
                             "s_per_call": round(t_B / len(B), 2)},
           "finish_reasons": {k: sum(1 for r in rows if r["finish_reason"] == k) for k in {r["finish_reason"] for r in rows}},
           "eos_ids_generation_config": gen_cfg_eos, "tokenizer_eos": tok.eos_token_id,
           "last_token_of_stopped": sorted({r["last_token"] for r in rows if r["finish_reason"] == "stop"}),
           "stop_reasons": sorted({str(r["stop_reason"]) for r in rows})}
json.dump(summary, open(OUT + "vllm_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
