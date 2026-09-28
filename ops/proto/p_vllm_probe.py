# Preflight probe of the REAL generation path before the formal gate: PlannerLM(load_model=False).build_prompt ->
# VLLMPlanner -> planner-base on port 8031; then a runtime LoRA load (the v10 gate's u0 adapter) and a sampled request.
# Any missing token_ids / log-probs / end token raises inside VLLMPlanner (nothing is repaired).
import json
import os
import sys

C = sys.argv[1]
sys.path.insert(0, C)
import task2_env as T2  # noqa: E402
import vllm_planner as VP  # noqa: E402

T2.setup_environment("pend")
Q4 = os.environ["Q4"]
p = T2.PlannerLM(Q4, 0, load_model=False)
VP.VLLMPlanner("http://127.0.0.1:8031/v1").attach(p)
item = {"system": "You answer in JSON.", "user": "Reply with {\"ok\": true} and nothing else.",
        "temperature": 0.0, "top_p": 1.0, "seed": 1}
g = p.generate_batch([item])[0]
print(json.dumps({"base_greedy": {"n_gen": len(g["gen_ids"]), "last_is_eos": g["gen_ids"][-1] in p.eos_ids(),
                                  "logprobs": len(g["gen_logprobs"]), "hit_max_new": g["hit_max_new"],
                                  "adapter": g["gen_adapter"], "raw": g["raw"][:80]}}))
ad = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v10/ckpt/u00000/adapter"
name = p.remote.use_adapter(ad, "probe")
g = p.generate_batch([dict(item, temperature=1.0, seed=3)])[0]
print(json.dumps({"lora_sampled": {"adapter": name, "used": g["gen_adapter"], "n_gen": len(g["gen_ids"]),
                                   "last_is_eos": g["gen_ids"][-1] in p.eos_ids(), "hit_max_new": g["hit_max_new"]}}))
p.remote.use_base()
print("PROBE OK")
