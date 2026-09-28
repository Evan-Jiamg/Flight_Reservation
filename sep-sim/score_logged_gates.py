#!/usr/bin/env python3
"""Score stop adapters offline on the gate prompts logged by a no-gate Task 2 run.

Computation is identical to eval_stop_sft.py and to the online gate in
rollout_stop_sft.py: stop_prompt.encode_stop_prompt(max_length 2048), NF4 bf16
7B base + LoRA, batch 1, softmax over the single NO/YES tokens at the last position.

Leakage: each adapter is declared with the fold it was trained for
(name=path@fold, fold -1 = not TREC-trained, e.g. PRISM). A fold adapter is only
ever run on episodes whose scenario is in THAT fold's inner_train or
inner_validation IDs; everything else is skipped and counted. Outer-test
scenarios of a fold are therefore never scored by that fold's adapters.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stop_prompt import encode_stop_prompt  # noqa: E402

G = "/tmp2/mzjiang_usersim/grpo_planner"
GATE_BASE = ("/tmp2/hf_shared/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
             "a09a35458c702b33eeacc393d103063234e8bc28")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="no-gate JSONL with gate_prompt per step")
    ap.add_argument("--adapter", action="append", required=True, help="name=path@fold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    manifest = json.load(open(os.path.join(G, "nested/nested_manifest.json")))
    role = {}
    for f in manifest["folds"]:
        for side in ("inner_train", "inner_validation"):
            for cid in f["%s_ids" % side]:
                role[(f["fold"], cid)] = side
    union = {cid for (_, cid) in role}
    adapters = []
    for spec in args.adapter:
        name, rest = spec.split("=", 1)
        path, fold = rest.rsplit("@", 1)
        sha = hashlib.sha256(open(os.path.join(path, "adapter_model.safetensors"), "rb").read()).hexdigest()
        adapters.append({"name": name, "path": path, "fold": int(fold), "sha256": sha})
    episodes = [json.loads(l) for l in open(args.episodes, encoding="utf-8") if l.strip()]
    if any(e["arm"] != "nogate" for e in episodes):
        raise SystemExit("offline scoring is defined on no-gate episodes only")

    tok = AutoTokenizer.from_pretrained(GATE_BASE)
    yes_no = [tok.encode(x, add_special_tokens=False) for x in ("NO", "YES")]
    assert all(len(x) == 1 for x in yes_no)
    yes_no = [x[0] for x in yes_no]
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(GATE_BASE, quantization_config=quant,
                                                device_map={"": args.gpu}, low_cpu_mem_usage=True)
    model = None
    for a in adapters:
        if model is None:
            model = PeftModel.from_pretrained(base, a["path"], adapter_name=a["name"])
        else:
            model.load_adapter(a["path"], adapter_name=a["name"])
    model.eval()
    if os.path.exists(args.out):
        raise SystemExit("output exists; refusing to overwrite scores")
    skipped = {a["name"]: 0 for a in adapters}
    n = 0
    with open(args.out, "w", encoding="utf-8") as out, torch.no_grad():
        for a in adapters:
            model.set_adapter(a["name"])
            for e in episodes:
                cid = e["conversation_id"]
                if a["fold"] >= 0:
                    side = role.get((a["fold"], cid))
                else:
                    side = "inner_union" if cid in union else None
                if side is None:
                    skipped[a["name"]] += 1
                    continue
                for step in e["trace"]:
                    prompt = step.get("gate_prompt")
                    if prompt is None:
                        raise SystemExit("episode lacks gate_prompt (run with --log-prompts)")
                    ids, info = encode_stop_prompt(tok, prompt, 2048)
                    logits = model(input_ids=torch.tensor([ids], device="cuda:%d" % args.gpu),
                                   use_cache=False).logits[0, -1, yes_no].float()
                    p = torch.softmax(logits, dim=-1)[1].item()
                    out.write(json.dumps({"conversation_id": cid, "seed": e["seed"],
                                          "replicate": e.get("replicate", 0), "t": step["t"],
                                          "adapter": a["name"], "adapter_fold": a["fold"],
                                          "scenario_side": side, "p_stop": p,
                                          "removed_tokens": info["removed_tokens"]}) + "\n")
                    n += 1
            print("scored", a["name"], "skipped episodes", skipped[a["name"]], flush=True)
    meta = {"episodes": args.episodes,
            "episodes_sha256": hashlib.sha256(open(args.episodes, "rb").read()).hexdigest(),
            "adapters": adapters, "rows": n, "skipped_episodes": skipped,
            "stop_prompt_sha256": hashlib.sha256(open(os.path.join(HERE, "stop_prompt.py"), "rb").read()).hexdigest()}
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
