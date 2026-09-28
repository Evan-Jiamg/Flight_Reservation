#!/usr/bin/env python3
"""Stage D1: one forward pass per gate prompt -> (Stage B logit, last hidden state).

Items:
  * every step of logged no-gate Task 2 episodes (gate_prompt), for the fold's inner
    train / inner validation scenarios only;
  * every gold-prefix position of the fold's nested inner_train / inner_validation rows.
Encoding is stop_prompt.encode_stop_prompt(max_length 2048), identical to training/eval.
Outputs one .npz per fold: meta rows (JSON) + float16 hidden matrix + float32 logit.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stop_prompt import encode_stop_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--nested-dir", required=True)
    ap.add_argument("--episodes", action="append", default=[], help="logged no-gate JSONL(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    manifest = json.load(open(os.path.join(args.nested_dir, "nested_manifest.json")))
    fold = [f for f in manifest["folds"] if f["fold"] == args.fold][0]
    side_of = {c: "inner_train" for c in fold["inner_train_ids"]}
    side_of.update({c: "inner_validation" for c in fold["inner_validation_ids"]})

    items = []
    for side in ("inner_train", "inner_validation"):
        for l in open(os.path.join(args.nested_dir, "fold%d_%s.jsonl" % (args.fold, side))):
            r = json.loads(l)
            items.append(({"kind": "gold", "side": side, "record_id": r["record_id"],
                           "turn_index": int(r["turn_index"]), "target_stop": bool(r["should_stop"])},
                          r["user"]))
    for path in args.episodes:
        for l in open(path, encoding="utf-8"):
            e = json.loads(l)
            side = side_of.get(e["conversation_id"])
            if side is None:
                continue          # outer test or mixed for this fold: never touched
            for s in e["trace"]:
                items.append(({"kind": "episode", "side": side, "conversation_id": e["conversation_id"],
                               "seed": e["seed"], "replicate": e.get("replicate", 0),
                               "speaker": e.get("speaker_kind", "userlm"), "t": s["t"],
                               "source": os.path.basename(os.path.dirname(path))},
                              s["gate_prompt"]))
    print("items", len(items), flush=True)

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    yes_no = [tok.encode(x, add_special_tokens=False)[0] for x in ("NO", "YES")]
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quant,
                                                device_map={"": args.gpu}, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    H = None
    logits = np.zeros(len(items), dtype=np.float32)
    with torch.no_grad():
        for i, (meta, prompt) in enumerate(items):
            ids, info = encode_stop_prompt(tok, prompt, 2048)
            out = model(input_ids=torch.tensor([ids], device="cuda:%d" % args.gpu),
                        use_cache=False, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1].float().cpu().numpy()
            if H is None:
                H = np.zeros((len(items), h.shape[0]), dtype=np.float16)
            H[i] = h
            lg = out.logits[0, -1, yes_no].float()
            logits[i] = (lg[1] - lg[0]).item()      # log-odds of YES
            meta["removed_tokens"] = info["removed_tokens"]
            if (i + 1) % 100 == 0:
                print("done", i + 1, flush=True)
    sha = hashlib.sha256(open(os.path.join(args.adapter, "adapter_model.safetensors"), "rb").read()).hexdigest()
    np.savez_compressed(args.out, hidden=H, logit=logits,
                        meta=np.array(json.dumps([m for m, _ in items])),
                        info=np.array(json.dumps({"adapter": args.adapter, "adapter_sha256": sha,
                                                  "fold": args.fold, "dtype": args.dtype,
                                                  "episodes": args.episodes,
                                                  "base_model": args.base_model})))
    print("saved", args.out, H.shape, flush=True)


if __name__ == "__main__":
    main()
