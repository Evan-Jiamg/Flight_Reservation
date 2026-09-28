#!/usr/bin/env python3
"""Evaluate a frozen YES/NO stop adapter with the exact SFT prompt and truncation."""
import argparse
import hashlib
import json
from pathlib import Path

from stop_prompt import encode_stop_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--purpose", choices=("prism_validation", "prism_holdout", "trec_inner_validation", "trec_outer_test"), required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    rows = [json.loads(x) for x in Path(args.prompts).read_text(encoding="utf-8").splitlines()
            if x.strip()]
    if not rows or len({(r["record_id"], r["turn_index"]) for r in rows}) != len(rows):
        raise ValueError("empty or duplicate evaluation positions")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    choices = [tok.encode(x, add_special_tokens=False) for x in ("NO", "YES")]
    if any(len(x) != 1 for x in choices):
        raise ValueError("YES/NO must tokenize to one token")
    tokens = [x[0] for x in choices]
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model,
        quantization_config=quant, device_map={"": args.gpu}, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError("evaluation output already exists; do not silently rescore test")
    with out.open("w", encoding="utf-8") as target, torch.no_grad():
        for index, row in enumerate(rows, 1):
            ids, info = encode_stop_prompt(tok, row["user"], args.max_length)
            encoded = torch.tensor([ids], device=f"cuda:{args.gpu}")
            logits = model(input_ids=encoded, use_cache=False).logits[0, -1, tokens].float()
            p_stop = torch.softmax(logits, dim=-1)[1].item()
            target.write(json.dumps({"record_id": row["record_id"],
                "turn_index": row["turn_index"], "position": row["position"],
                "target_stop": row["should_stop"], "p_stop": p_stop,
                "stop": p_stop >= 0.5}) + "\n")
            target.flush()
            if index % 25 == 0:
                print(f"evaluated {index}/{len(rows)}", flush=True)
    meta = {"purpose": args.purpose, "rows": len(rows),
            "prompts_sha256": hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
            "adapter": args.adapter, "max_length": args.max_length,
            "decision_threshold": 0.5}
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                        encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()


