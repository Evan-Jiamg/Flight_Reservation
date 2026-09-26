#!/usr/bin/env python3
"""Audited single-token stop SFT for PRISM pretraining and TREC adaptation."""
import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from stop_prompt import encode_stop_prompt

DECISION_SYSTEM = (
    "You decide whether a specific human user should stop talking now. "
    "Answer YES if the user would end the session now, before writing another "
    "message. Answer NO if the user still has a message to send. "
    "Return exactly YES or NO."
)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_rows(train, validation):
    for side, rows in (("train", train), ("validation", validation)):
        if not rows:
            raise ValueError(f"empty {side}")
        keys = [(r["record_id"], int(r["turn_index"])) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate {side} positions")
        for row in rows:
            if not isinstance(row["should_stop"], bool) or not row["user"]:
                raise ValueError(f"invalid {side} row")
    train_sessions = {r["record_id"] for r in train}
    val_sessions = {r["record_id"] for r in validation}
    if train_sessions & val_sessions:
        raise ValueError("train/validation session overlap")
    if any("user_id" in r for r in train + validation):
        train_users = {r["user_id"] for r in train}
        val_users = {r["user_id"] for r in validation}
        if train_users & val_users:
            raise ValueError("PRISM train/validation user overlap")
    return {"train_sessions": len(train_sessions), "validation_sessions": len(val_sessions),
            "train_positions": len(train), "validation_positions": len(validation),
            "train_positive": sum(r["should_stop"] for r in train),
            "validation_positive": sum(r["should_stop"] for r in validation)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-adapter", default="")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--compute-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke only; 0 means full epochs")
    ap.add_argument("--max-validation-rows", type=int, default=0, help="smoke only")
    args = ap.parse_args()
    if args.epochs < 1 or args.micro_batch < 1 or args.grad_accum < 1:
        raise ValueError("invalid training batch or epoch parameter")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    train = read_rows(args.train)
    validation = read_rows(args.validation)
    if args.max_validation_rows:
        validation = validation[:args.max_validation_rows]
    counts = validate_rows(train, validation)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.compute_dtype)
    if args.compute_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bfloat16; select float16")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    choices = [tokenizer.encode(s, add_special_tokens=False) for s in ("NO", "YES")]
    if any(len(t) != 1 for t in choices):
        raise ValueError("YES/NO must each tokenize to one token")
    choice_ids = [t[0] for t in choices]

    encoding_audit = {}

    def tokenize(rows, side):
        inputs = []
        removed = []
        trec_compacted = 0
        for row in rows:
            ids, info = encode_stop_prompt(tokenizer, row["user"], args.max_length)
            inputs.append((ids, int(row["should_stop"])))
            removed.append(info["removed_tokens"])
            trec_compacted += int(info["trec_compacted"])
        encoding_audit[side] = {
            "rows": len(rows), "truncated_rows": sum(x > 0 for x in removed),
            "trec_compacted_rows": trec_compacted,
            "total_removed_tokens": sum(removed), "max_removed_tokens": max(removed),
        }
        print("prompt_encoding", side, json.dumps(encoding_audit[side]), flush=True)
        return inputs

    train_inputs = tokenize(train, "train")
    validation_inputs = tokenize(validation, "validation")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                              bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quant,
        device_map={"": args.gpu}, low_cpu_mem_usage=True)
    base = prepare_model_for_kbit_training(base)
    if args.init_adapter:
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate)
    device = f"cuda:{args.gpu}"

    def batch_logits(items):
        width = max(len(ids) for ids, _ in items)
        ids = torch.full((len(items), width), tokenizer.pad_token_id,
                         device=device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        lengths = []
        for i, (seq, _) in enumerate(items):
            ids[i, :len(seq)] = torch.tensor(seq, device=device)
            mask[i, :len(seq)] = 1
            lengths.append(len(seq))
        all_logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        positions = torch.tensor(lengths, device=device) - 1
        return all_logits[torch.arange(len(items), device=device), positions][:, choice_ids]

    @torch.no_grad()
    def evaluate():
        model.eval()
        total_loss, n, false_stops, continues, caught_ends, ends = 0.0, 0, 0, 0, 0, 0
        for start in range(0, len(validation_inputs), args.micro_batch):
            items = validation_inputs[start:start + args.micro_batch]
            logits = batch_logits(items).float()
            targets = torch.tensor([label for _, label in items], device=device)
            total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
            pred = logits.argmax(dim=-1)
            false_stops += int(((pred == 1) & (targets == 0)).sum().item())
            continues += int((targets == 0).sum().item())
            caught_ends += int(((pred == 1) & (targets == 1)).sum().item())
            ends += int((targets == 1).sum().item())
            n += len(items)
        return {"nll": total_loss / n, "false_stop": false_stops / continues,
                "k1_end": caught_ends / ends, "n_cont": continues, "n_end": ends}

    manifest = {"train_sha256": sha_file(args.train), "validation_sha256": sha_file(args.validation),
                "base_model": args.base_model, "init_adapter": args.init_adapter or None,
                "seed": args.seed, "epochs": args.epochs, "micro_batch": args.micro_batch,
                "grad_accum": args.grad_accum,
                "effective_batch_positions": args.micro_batch * args.grad_accum,
                "learning_rate": args.learning_rate, "max_length": args.max_length,
                "compute_dtype": args.compute_dtype, "max_steps": args.max_steps,
                "max_validation_rows": args.max_validation_rows,
                "positive_label": "post observed human conversation end",
                "checkpoint_rule": "minimum validation NLL, predeclared",
                "prompt_encoding": encoding_audit,
                **counts}
    (out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "metrics.jsonl").write_text("", encoding="utf-8")
    best_nll = math.inf
    optimizer_steps = 0
    baseline = evaluate()
    with (out / "metrics.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"epoch": 0, "optimizer_steps": 0, "validation": baseline}) + "\n")
    print("epoch 0 validation", json.dumps(baseline), flush=True)
    if args.init_adapter:
        best_nll = baseline["nll"]
        model.save_pretrained(out / "best")
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_inputs)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, minibatches, seen_positions = 0.0, 0, 0
        for start in range(0, len(order), args.micro_batch):
            indices = order[start:start + args.micro_batch]
            items = [train_inputs[i] for i in indices]
            targets = torch.tensor([label for _, label in items], device=device)
            logits = batch_logits(items).float()
            loss = F.cross_entropy(logits, targets)
            (loss / args.grad_accum).backward()
            total_loss += loss.item() * len(items)
            seen_positions += len(items)
            minibatches += 1
            if minibatches % args.grad_accum == 0 or start + args.micro_batch >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if optimizer_steps % 20 == 0:
                    print(f"epoch {epoch} optimizer_step {optimizer_steps}", flush=True)
                if args.max_steps and optimizer_steps >= args.max_steps:
                    break
        metrics = evaluate()
        epoch_dir = out / f"epoch{epoch}"
        model.save_pretrained(epoch_dir)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"epoch": epoch, "optimizer_steps": optimizer_steps,
                                  "train_nll": total_loss / seen_positions,
                                  "validation": metrics}) + "\n")
        print(f"epoch {epoch} train_nll={total_loss / seen_positions:.5f} "
              f"validation={json.dumps(metrics)}", flush=True)
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            model.save_pretrained(out / "best")
        if args.max_steps and optimizer_steps >= args.max_steps:
            break
        torch.cuda.empty_cache()
    print(f"complete optimizer_steps={optimizer_steps} best_val_nll={best_nll:.5f}", flush=True)


if __name__ == "__main__":
    main()





