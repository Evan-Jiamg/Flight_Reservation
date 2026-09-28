#!/usr/bin/env python3
"""Position-offset ("Cox") single-token stop SFT (Addendum D).

YES log-odds = content(x) + b[min(turn_index, TMAX)], where content(x) = logit(YES) - logit(NO)
of the (LoRA) language model and b is a learned per-turn offset vector (the baseline hazard).
The offsets absorb the marginal effect of position, so gradient pressure on content(x) comes
from within-turn differences. content(x) is the transferable gate; b is dataset-specific and
reported explicitly. With --no-offset this reduces to the original train_stop_sft.py objective
(binary CE on the YES-NO difference == 2-way CE).

Per epoch (including epoch 0 = init) it writes validation predictions with p_stop (full,
with offsets), p_content (content only), turn index, and the metrics: NLL (full), full AUC,
content AUC, within-same-turn content AUC. --epochs 0 evaluates an existing adapter only.
Checkpoint rule (predeclared): minimum validation NLL of the full model.
"""
import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

from stop_prompt_v2 import encode_stop_prompt_v2

TMAX = 12


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def within_turn_auc(rows, key):
    by = defaultdict(lambda: ([], []))
    for r in rows:
        by[r["turn"]][0 if r["target_stop"] else 1].append(r[key])
    num = den = 0.0
    for p, n in by.values():
        if p and n:
            num += auc(p, n) * len(p) * len(n)
            den += len(p) * len(n)
    return (num / den if den else None), int(den)


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
    ap.add_argument("--offset-lr", type=float, default=1e-2)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--compute-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--no-offset", action="store_true")
    ap.add_argument("--mask-rules", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke only")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    train, validation = read_rows(args.train), read_rows(args.validation)
    if {r["record_id"] for r in train} & {r["record_id"] for r in validation}:
        raise ValueError("train/validation session overlap")
    if any("user_id" in r for r in train + validation):
        if {r["user_id"] for r in train} & {r["user_id"] for r in validation}:
            raise ValueError("user overlap")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.compute_dtype)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.pad_token = tok.eos_token
    choice = [tok.encode(s, add_special_tokens=False) for s in ("NO", "YES")]
    assert all(len(c) == 1 for c in choice)
    choice = [c[0] for c in choice]

    audit = {}

    def encode(rows, side):
        items, removed = [], []
        for r in rows:
            ids, info = encode_stop_prompt_v2(tok, r["user"], args.max_length, mask=args.mask_rules)
            items.append((ids, int(r["should_stop"]), min(int(r["turn_index"]), TMAX),
                          r["record_id"], int(r["turn_index"])))
            removed.append(info["removed_tokens"])
        audit[side] = {"rows": len(rows), "truncated_rows": sum(x > 0 for x in removed),
                       "max_removed_tokens": max(removed)}
        return items

    tr_items, va_items = encode(train, "train"), encode(validation, "validation")
    # offset init: smoothed empirical log-odds of stop per turn on TRAIN only
    cnt = defaultdict(lambda: [1.0, 1.0])
    for _, y, t, _, _ in tr_items:
        cnt[t][y] += 1
    b0 = torch.tensor([math.log(cnt[t][1] / cnt[t][0]) for t in range(TMAX + 1)])
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quant,
                                                device_map={"": args.gpu}, low_cpu_mem_usage=True)
    base = prepare_model_for_kbit_training(base)
    if args.init_adapter:
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                                                task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    device = "cuda:%d" % args.gpu
    offset = torch.nn.Parameter(b0.clone().to(device), requires_grad=not args.no_offset)
    groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.learning_rate}]
    if not args.no_offset:
        groups.append({"params": [offset], "lr": args.offset_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)

    def content_logits(items):
        width = max(len(i[0]) for i in items)
        ids = torch.full((len(items), width), tok.pad_token_id, device=device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for k, it in enumerate(items):
            ids[k, :len(it[0])] = torch.tensor(it[0], device=device)
            mask[k, :len(it[0])] = 1
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        last = torch.tensor([len(i[0]) - 1 for i in items], device=device)
        sel = logits[torch.arange(len(items), device=device), last][:, choice].float()
        return sel[:, 1] - sel[:, 0]

    def full_logit(items, d):
        if args.no_offset:
            return d
        return d + offset[torch.tensor([i[2] for i in items], device=device)]

    @torch.no_grad()
    def evaluate(epoch):
        model.eval()
        rows, nll = [], 0.0
        for s in range(0, len(va_items), args.micro_batch):
            items = va_items[s:s + args.micro_batch]
            d = content_logits(items)
            z = full_logit(items, d)
            y = torch.tensor([float(i[1]) for i in items], device=device)
            nll += F.binary_cross_entropy_with_logits(z, y, reduction="sum").item()
            for k, it in enumerate(items):
                rows.append({"record_id": it[3], "turn_index": it[4], "turn": it[2],
                             "target_stop": bool(it[1]), "p_stop": torch.sigmoid(z[k]).item(),
                             "p_content": torch.sigmoid(d[k]).item(), "content_logit": d[k].item(),
                             "stop": bool(z[k].item() >= 0)})
        with open(out / ("epoch%d_val_predictions.jsonl" % epoch), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        pos = [r for r in rows if r["target_stop"]]
        neg = [r for r in rows if not r["target_stop"]]
        wt, pairs = within_turn_auc(rows, "p_content")
        m = {"nll": nll / len(rows),
             "auc_full": auc([r["p_stop"] for r in pos], [r["p_stop"] for r in neg]),
             "auc_content": auc([r["p_content"] for r in pos], [r["p_content"] for r in neg]),
             "within_turn_auc_content": wt, "within_turn_pairs": pairs,
             "turn_only_auc": auc([r["turn"] for r in pos], [r["turn"] for r in neg]),
             "false_stop": sum(r["stop"] for r in neg) / len(neg), "k1_end": sum(r["stop"] for r in pos) / len(pos),
             "offsets": [round(x, 4) for x in offset.detach().cpu().tolist()]}
        model.train()
        return m

    manifest = {"train_sha256": sha_file(args.train), "validation_sha256": sha_file(args.validation),
                "base_model": args.base_model, "init_adapter": args.init_adapter or None, **vars(args),
                "offset_init": b0.tolist(), "tmax": TMAX, "prompt_encoding": audit,
                "checkpoint_rule": "minimum validation NLL of the full model (content + offsets)"}
    (out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log = (out / "metrics.jsonl").open("a")
    m0 = evaluate(0)
    log.write(json.dumps({"epoch": 0, "optimizer_steps": 0, "validation": m0}) + "\n"); log.flush()
    print("epoch 0", json.dumps(m0), flush=True)
    best, steps = m0["nll"], 0
    model.save_pretrained(out / "best")
    torch.save(offset.detach().cpu(), out / "best" / "turn_offsets.pt")
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(tr_items)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        opt.zero_grad(set_to_none=True)
        tot, seen, mb = 0.0, 0, 0
        for s in range(0, len(order), args.micro_batch):
            items = [tr_items[i] for i in order[s:s + args.micro_batch]]
            d = content_logits(items)
            z = full_logit(items, d)
            y = torch.tensor([float(i[1]) for i in items], device=device)
            loss = F.binary_cross_entropy_with_logits(z, y)
            (loss / args.grad_accum).backward()
            tot += loss.item() * len(items); seen += len(items); mb += 1
            if mb % args.grad_accum == 0 or s + args.micro_batch >= len(order):
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); steps += 1
                if steps % 20 == 0:
                    print("epoch %d optimizer_step %d" % (epoch, steps), flush=True)
                if args.max_steps and steps >= args.max_steps:
                    break
        m = evaluate(epoch)
        model.save_pretrained(out / ("epoch%d" % epoch))
        torch.save(offset.detach().cpu(), out / ("epoch%d" % epoch) / "turn_offsets.pt")
        log.write(json.dumps({"epoch": epoch, "optimizer_steps": steps, "train_nll": tot / seen,
                              "validation": m}) + "\n"); log.flush()
        print("epoch %d train_nll=%.5f %s" % (epoch, tot / seen, json.dumps(m)), flush=True)
        if m["nll"] < best:
            best = m["nll"]
            model.save_pretrained(out / "best")
            torch.save(offset.detach().cpu(), out / "best" / "turn_offsets.pt")
        if args.max_steps and steps >= args.max_steps:
            break
        torch.cuda.empty_cache()
    print("complete optimizer_steps=%d best_val_nll=%.5f" % (steps, best), flush=True)


if __name__ == "__main__":
    main()
