#!/usr/bin/env python3
"""Train one cross-fitted goal-satisfaction judge: Qwen3-4B-Instruct-2507 + LoRA.

Judge (fold f, held-out group g) trains ONLY on samples whose scenario is in the fold-f cross-fit
set minus group g (crossfit_manifest_v1.json), with parsed labels from the label judge. The
target is goal_judge.canonical(label) followed by the chat template's end-of-turn; the loss is
on target tokens only. Hyperparameters are fixed in advance (no checkpoint selection): LoRA r16,
alpha 32, dropout 0.05 on q/k/v/o; lr 1e-4; 3 epochs; effective batch 16; seed 20260925.
After training it labels the held-out group's samples (greedy, same builder) and reports
agreement with the label judge -- the out-of-fold estimate. train_manifest.json is what the
rollout runner's leak gate checks.
"""
import argparse
import hashlib
import json
import os
import random
from collections import Counter

import goal_judge as GJ


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--crossfit-manifest", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=False)
    cf = json.load(open(args.crossfit_manifest))
    fold = [f for f in cf["folds"] if f["fold"] == args.fold][0]
    held = set(fold["groups"][args.group])
    train_scen = set(fold["scenarios"]) - held
    labels = {}
    for l in open(args.labels):
        r = json.loads(l)
        if r.get("parse_ok") and r["status"] in GJ.STATUSES:
            labels[r["id"]] = {"status": r["status"], "unmet": r["unmet"]}
    samples = [json.loads(l) for l in open(args.samples, encoding="utf-8") if l.strip()]
    train = [s for s in samples if s["conversation_id"] in train_scen and s["id"] in labels]
    heldout = [s for s in samples if s["conversation_id"] in held and s["id"] in labels]
    assert train and not ({s["conversation_id"] for s in train} & held)
    manifest = {"fold": args.fold, "held_out_group": args.group,
                "train_scenarios": sorted({s["conversation_id"] for s in train}),
                "held_out_scenarios": sorted(held), "n_train_samples": len(train),
                "n_heldout_samples": len(heldout),
                "train_label_dist": dict(Counter(labels[s["id"]]["status"] for s in train)),
                "sources": dict(Counter(s["source"] for s in train)),
                "samples_sha256": sha(args.samples), "labels_sha256": sha(args.labels),
                "labels_meta": json.load(open(args.labels + ".meta.json")) if os.path.exists(args.labels + ".meta.json") else None,
                "crossfit_manifest_sha256": sha(args.crossfit_manifest),
                "judge_system_sha256": hashlib.sha256(GJ.SYSTEM.encode()).hexdigest(),
                "hyper": {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                          "targets": ["q_proj", "k_proj", "v_proj", "o_proj"], "lr": args.lr,
                          "epochs": args.epochs, "accum": args.accum, "micro_batch": 1, "seed": args.seed,
                          "dtype": args.dtype, "budget": GJ.JUDGE_BUDGET},
                "base": args.base}
    json.dump(manifest, open(os.path.join(args.out, "train_manifest.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("train_scenarios", "held_out_scenarios")}),
          flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)
    dev = "cuda:%d" % args.gpu
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=getattr(torch, args.dtype),
                                                 device_map={"": args.gpu}, low_cpu_mem_usage=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    for p in model.parameters():          # LoRA weights and Adam states in fp32 (base stays bf16)
        if p.requires_grad:
            p.data = p.data.float()

    def encode(s):
        msgs, _ = GJ.fit_messages(tok, s["scenario_text"], s["hist_u"], s["hist_a"])
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": GJ.canonical(labels[s["id"]])}],
                                       tokenize=False, add_generation_prompt=False)
        if not full.startswith(prompt):
            raise ValueError("chat template does not extend the generation prompt")
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]     # template carries any BOS
        ids = tok(full, add_special_tokens=False)["input_ids"]
        assert ids[:len(p_ids)] == p_ids
        return ids, len(p_ids)

    data = [encode(s) for s in train]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps, log = 0, open(os.path.join(args.out, "train_log.jsonl"), "w")
    model.train()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(data)))
        random.Random(args.seed + epoch).shuffle(order)
        opt.zero_grad(set_to_none=True)
        tot, n = 0.0, 0
        for j, i in enumerate(order, 1):
            ids, plen = data[i]
            x = torch.tensor([ids], device=dev)
            labels_t = x.clone()
            labels_t[0, :plen] = -100
            loss = model(input_ids=x, labels=labels_t).loss
            (loss / args.accum).backward()
            tot += loss.item()
            n += 1
            if j % args.accum == 0 or j == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
                if args.max_steps and steps >= args.max_steps:
                    break
        log.write(json.dumps({"epoch": epoch, "optimizer_steps": steps, "train_loss": tot / max(1, n)}) + "\n")
        log.flush()
        print("epoch", epoch, "steps", steps, "loss %.4f" % (tot / max(1, n)), flush=True)
        if args.max_steps and steps >= args.max_steps:
            break
    model.save_pretrained(args.out)

    # out-of-fold agreement on the held-out group (greedy, same builder as inference)
    model.eval()
    rows, conf = [], Counter()
    with torch.no_grad():
        for s in heldout:
            msgs, _ = GJ.fit_messages(tok, s["scenario_text"], s["hist_u"], s["hist_a"])
            enc = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                      return_tensors="pt", add_special_tokens=False).to(dev)
            out = model.generate(**enc, max_new_tokens=GJ.MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            res, ok = GJ.parse(raw)
            gold = labels[s["id"]]["status"]
            conf[(gold, res["status"])] += 1
            rows.append({"id": s["id"], "label": gold, "pred": res["status"], "parse_ok": ok, "raw": raw})
    with open(os.path.join(args.out, "heldout_predictions.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    per = {}
    for c in GJ.STATUSES:
        tp = conf[(c, c)]
        fp = sum(v for (g, p), v in conf.items() if p == c and g != c)
        fn = sum(v for (g, p), v in conf.items() if g == c and p != c)
        per[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    f1s = [v for v in per.values() if v is not None]
    ev = {"n_heldout": n, "accuracy": sum(conf[(c, c)] for c in GJ.STATUSES) / max(1, n),
          "macro_f1": sum(f1s) / len(f1s) if f1s else None, "per_class_f1": per,
          "unparsed": sum(not r["parse_ok"] for r in rows),
          "confusion_label_by_pred": {"%s|%s" % k: v for k, v in sorted(conf.items())}}
    json.dump(ev, open(os.path.join(args.out, "eval_heldout.json"), "w"), indent=1)
    print("HELDOUT", json.dumps(ev), flush=True)


if __name__ == "__main__":
    main()
