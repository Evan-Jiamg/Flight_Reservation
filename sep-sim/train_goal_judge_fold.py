#!/usr/bin/env python3
"""S2: per-fold goal-satisfaction judge SFT (Qwen3-4B-Instruct-2507 + LoRA), splits_v1 based.

Leakage control. Judge f trains ONLY on samples whose conversation_id is in splits[f]["train"]
and whose primary label parsed; asserted disjoint from splits[f]["forbidden_for_training"] (and
from every validation/test list). Validation samples (splits[f]["validation"]) are used only for
the report and the gate. Test conversations are never read into any set.

Training (fixed in advance, no checkpoint selection): LoRA r16 / alpha 32 / dropout 0.05 on
q,k,v,o; lr 1e-4; 3 epochs; micro-batch 1, gradient accumulation 16; trainable params in fp32
(base bf16); loss on target tokens only; target = goal_judge.canonical(label) + the template's
end of turn. Prompts come from goal_judge.fit_messages (never truncated; lengths asserted),
tokenized with add_special_tokens=False (single BOS).

Resumable: one checkpoint per finished epoch (adapter + optimizer + RNG states); validation
predictions are appended per sample and skipped on restart. train_manifest.json is written before
training and must match exactly on resume (the RL trainer's leak gate reads it).

Validation report: accuracy, macro-F1, Cohen's kappa vs the primary labels (greedy decode, same
builder as inference), Brier and NLL of status_probs (teacher-forced), per source.
Gate (declared; printed): macro-F1 >= --gate-macro-f1 (0.5) AND unparsed rate < --gate-unparsed (0.02).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter

import goal_judge as GJ
import judge_metrics as JM
import label_agreement as LA

HERE = os.path.dirname(os.path.abspath(__file__))


def select(samples, labels, fold_split):
    """-> (train, validation, stats). Pure; asserts the leakage rules."""
    tr_ids, va_ids = set(fold_split["train"]), set(fold_split["validation"])
    forbidden = set(fold_split["forbidden_for_training"])
    assert not tr_ids & forbidden, "split file: train overlaps forbidden_for_training"
    assert va_ids <= forbidden, "split file: validation not listed as forbidden_for_training"
    train = [s for s in samples if s["conversation_id"] in tr_ids and s["id"] in labels]
    val = [s for s in samples if s["conversation_id"] in va_ids and s["id"] in labels]
    JM.assert_no_leak({s["conversation_id"] for s in train}, fold_split)
    assert not {s["id"] for s in train} & {s["id"] for s in val}
    stats = {"train_unlabelled_or_unparsed": sum(s["conversation_id"] in tr_ids and s["id"] not in labels
                                                 for s in samples),
             "validation_unlabelled_or_unparsed": sum(s["conversation_id"] in va_ids and s["id"] not in labels
                                                      for s in samples),
             "samples_outside_train_and_validation": sum(s["conversation_id"] not in tr_ids | va_ids
                                                         for s in samples)}
    if not train:
        raise SystemExit("no training samples for this fold")
    return train, val, stats


def build_manifest(args, train, val, labels, sel_stats, label_stats):
    lm = args.labels + ".meta.json"
    return {"fold": args.fold, "split_file": os.path.abspath(args.splits),
            "split_file_sha256": JM.sha256_file(args.splits),
            "train_scenarios": sorted({s["conversation_id"] for s in train}),
            "validation_scenarios": sorted({s["conversation_id"] for s in val}),
            "n_train_samples": len(train), "n_validation_samples": len(val),
            "train_label_dist": dict(sorted(Counter(labels[s["id"]]["status"] for s in train).items())),
            "train_sources": dict(sorted(Counter(s["source"] for s in train).items())),
            "selection": sel_stats, "label_stats": label_stats,
            "label_file": os.path.abspath(args.labels), "label_file_sha256": JM.sha256_file(args.labels),
            "labels_meta": json.load(open(lm)) if os.path.exists(lm) else None,
            "samples_file": os.path.abspath(args.samples), "samples_sha256": JM.sha256_file(args.samples),
            "judge_system_sha256": hashlib.sha256(GJ.SYSTEM.encode()).hexdigest(),
            "code_sha256": {f: JM.sha256_file(os.path.join(HERE, f))
                            for f in ("goal_judge.py", "train_goal_judge_fold.py", "judge_metrics.py")},
            "forbidden_for_training_checked": True,
            "hyper": {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                      "targets": ["q_proj", "k_proj", "v_proj", "o_proj"], "lr": args.lr,
                      "epochs": args.epochs, "accum": args.accum, "micro_batch": 1, "seed": args.seed,
                      "dtype": args.dtype, "trainable_dtype": "float32", "budget": GJ.JUDGE_BUDGET,
                      "max_new": GJ.MAX_NEW, "max_steps": args.max_steps},
            "gates": {"macro_f1_min": args.gate_macro_f1, "unparsed_rate_max_exclusive": args.gate_unparsed},
            "base": args.base}


def write_or_check_manifest(path, manifest):
    if os.path.exists(path):
        old = json.load(open(path))
        diff = sorted(k for k in set(old) | set(manifest) if old.get(k) != manifest.get(k))
        if diff:
            raise SystemExit("existing train_manifest.json differs in %s; refusing to resume" % diff)
        return False
    json.dump(manifest, open(path, "w"), indent=1)
    return True


def evaluate_rows(rows, gate_f1, gate_unparsed):
    """rows: {"label","pred","parse_ok","probs","source"} -> report with gate. Pure."""
    def block(rs):
        gold, pred = [r["label"] for r in rs], [r["pred"] for r in rs]
        probs = [r["probs"] for r in rs]
        arg = [max(JM.CLASSES, key=lambda c: p[c]) for p in probs]
        parsed = [(g, p) for g, p in zip(gold, pred) if p in JM.CLASSES]
        return {"n": len(rs), "accuracy": JM.accuracy(gold, pred), "macro_f1": JM.macro_f1(gold, pred),
                "per_class_f1": JM.per_class_f1(gold, pred),
                "kappa_all": LA.kappa(list(zip(gold, pred))), "kappa_parsed": LA.kappa(parsed),
                "unparsed": sum(not r["parse_ok"] for r in rs),
                "unparsed_rate": sum(not r["parse_ok"] for r in rs) / len(rs) if rs else None,
                "brier": JM.brier(gold, probs), "nll": JM.nll(gold, probs),
                "argmax_probs_accuracy": JM.accuracy(gold, arg),
                "label_dist": dict(Counter(gold)), "pred_dist": dict(Counter(pred)),
                "confusion_label_by_pred": {"%s|%s" % k: v for k, v in sorted(JM.confusion(gold, pred).items())}}
    rep = block(rows)
    rep["by_source"] = {src: block([r for r in rows if r["source"] == src])
                        for src in sorted({r["source"] for r in rows})}
    ok = (rep["macro_f1"] is not None and rep["macro_f1"] >= gate_f1
          and rep["unparsed_rate"] is not None and rep["unparsed_rate"] < gate_unparsed)
    rep["gate"] = {"macro_f1_min": gate_f1, "unparsed_rate_max_exclusive": gate_unparsed,
                   "result": "PASS" if ok else "FAIL"}
    return rep


def last_epoch_ckpt(out):
    best = 0
    for e in range(1, 100):
        if os.path.exists(os.path.join(out, "ckpt", "epoch%d" % e, "DONE")):
            best = e
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--gate-macro-f1", type=float, default=0.5)
    ap.add_argument("--gate-unparsed", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke only (0 = full training)")
    ap.add_argument("--max-val", type=int, default=0, help="smoke only (0 = all validation samples)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    splits = json.load(open(args.splits))
    fs = JM.fold_of(splits, args.fold)
    labels, label_stats = JM.load_labels(args.labels)
    samples = JM.load_samples(args.samples)
    train, val, sel = select(samples, labels, fs)
    manifest = build_manifest(args, train, val, labels, sel, label_stats)
    new = write_or_check_manifest(os.path.join(args.out, "train_manifest.json"), manifest)
    print(("NEW " if new else "RESUME ") + json.dumps({k: v for k, v in manifest.items()
                                                       if k not in ("train_scenarios", "validation_scenarios",
                                                                    "labels_meta")}), flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from fit_prompts import dtype_kwarg
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)
    dev = "cuda:%d" % args.gpu
    model = AutoModelForCausalLM.from_pretrained(args.base, **dtype_kwarg(getattr(torch, args.dtype)),
                                                 device_map={"": args.gpu}, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    for p in model.parameters():          # LoRA weights and Adam states in fp32 (base stays bf16)
        if p.requires_grad:
            p.data = p.data.float()

    def encode(s):
        msgs, info = GJ.fit_messages(tok, s["scenario_text"], s["hist_u"], s["hist_a"])
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": GJ.canonical(labels[s["id"]])}],
                                       tokenize=False, add_generation_prompt=False)
        if not full.startswith(prompt):
            raise ValueError("chat template does not extend the generation prompt")
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]     # template carries any BOS
        ids = tok(full, add_special_tokens=False)["input_ids"]
        assert ids[:len(p_ids)] == p_ids
        assert len(p_ids) <= GJ.JUDGE_BUDGET, "prompt %d > budget" % len(p_ids)
        assert len(ids) - len(p_ids) <= GJ.MAX_NEW, "target longer than the judge's generation budget"
        return ids, len(p_ids), info["dropped_exchanges"]

    final_done = os.path.exists(os.path.join(args.out, "TRAIN_DONE"))
    if not final_done:
        data = [encode(s) for s in train]
        fit_log = {"n": len(data), "max_prompt_tokens": max(d[1] for d in data),
                   "max_total_tokens": max(len(d[0]) for d in data),
                   "n_with_dropped_exchanges": sum(d[2] > 0 for d in data)}
        print("FIT", json.dumps(fit_log), flush=True)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        start, steps = last_epoch_ckpt(args.out), 0
        if start:
            ck = os.path.join(args.out, "ckpt", "epoch%d" % start)
            from safetensors.torch import load_file
            set_peft_model_state_dict(model, load_file(os.path.join(ck, "adapter_model.safetensors")))
            st = torch.load(os.path.join(ck, "trainer_state.pt"), map_location="cpu", weights_only=False)
            opt.load_state_dict(st["optimizer"])
            steps = st["optimizer_steps"]
            random.setstate(st["py_rng"])
            torch.set_rng_state(st["torch_rng"])
            torch.cuda.set_rng_state_all(st["cuda_rng"])
            print("resumed from epoch", start, "steps", steps, flush=True)
        log = open(os.path.join(args.out, "train_log.jsonl"), "a")
        model.train()
        for epoch in range(start + 1, args.epochs + 1):
            order = list(range(len(data)))
            random.Random(args.seed + epoch).shuffle(order)
            opt.zero_grad(set_to_none=True)
            tot, n, t0 = 0.0, 0, time.time()
            for j, i in enumerate(order, 1):
                ids, plen, _ = data[i]
                x = torch.tensor([ids], device=dev)
                lab = x.clone()
                lab[0, :plen] = -100
                loss = model(input_ids=x, labels=lab).loss
                (loss / args.accum).backward()
                tot += loss.item()
                n += 1
                if j % args.accum == 0 or j == len(order):
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    steps += 1
                    if args.max_steps and steps >= args.max_steps:
                        break
            rec = {"epoch": epoch, "optimizer_steps": steps, "train_loss": tot / max(1, n),
                   "seconds": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print("EPOCH", json.dumps(rec), flush=True)
            ck = os.path.join(args.out, "ckpt", "epoch%d" % epoch)
            model.save_pretrained(ck)
            torch.save({"optimizer": opt.state_dict(), "optimizer_steps": steps, "py_rng": random.getstate(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
                       os.path.join(ck, "trainer_state.pt"))
            open(os.path.join(ck, "DONE"), "w").write(json.dumps(rec))
            if args.max_steps and steps >= args.max_steps:
                break
        model.save_pretrained(args.out)
        open(os.path.join(args.out, "TRAIN_DONE"), "w").write(json.dumps({"optimizer_steps": steps}))
    else:
        from safetensors.torch import load_file      # training finished earlier: load the final adapter
        set_peft_model_state_dict(model, load_file(os.path.join(args.out, "adapter_model.safetensors")))
        print("training already done; evaluating the saved adapter", flush=True)

    # ---- validation report (greedy decode + teacher-forced status_probs, inference code path) ----
    model.eval()
    judge = GJ.GoalJudge(args.base, adapter=args.out, gpu=args.gpu, dtype=args.dtype)
    judge.tok, judge.model = tok, model
    vpath = os.path.join(args.out, "validation_predictions.jsonl")
    done = {}
    if os.path.exists(vpath):
        for l in open(vpath, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[r["id"]] = r
    todo = val[:args.max_val] if args.max_val else val
    with open(vpath, "a", encoding="utf-8") as f:
        for s in todo:
            if s["id"] in done:
                continue
            r = judge.assess(s["scenario_text"], s["hist_u"], s["hist_a"])
            row = {"id": s["id"], "source": s["source"], "conversation_id": s["conversation_id"], "t": s["t"],
                   "label": labels[s["id"]]["status"], "pred": r["status"], "unmet": r["unmet"],
                   "parse_ok": r["parse_ok"], "raw": r["raw"], "probs": r["status_probs"],
                   "probs_mass": r["status_probs_mass"], "prompt_tokens": r["prompt_tokens"],
                   "dropped_exchanges": r["dropped_exchanges"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[s["id"]] = row
    rows = [done[s["id"]] for s in todo]
    rep = evaluate_rows(rows, args.gate_macro_f1, args.gate_unparsed)
    rep.update({"fold": args.fold, "split": "validation", "adapter": os.path.abspath(args.out),
                "predictions_sha256": JM.sha256_file(vpath), "smoke": bool(args.max_steps or args.max_val)})
    json.dump(rep, open(os.path.join(args.out, "eval_validation.json"), "w"), indent=1)
    print("VALIDATION", json.dumps({k: v for k, v in rep.items() if k != "by_source"}), flush=True)
    print("JUDGE GATE fold %d (macro-F1 >= %.2f and unparsed < %.1f%%): %s  macro_f1=%s unparsed_rate=%s"
          % (args.fold, args.gate_macro_f1, 100 * args.gate_unparsed, rep["gate"]["result"],
             rep["macro_f1"], rep["unparsed_rate"]), flush=True)
    return 0 if rep["gate"]["result"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
