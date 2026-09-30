#!/usr/bin/env python3
"""GPU smoke test of the v17 path (SPEC v17 S14, user 2026-09-30) with the REAL models (Qwen3-4B learner + vLLM Planner +
Ditto-8B), no R0 needed (Task 1 only). Uses the Trainer's own methods (no copies), on a SUBSET of the fold's conversations
(--smoke-convs train_all conversations for the SFT data and the Task 1 groups, --val-convs validation_all conversations
for the probe) in a scratch run dir:
  1. the start policy saved as ckpt/sft_e0 and served (B3); SFT examples + base P_end on the subset (sft_examples);
  2. P_end: the learner's p_end_batch on every example equals end_prob item by item, and on a greedy plan equals the
     validation probe's value (task1_end_probe + end_prob);
  3. one SFT epoch (sft_begin / sft_step, value_nll_loss), candidate ckpt/sft_e1, its validation_all probe (sft_probe);
  4. u0 saved (save_checkpoint(0)), the ref adapter loaded (load_ref): the trainable set, the optimizer's parameters and
     the policy sha are unchanged; a ref forward (the KL reference) leaves them unchanged too; every ref parameter is
     frozen; the saved adapter directories have no ref/;
  5. one update on the Task 1 groups of the subset (task1_rollouts -> Brier rewards, task1_samples, aux_examples):
     learner.update with the aux split -> optimizer_steps == 8 when there are >= 4 samples; step statistics, KL to ref;
     the ref parameters are unchanged by the update; ckpt u1 has no ref/.
Writes <out>/smoke_v17.json. The out dir must be a fresh scratch dir.

  smoke_v17.py --fold 2 --planner-path Q4 --gpu 0 --out /tmp2/.../runs/smoke_v17 [--smoke-convs 3] [--val-convs 2]
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rl_algos as RA  # noqa: E402
import train_planner_rl as T  # noqa: E402


def pop_int(argv, flag, default):
    if flag in argv:
        i = argv.index(flag)
        v = int(argv[i + 1])
        del argv[i:i + 2]
        return v
    return default


def ref_sha(model):
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters(), key=lambda x: x[0]):
        if (".%s." % RA.REF_ADAPTER) in n:
            h.update(n.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def no_ref_dirs(root):
    return [os.path.join(dp, d) for dp, ds, _ in os.walk(root) for d in ds if d == RA.REF_ADAPTER]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    n_conv = pop_int(argv, "--smoke-convs", 3)
    n_val = pop_int(argv, "--val-convs", 2)
    a = T.parse_args(argv + ["--ablation", "smoke-v17"])
    if os.path.exists(os.path.join(a.out, "ckpt", "LATEST.json")) or os.path.exists(os.path.join(a.out, "sft_examples.jsonl")):
        raise SystemExit("%s already holds a run; use a fresh scratch dir" % a.out)
    tr = T.Trainer(a)
    t0 = time.time()
    tr.build()
    res = {"build_s": round(time.time() - t0, 1), "checks": {}}
    chk = res["checks"]
    lr = tr.learner
    # the subset (after build: p_h and the few-shot pool keep the whole fold)
    tr.split = dict(tr.split, train_all=sorted(tr.split["train_all"])[:n_conv],
                    validation_all=sorted(tr.split["validation_all"])[:n_val])
    a.task1_convs = n_conv
    # ---- 1. start policy + SFT data
    start_sha = lr.policy_sha()
    tr.save_sft_candidate(0)
    t1 = time.time()
    examples, emeta = tr.sft_examples(start_sha)
    res["sft_data"] = {"seconds": round(time.time() - t1, 1), "counts": emeta["counts"], "gen_adapter": emeta["gen_adapter"]}
    # ---- 2. P_end consistency
    items = [{"prompt_ids": e["prompt_ids"], "prefix_ids": e["prefix_ids"], "target_true": e["target_true"],
              "target_false": e["target_false"]} for e in examples[:12]]
    batch = lr.p_end_batch(items)
    single = [lr.end_prob(x) for x in items]
    chk["p_end_batch_equals_end_prob"] = max(abs(x - y) for x, y in zip(batch, single)) if items else None
    cid = sorted(tr.split["train_all"])[0]
    prompts = tr.env.task1_prompts(cid)
    probe_vs = []
    for pr in prompts[1:3]:
        pb = tr.env.task1_end_probe(cid, pr["t"], pr["user_prompt"], pr["real_final"])
        g = tr.env.task1_sample(cid, pr["t"], pr["user_prompt"], pr["real_final"], 1, 0.0, 1.0, 0)[0]
        if pb["valid"] and g.get("mask_ok"):
            tr.score_task1([g], pr["real_final"])
            probe_vs.append(abs(lr.end_prob(pb) - g["p_end"]))
    chk["greedy_probe_vs_sample_p_end_maxdiff"] = max(probe_vs) if probe_vs else None
    # ---- 3. one SFT epoch + its probe
    rows = [tr.sft_probe(1, 0, examples, [])]
    lr.sft_begin(a.sft_lr)
    steps = []
    t3 = time.time()
    for i in range(0, len(examples), T.SFT_BATCH):
        steps.append(lr.sft_step([dict(x, weight=1.0) for x in examples[i:i + T.SFT_BATCH]]))
    lr.sft_end()
    tr.save_sft_candidate(1)
    rows.append(tr.sft_probe(1, 1, examples, steps))
    res["sft_epoch"] = {"seconds": round(time.time() - t3, 1), "steps": len(steps),
                        "loss_first": steps[0]["loss"] if steps else None, "loss_last": steps[-1]["loss"] if steps else None,
                        "val": {r["epoch"]: r["val"] for r in rows},
                        "train_p_end": {r["epoch"]: [r["train_p_end_final_mean"], r["train_p_end_nonfinal_mean"]] for r in rows}}
    # ---- 4. u0 + ref adapter
    tr.sft_info = {"chosen_epoch": 1, "policy_sha": lr.policy_sha(), "smoke": True}
    names0, opt0, sha0 = lr.trainable_names(), [id(p) for g in lr.optimizer.param_groups for p in g["params"]], lr.policy_sha()
    tr.save_checkpoint(0, None)
    lr.load_ref(os.path.join(tr.ckpt_dir(0), "adapter"))
    chk["ref_trainables_unchanged"] = lr.trainable_names() == names0
    chk["ref_optimizer_params_unchanged"] = [id(p) for g in lr.optimizer.param_groups for p in g["params"]] == opt0
    chk["ref_policy_sha_unchanged"] = lr.policy_sha() == sha0
    chk["ref_params_frozen"] = not any(p.requires_grad for n, p in lr.model.named_parameters() if ".ref." in n)
    s0 = {"prompt_ids": examples[0]["prompt_ids"], "gen_ids": examples[0]["prefix_ids"] + examples[0]["target_ids"],
          "temperature": 1.0}
    lp = lr._logp(s0, grad=False)[0]
    lr_ = lr._logp(s0, grad=False, reference=True)[0]
    chk["ref_forward_trainables_unchanged"] = lr.trainable_names() == names0
    chk["ref_equals_policy_at_u0_maxdiff"] = float((lp - lr_).abs().max())       # u0 = ref: ~0
    rs0 = ref_sha(lr.model)
    # ---- 5. one update on the Task 1 groups of the subset
    tr.sync_generation_policy(0)
    t5 = time.time()
    t1rows = tr.task1_rollouts(1)
    samples, cnt, _ = tr.task1_samples(t1rows, 0, lr.policy_sha())
    aux = tr.aux_examples(t1rows)
    orders = []
    for ep in range(int(tr.acfg["epochs"])):
        o = list(range(len(aux)))
        random.Random(T.seed_of(a.seed, "aux_mb", 1, ep)).shuffle(o)
        orders.append(o)
    bad = sorted({s_.get("gen_adapter") for s_ in samples if s_.get("gen_adapter") != tr.gen_name})
    try:
        st = lr.update(samples, tr.cfg, seed=T.seed_of(a.seed, "update", 1), aux=aux or None,
                       aux_orders=orders if aux else None, mismatch_abort=a.behav_mismatch_abort)
    except RA.MismatchAbort as e:
        st = {"mismatch_abort": e.value}
    res["update"] = {"seconds": round(time.time() - t5, 1), "n_rows": len(t1rows), "n_samples": len(samples),
                     "counts": cnt, "aux_n": len(aux), "off_policy_adapters": bad,
                     "status": {k: sum(1 for r in t1rows for x in r["samples"] if x["status"] == k)
                                for k in ("valid", "invalid", "dropped")},
                     **{k: st.get(k) for k in ("optimizer_steps", "n_minibatches", "rl_grad_norm", "rl_grad_norm_max",
                                               "aux_grad_norm", "aux_grad_norm_max", "kl", "kl_step_mean", "kl_step_max",
                                               "clip_frac_step_max", "ratio_init_maxdev", "behav_mismatch_mean",
                                               "aux_p_correct_before", "mismatch_abort")}}
    chk["update_steps_8"] = (st.get("optimizer_steps") == 8) if len(samples) >= 4 else "fewer than 4 samples"
    chk["ref_unchanged_by_update"] = ref_sha(lr.model) == rs0
    chk["update_trainables_unchanged"] = lr.trainable_names() == names0
    tr.save_checkpoint(1, None)
    chk["no_ref_dir_in_checkpoints"] = no_ref_dirs(tr.ckpt_root) == []
    res["total_s"] = round(time.time() - t0, 1)
    res["passed"] = all(v is True or (isinstance(v, float) and v < 1e-4) for k, v in chk.items()
                        if not isinstance(v, str) and v is not None)
    json.dump(res, open(os.path.join(a.out, "smoke_v17.json"), "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    main()
