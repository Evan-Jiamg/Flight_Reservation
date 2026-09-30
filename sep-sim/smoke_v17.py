#!/usr/bin/env python3
"""GPU smoke test of the v17 path (SPEC v17 S14 + fix round 1 B-1 / B-S2 / B-S3 / C-N8 / D-N3 / D-N7, 2026-09-30) with
the REAL models (Qwen3-4B learner + vLLM Planner + Ditto-8B); Task 2 episodes need the gpt-oss server (R0 / ledger).
Uses the Trainer's own methods (no copies), on a SUBSET of the fold (--smoke-convs train_all conversations for the SFT data
and the Task 1 groups, --val-convs validation_all conversations for the probe, one Task 2 scenario x G=2) in a scratch run
dir. Every check below enters `passed`:
  b1   the start policy sha equals the one a fresh subprocess computes with the same seed (seeded LoRA init, B1);
  pend P_end: p_end_batch == end_prob item by item; the fp32 value logits (token_logprobs_fp32) agree with the bf16 path
       within 0.05 nats (the hook really reads the normed lm_head input); a greedy sample's P_end == the probe's;
  sft  one SFT epoch lowers the SFT loss; load_policy(sft_e1) restores the recorded sha (S8);
  ref  u0 saved, ref loaded: trainables / optimizer params / policy sha unchanged, ref frozen, ref on --gpu; the u0
       adapter file is byte-identical to sft_e1's (C-N8); no ref/ in any saved adapter directory;
  mix  ONE learner.update mixing Task 2 samples (stop credit) and Task 1 samples (prefix advantage) + the aux split:
       8 optimizer steps; token alignment: the learner's per-token logp of the sampled tokens vs the vLLM gen_logprobs
       (mean |diff| <= the TIS abort 0.1); afterwards max|ref_logp - u0_logp| < 1e-4 (the ref did not move),
       max|policy_logp - u0_logp| > 0 (the policy did) and kl_step_max > 0 (B-S3);
  gpu  torch.cuda.memory_allocated(i) == 0 for every other visible GPU i after the ref load and after the update (B-1);
  resume  a second Trainer built on the scratch dir (ref loaded at build) -> load_checkpoint (u1) -> optimizer state with
       the ref present, policy sha = u1's;
  base an eval_test_rl-style load_policy(ckpt/sft_e0) while the ref is loaded: sha = the start policy's, ref unchanged.
Logs: the value round-trip failure rate (D-N7), the logits dtype, timings. Writes <out>/smoke_v17.json (a fresh dir).

  smoke_v17.py --fold 2 --planner-path Q4 --gpu 1 --out /tmp2/.../runs/smoke_v17 [--smoke-convs 3] [--val-convs 2]
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
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


def ref_dirs(root):
    return [os.path.join(dp, d) for dp, ds, _ in os.walk(root) for d in ds if d == RA.REF_ADAPTER]


def other_gpu_bytes(gpu):
    import torch
    return {i: int(torch.cuda.memory_allocated(i)) for i in range(torch.cuda.device_count()) if i != gpu}


START_SHA_CODE = r'''
import json, sys
sys.path.insert(0, %r)
import torch
torch.cuda.set_device(%d)
import rl_algos as RA
from task2_env import PlannerLM
p = PlannerLM(%r, gpu=%d, dtype="bfloat16")
m = RA.setup_policy(p, %d)
print("SHA " + RA.tensor_sha(RA.lora_state(m)))
'''


def main(argv=None):
    import torch
    argv = list(sys.argv[1:] if argv is None else argv)
    n_conv = pop_int(argv, "--smoke-convs", 3)
    n_val = pop_int(argv, "--val-convs", 2)
    a = T.parse_args(argv + ["--ablation", "smoke-v17", "--scenarios-per-update", "1", "--G", "2"])
    if os.path.exists(os.path.join(a.out, "ckpt", "LATEST.json")) or os.path.exists(os.path.join(a.out, "sft_examples.jsonl")):
        raise SystemExit("%s already holds a run; use a fresh scratch dir" % a.out)
    tr = T.Trainer(a)
    t0 = time.time()
    tr.build()                                           # torch.cuda.set_device(--gpu) inside (B-1)
    res = {"build_s": round(time.time() - t0, 1), "checks": {}, "logs": {}}
    chk, logs = res["checks"], res["logs"]
    lr = tr.learner
    gpu = int(a.gpu)
    # the subset (after build: p_h and the few-shot pool keep the whole fold)
    tr.split = dict(tr.split, train_all=sorted(tr.split["train_all"])[:n_conv],
                    validation_all=sorted(tr.split["validation_all"])[:n_val])
    a.task1_convs = n_conv
    # ---- b1: the start policy is reproducible (fresh process, same seed)
    start_sha = lr.policy_sha()
    code = START_SHA_CODE % (HERE, gpu, a.planner_path, gpu, T.seed_of(a.seed, "lora_init"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1800)
    sub = [l[4:] for l in out.stdout.splitlines() if l.startswith("SHA ")]
    logs["start_policy_sha"], logs["start_policy_sha_subprocess"] = start_sha, (sub[-1] if sub else out.stderr[-500:])
    chk["b1_start_policy_reproducible"] = bool(sub) and sub[-1] == start_sha
    # ---- SFT data (start policy served as sft_e0)
    tr.save_sft_candidate(0)
    t1 = time.time()
    examples, emeta = tr.sft_examples(start_sha)
    logs["sft_data"] = {"seconds": round(time.time() - t1, 1), "counts": emeta["counts"], "gen_adapter": emeta["gen_adapter"]}
    n_ok = emeta["counts"]["n_examples"]
    logs["value_roundtrip_fail_rate_sft"] = (emeta["counts"]["value_roundtrip_fail"] / n_ok) if n_ok else None
    # ---- P_end
    items = [{"prompt_ids": e["prompt_ids"], "prefix_ids": e["prefix_ids"], "target_true": e["target_true"],
              "target_false": e["target_false"]} for e in examples[:12]]
    batch, single = lr.p_end_batch(items), [lr.end_prob(x) for x in items]
    chk["pend_batch_equals_single"] = bool(items) and max(abs(x - y) for x, y in zip(batch, single)) < 1e-9
    diffs = []
    with torch.no_grad():
        for x in items:
            ctx = list(x["prompt_ids"]) + list(x["prefix_ids"])
            a32 = RA.token_logprobs_fp32(lr.model, ctx, x["target_true"])
            a16, _ = RA.token_logprobs(lr.model, ctx, x["target_true"], 1.0)
            diffs.append(float((a32 - a16).abs().max()))
    logs["value_logit_fp32_vs_bf16_maxdiff"] = max(diffs) if diffs else None
    logs["model_dtype"] = str(next(p for n, p in lr.model.named_parameters() if not p.requires_grad).dtype)
    chk["pend_fp32_path_consistent"] = bool(diffs) and max(diffs) < 0.05
    cid = sorted(tr.split["train_all"])[0]
    prompts = tr.env.task1_prompts(cid)
    probe_vs = []
    for pr in prompts[1:3]:
        pb = tr.env.task1_end_probe(cid, pr["t"], pr["user_prompt"], pr["real_final"])
        g = tr.env.task1_sample(cid, pr["t"], pr["user_prompt"], pr["real_final"], 1, 0.0, 1.0, 0)[0]
        if pb["valid"] and g.get("mask_ok"):
            tr.score_task1([g], pr["real_final"])
            probe_vs.append(abs(lr.end_prob(pb) - g["p_end"]))
    chk["pend_greedy_sample_equals_probe"] = bool(probe_vs) and max(probe_vs) < 1e-6
    # ---- one SFT epoch, candidate sft_e1, load_policy + sha (S8)
    rows = [tr.sft_probe(1, 0, examples, [])]
    lr.sft_begin(a.sft_lr)
    steps = []
    t3 = time.time()
    order = list(range(len(examples)))
    random.Random(T.seed_of(a.seed, "sft_shuffle", 1)).shuffle(order)
    for i in range(0, len(order), T.SFT_BATCH):
        steps.append(lr.sft_step([dict(examples[j], weight=1.0) for j in order[i:i + T.SFT_BATCH]]))
    lr.sft_end()
    tr.save_sft_candidate(1)
    rows.append(tr.sft_probe(1, 1, examples, steps))
    logs["sft_epoch"] = {"seconds": round(time.time() - t3, 1), "steps": len(steps),
                         "loss_first": steps[0]["loss"] if steps else None, "loss_last": steps[-1]["loss"] if steps else None,
                         "val": {r["epoch"]: r["val"] for r in rows}}
    chk["sft_train_nll_decreased"] = rows[1]["train_p_end_final_mean"] is not None and \
        (rows[1]["train_p_end_final_mean"] - rows[1]["train_p_end_nonfinal_mean"]) > \
        (rows[0]["train_p_end_final_mean"] - rows[0]["train_p_end_nonfinal_mean"])
    e1_sha = rows[1]["policy_sha"]
    lr.load_policy(tr.sft_dir(0))
    lr.load_policy(tr.sft_dir(1))
    chk["sft_load_policy_sha"] = lr.policy_sha() == e1_sha
    # ---- u0 + ref (B4, B-1, C-N8)
    tr.sft_info = {"chosen_epoch": 1, "policy_sha": e1_sha, "smoke": True}
    names0, opt0 = lr.trainable_names(), [id(p) for g in lr.optimizer.param_groups for p in g["params"]]
    tr.save_checkpoint(0, None)
    lr.load_ref(os.path.join(tr.ckpt_dir(0), "adapter"))
    chk["ref_trainables_unchanged"] = lr.trainable_names() == names0
    chk["ref_optimizer_params_unchanged"] = [id(p) for g in lr.optimizer.param_groups for p in g["params"]] == opt0
    chk["ref_policy_sha_unchanged"] = lr.policy_sha() == e1_sha
    chk["ref_params_frozen"] = not any(p.requires_grad for n, p in lr.model.named_parameters() if ".ref." in n)
    chk["ref_on_train_gpu"] = {str(p.device) for n, p in lr.model.named_parameters() if ".ref." in n} == {"cuda:%d" % gpu}
    chk["u0_adapter_bytes_equal_chosen"] = open(os.path.join(tr.ckpt_dir(0), "adapter", "adapter_model.safetensors"), "rb").read() \
        == open(os.path.join(tr.sft_dir(1), "adapter", "adapter_model.safetensors"), "rb").read()
    og = other_gpu_bytes(gpu)
    logs["other_gpu_bytes_after_ref"] = og
    chk["no_other_gpu_memory_after_ref"] = all(v == 0 for v in og.values())
    rs0 = ref_sha(lr.model)
    # ---- one mixed update: Task 2 (stop credit) + Task 1 (prefix) + aux
    tr.sync_generation_policy(0)
    t5 = time.time()
    groups_all = tr.rollouts(1)
    t2s = []
    for grp in groups_all:
        clean = [r for r in grp if r["episode"]["clean"]]
        for j, r in enumerate(clean):
            for s_ in RA.episode_samples(r["episode"], policy_version=0):
                s_.update(ret=0.0, adv=0.1 * (1 if j % 2 else -1), adv_stop=0.2 * (1 if j % 2 else -1), source="task2")
                t2s.append(s_)
    t1rows = tr.task1_rollouts(1)
    t1s, cnt, _ = tr.task1_samples(t1rows, 0, lr.policy_sha())
    aux = tr.aux_examples(t1rows)
    samples = t2s + t1s
    orders = []
    for ep in range(int(tr.acfg["epochs"])):
        o = list(range(len(aux)))
        random.Random(T.seed_of(a.seed, "aux_mb", 1, ep)).shuffle(o)
        orders.append(o)
    probe = next(s_ for s_ in samples)                       # a fixed sample for the logp comparisons
    probe = {k: probe[k] for k in ("prompt_ids", "gen_ids", "temperature")}
    u0_lp = lr._logp(dict(probe), grad=False)[0].float().cpu()
    # token alignment: learner logp of the sampled tokens vs the vLLM behaviour log-probs
    dev_ = []
    for s_ in samples[:8]:
        if s_.get("behav_logp") is not None:
            lp_ = lr._logp(s_, grad=False)[0].float().cpu()
            dev_.append(float((lp_ - torch.tensor(s_["behav_logp"])).abs().mean()))
    logs["token_alignment_mean_abs"] = dev_
    chk["token_alignment_within_tis_tol"] = bool(dev_) and max(dev_) <= a.behav_mismatch_abort
    try:
        st = lr.update(samples, tr.cfg, seed=T.seed_of(a.seed, "update", 1), aux=aux or None,
                       aux_orders=orders if aux else None, mismatch_abort=a.behav_mismatch_abort)
    except RA.MismatchAbort as e:
        st = {"mismatch_abort": e.value}
    logs["update"] = {"seconds": round(time.time() - t5, 1), "n_task2": len(t2s), "n_task1": len(t1s), "counts": cnt,
                      "aux_n": len(aux), **{k: st.get(k) for k in (
                          "optimizer_steps", "n_minibatches", "rl_grad_norm", "rl_grad_norm_max", "aux_grad_norm",
                          "aux_grad_norm_max", "kl", "kl_step_mean", "kl_step_max", "clip_frac_step_max",
                          "ratio_init_maxdev", "behav_mismatch_mean", "aux_p_correct_before", "mismatch_abort")}}
    nsc = sum(1 for r in t1rows for x in r["samples"] if x.get("mask_ok"))
    logs["value_roundtrip_fail_rate_task1"] = (sum(1 for r in t1rows for x in r["samples"]
                                                   if x.get("value_roundtrip_ok") is False) / nsc) if nsc else None
    chk["update_mixed_sources"] = bool(t2s) and bool(t1s)
    chk["update_steps_8"] = st.get("optimizer_steps") == 8 and len(samples) >= 4
    ref_lp = lr._logp(dict(probe), grad=False, reference=True)[0].float().cpu()
    pol_lp = lr._logp(dict(probe), grad=False)[0].float().cpu()
    logs["ref_vs_u0_maxdiff"] = float((ref_lp - u0_lp).abs().max())
    logs["policy_vs_u0_maxdiff"] = float((pol_lp - u0_lp).abs().max())
    chk["ref_equals_u0_after_update"] = logs["ref_vs_u0_maxdiff"] < 1e-4
    chk["policy_moved_after_update"] = logs["policy_vs_u0_maxdiff"] > 0
    chk["kl_step_max_positive"] = (st.get("kl_step_max") or 0.0) > 0
    chk["ref_unchanged_by_update"] = ref_sha(lr.model) == rs0
    chk["update_trainables_unchanged"] = lr.trainable_names() == names0
    og = other_gpu_bytes(gpu)
    logs["other_gpu_bytes_after_update"] = og
    chk["no_other_gpu_memory_after_update"] = all(v == 0 for v in og.values())
    tr.update_done = 1
    tr.save_checkpoint(1, None)
    u1_sha = lr.policy_sha()
    # ---- resume build: a second Trainer on the same dir (ref at build) -> load_checkpoint(u1) with the ref present
    tr2 = T.Trainer(a)
    tr2.build()
    chk["resume_ref_loaded_at_build"] = tr2.learner.has_ref
    chk["resume_load_checkpoint"] = tr2.load_checkpoint() and tr2.learner.policy_sha() == u1_sha
    chk["resume_optimizer_state_loaded"] = len(tr2.learner.optimizer.state_dict()["state"]) > 0
    # ---- eval_test_rl-style base load while the ref is loaded
    tr2.learner.load_policy(tr2.sft_dir(0))
    chk["base_load_with_ref"] = tr2.learner.policy_sha() == start_sha and ref_sha(tr2.learner.model) == rs0
    chk["no_ref_dir_in_checkpoints"] = ref_dirs(tr.ckpt_root) == []
    og = other_gpu_bytes(gpu)
    logs["other_gpu_bytes_end"] = og
    chk["no_other_gpu_memory_end"] = all(v == 0 for v in og.values())
    res["total_s"] = round(time.time() - t0, 1)
    res["passed"] = all(v is True for v in chk.values())
    res["failed_checks"] = sorted(k for k, v in chk.items() if v is not True)
    json.dump(res, open(os.path.join(a.out, "smoke_v17.json"), "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    print("SMOKE V17 %s" % ("PASSED" if res["passed"] else "FAILED %s" % res["failed_checks"]))
    return res


if __name__ == "__main__":
    main()
