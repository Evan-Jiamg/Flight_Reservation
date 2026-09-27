#!/usr/bin/env python3
"""GPU smoke test of the v16 Task 1 path with the REAL models (Qwen3-4B learner + vLLM Planner + Ditto), no R0 needed.

Uses the Trainer's own methods (no copies): task1_eval_row (validation Task 1 + teacher-forced end probabilities,
v16 item 7), task1_rollouts (task1_G 8, task1_convs 8, refill; items 1-2), task1_samples + aux_examples (Dr. GRPO,
aux floor; items 3-4) and one learner.update on the Task 1 samples + stop supervision, to measure the step-size
statistics the audit asked for (rl_grad_norm, aux_grad_norm, grad_norm, adv_abs_mean, and the same advantages with
the old std normalisation for comparison). Order: validation Task 1 first (policy 0 on both the vLLM server and the
learner), then the Task 1 groups, then one update. Writes <out>/smoke_v16.json. The out dir is a scratch run dir.

  smoke_v16.py --fold 2 --planner-path Q4 --gpu 0 --out /tmp2/.../runs/smoke_v16 [--val-convs 2]
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rl_algos as RA  # noqa: E402
import task1_stop as T1  # noqa: E402
import train_planner_rl as T  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    nval = 2
    if "--val-convs" in argv:
        i = argv.index("--val-convs")
        nval = int(argv[i + 1])
        del argv[i:i + 2]
    a = T.parse_args(argv)
    if os.path.exists(os.path.join(a.out, "ckpt", "LATEST.json")):
        raise SystemExit("%s already holds checkpoints; use a fresh scratch dir" % a.out)
    tr = T.Trainer(a)
    t0 = time.time()
    tr.build()
    res = {"build_s": round(time.time() - t0, 1), "args": {k: v for k, v in vars(a).items() if k in (
        "task1_G", "task1_convs", "stop_sup_floor", "t1_trigger_margin", "stop_sup_weight", "planner_backend")},
           "algo_cfg_grpo_std_norm": tr.acfg["grpo_std_norm"],
           "controller_w_dist_bounds": tr.controller.opt["bounds"]["w_dist"] if hasattr(tr.controller, "opt") else None}
    tr.save_checkpoint(0, None)
    psha = tr.learner.policy_sha()
    tr.sync_generation_policy(0)
    # 1. validation Task 1 (item 7)
    t1 = time.time()
    rows = [tr.task1_eval_row(0, psha, cid) for cid in sorted(tr.split["validation"])[:nval]]
    pts = [p for r in rows for p in r["end_probs"]]
    agree = [p for p in pts if p["valid"]]
    res["val_task1"] = {
        "seconds": round(time.time() - t1, 1), "n_conversations": len(rows), "n_points": len(pts),
        "n_valid": len(agree), "metrics": T1.task1_prob_metrics(pts) if pts else None,
        "term_metrics": T1.task1_stop_metrics([r["task1"] for r in rows]) if rows else None,
        # a scored point's greedy decision should mostly agree with p_end > 0.5 (same prompt, same policy)
        "greedy_vs_p_agree": (sum((p["p_end"] > 0.5) == p["greedy_end"] for p in agree) / len(agree)) if agree else None,
        "points": [{k: p[k] for k in ("t", "real_final", "p_end", "valid", "decision_valid", "greedy_end")} for p in pts]}
    # 2. Task 1 groups with refill (items 1-2)
    t2 = time.time()
    t1rows = tr.task1_rollouts(1)
    res["task1_groups"] = {"seconds": round(time.time() - t2, 1), "n_rows": len(t1rows),
                           "samples_per_row": sorted({len(r["samples"]) for r in t1rows}),
                           "valid_rate": (sum(x["decision_valid"] for r in t1rows for x in r["samples"])
                                          / max(1, sum(len(r["samples"]) for r in t1rows))),
                           **{k: v for k, v in tr.t1_refill.items() if k not in ("groups",)}}
    # 3. one update on the Task 1 samples + stop supervision (items 3-4), step-size statistics
    samples, skipped = tr.task1_samples(t1rows, 0, psha)
    w_aux = tr.aux_weight(1, tr.cfg)
    aux = tr.aux_examples(t1rows, w_aux)
    rs = [[x["reward"] for x in r["samples"]] for r in t1rows]
    old, _ = RA.advantages_for_groups(rs, "grpo", RA.algo_cfg(grpo_std_norm=True))
    new, _ = RA.advantages_for_groups(rs, "grpo", tr.acfg)
    mean_abs = lambda xs: (sum(abs(v) for a_ in xs if a_ for v in a_) / max(1, sum(len(a_) for a_ in xs if a_)))
    bad = sorted({s_.get("gen_adapter") for s_ in samples if s_.get("gen_adapter") != tr.gen_name})
    res["task1_groups"]["off_policy_adapters"] = bad          # must be [] (as one_update asserts)
    t3 = time.time()
    try:
        stats = tr.learner.update(samples, tr.cfg, seed=T.seed_of(a.seed, "update", 1), aux=aux or None,
                                  mismatch_abort=a.behav_mismatch_abort)
    except RA.MismatchAbort as e:                             # keep what was measured before the abort
        stats = {"mismatch_abort": e.value}
    res["update"] = {"seconds": round(time.time() - t3, 1), "n_samples": len(samples), "groups_skipped": skipped,
                     "aux_weight": w_aux, "aux_n": len(aux), "adv_abs_mean_dr_grpo": mean_abs(new),
                     "adv_abs_mean_if_std_norm": mean_abs(old),
                     "mismatch_abort": stats.get("mismatch_abort"),
                     **{k: stats.get(k) for k in ("rl_grad_norm", "aux_grad_norm", "grad_norm", "kl", "loss", "n_tokens",
                                                  "behav_mismatch_mean", "tis_w_mean", "tis_capped_frac", "aux_p_correct_before",
                                                  "aux_p_correct_end", "optimizer_steps")}}
    res["total_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out, "smoke_v16.json"), "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "val_task1"}, indent=1, default=str))
    print(json.dumps({k: v for k, v in res["val_task1"].items() if k != "points"}, indent=1, default=str))
    return res


if __name__ == "__main__":
    main()
