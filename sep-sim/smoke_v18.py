#!/usr/bin/env python3
"""GPU smoke test of the SPEC v18 path (ops/SPEC_v18_multiobj_rerank.md §16 step 5) with the REAL models (Qwen3-4B learner
+ Planner vLLM + Ditto-8B + the reranker; Task 2 episodes need the gpt-oss server). Uses the Trainer's own v18 methods (no
copies) on a SUBSET of the fold in a scratch run dir: the init stage (v17 u0 copied, ref = u0), the Task 1 validation of
u0 on --val-convs validation_all conversations, ONE update (Task 2: --scenarios-per-update x --G episodes; Task 1:
--smoke-convs train_all conversations at t = 1..n), then the logged rows re-checked by verify_v18 (rewards, scales /
kappa, advantages, tau) -- the SAME recomputation the verifier runs on a full run.

Smoke-only settings (named ablation "smoke-v18", recorded): fewer scenarios / conversations, and the spread-group freeze
thresholds lowered to 1 so that the small batch can be measured (the full run keeps 3 / 2).

Checks (True / False / "not measurable"; an essential check must be True):
  init        u0 policy sha == the v17 u0's, the u0 adapter a byte copy (plain files), ref loaded = u0 (ref logp == policy
              logp before the update), no ref/ directory saved
  pend        p_end_batch == end_prob item by item (t >= 2 samples)
  t1          at least one turn-1 sample is valid with its value tokens located (decode-checked), no turn-1 P_end, no
              turn-1 aux example, the plan mask excludes the value tokens
  components  r_stop, r_act, r_len, r_fmt each computed on some sample; r_turn / r_fmt2 on the Task 2 groups
  scales      adv_scales.json written with kappa > 0; the u1 token-weighted tau equals the targets 0.0197 / 0.0762
              (by construction on the measurement batch) -- within 1e-6
  recompute   verify_v18 on the smoke dir: rl.v18_reward / v18_task1 / v18_advantage / v18_tau / v18_scales without a
              failure (and with checks)
  rerank      with selector borda_rerank: reranked selections logged and every one recomputed (§6.4); else not measurable
  update      8 optimizer steps; kl_step_max > 0; the policy moved; the ref unchanged; the trainables unchanged; vLLM vs
              learner token alignment within the TIS abort; the §4.3 alarms reported
  gpu/budget  as smoke_v17 (no memory on another GPU, no context on another GPU, the budget still reserved at the end)
  resume      scales_v18 at update 2 reads adv_scales.json back (fingerprint of the logged update-1 rows) unchanged
Writes <out>/smoke_v18.json.

  smoke_v18.py --fold 2 --planner-path Q4 --gpu 1 --out RUNS/smoke_v18_f2 --init-adapter RUN17/ckpt/u00000 \
      --act-labels L/act_labels_train_f2.jsonl --act-labels-val L/act_labels_val_f2.jsonl --reranker L/reranker_v18_f2.json
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rl_algos as RA  # noqa: E402
import smoke_v17 as S17  # noqa: E402
import train_planner_rl as T  # noqa: E402
import v18_rules as V18  # noqa: E402

NM = S17.NM
GIB = S17.GIB
ESSENTIAL = ("init_policy_sha", "init_adapter_copied", "ref_equals_policy_at_u0", "no_ref_dir_in_checkpoints",
             "pend_batch_equals_single", "t1_value_mask_located", "t1_no_pend_no_aux", "t1_plan_mask_excludes_value",
             "components_task1", "components_task2", "scales_written", "tau_u1_equals_targets",
             "recompute_reward", "recompute_task1", "recompute_advantage", "recompute_tau", "recompute_scales",
             "update_steps_8", "kl_step_max_positive", "policy_moved_after_update", "ref_unchanged_by_update",
             "update_trainables_unchanged", "token_alignment_within_tis_tol", "no_other_gpu_memory_end",
             "no_context_on_other_gpu", "memory_under_budget", "budget_reserved_at_build", "budget_still_reserved_at_end",
             "resume_reads_scales", "validation_task1_metrics")


def main(argv=None):
    import torch
    argv = list(sys.argv[1:] if argv is None else argv)
    n_conv = S17.pop_int(argv, "--smoke-convs", 3)
    n_val = S17.pop_int(argv, "--val-convs", 2)
    a = T.parse_args(argv + ["--spec", "v18", "--ablation", "smoke-v18", "--scenarios-per-update", "3",
                             "--task1-convs", str(n_conv), "--adv-min-spread-groups", "1",
                             "--adv-min-spread-groups-task2", "1"])
    if os.path.exists(os.path.join(a.out, "ckpt", "LATEST.json")):
        raise SystemExit("%s already holds a run; use a fresh scratch dir" % a.out)
    gpu = int(a.gpu)
    tr = T.Trainer(a)
    t0 = time.time()
    tr.build()
    res = {"build_s": round(time.time() - t0, 1), "checks": {}, "logs": {}, "memory_gib": {}}
    chk, logs, mem = res["checks"], res["logs"], res["memory_gib"]

    def phase(name):
        mem[name] = round(torch.cuda.max_memory_reserved(gpu) / GIB, 2)

    target = (float(a.gpu_budget_gib) - 0.5) * GIB
    budget_on = float(a.gpu_budget_gib) > 0
    chk["budget_reserved_at_build"] = (torch.cuda.memory_reserved(gpu) >= target) if budget_on else NM
    phase("build")
    lr = tr.learner
    tr.split = dict(tr.split, train_all=sorted(tr.split["train_all"])[:n_conv],
                    validation_all=sorted(tr.split["validation_all"])[:n_val])
    # ---- init (§2)
    T.append_jsonl(tr.p_meta, tr.meta("start"))
    tr.init_stage_v18()
    phase("init")
    st0 = json.load(open(os.path.join(tr.ckpt_dir(0), "state.json")))
    chk["init_policy_sha"] = st0["policy_sha"] == a.init_policy_sha == lr.policy_sha()
    ad0 = os.path.join(tr.ckpt_dir(0), "adapter")
    chk["init_adapter_copied"] = (T.sha_path(ad0) == T.sha_path(os.path.join(a.init_adapter, "adapter"))
                                  and all(not os.path.islink(os.path.join(ad0, f)) and os.stat(os.path.join(ad0, f)).st_nlink == 1
                                          for f in os.listdir(ad0)))
    names0 = lr.trainable_names()
    rs0 = S17.ref_sha(lr.model)
    # ---- validation u0, Task 1 only on the subset (the v18 metrics with the validation labels)
    v0 = tr.validate_v18(0, task2=False)
    m0 = v0.get("v18") or {}
    logs["validation_u0"] = m0
    chk["validation_task1_metrics"] = all(m0.get(k) is not None for k in ("stop_score", "act_score", "len_score",
                                                                          "act_entropy", "n_invalid"))
    phase("validate")
    # ---- one update (the trainer's own v18 update)
    t5 = time.time()
    row = tr.one_update_v18(1)
    phase("update")
    st = row["learner_stats"]
    logs["update"] = {"seconds": round(time.time() - t5, 1), "n_samples": row["n_samples"],
                      **{k: st.get(k) for k in ("optimizer_steps", "kl", "kl_step_max", "rl_grad_norm", "aux_grad_norm",
                                                "behav_mismatch_mean", "tis_w_mean", "aux_n")},
                      "tau": row["tau"], "kappa": row["kappa"], "scales": row["scales"], "frozen": row["frozen"],
                      "alarms": row["alarms"], "z_clip_frac": row["z_clip_frac"],
                      "task1": {k: row["task1_stats"].get(k) for k in ("n_points", "n_invalid", "n_dropped",
                                                                       "n_t1_value_mismatch", "components_n",
                                                                       "spread_groups", "skips")},
                      "task2": {k: row["task2_stats"].get(k) for k in ("n_groups", "n_excluded", "spread_groups",
                                                                       "rerank_changed_frac", "coverage_diag_missing")}}
    chk["update_steps_8"] = st.get("optimizer_steps") == 8 if row["n_samples"] >= 4 else NM
    chk["kl_step_max_positive"] = (st.get("kl_step_max") or 0.0) > 0
    chk["ref_unchanged_by_update"] = S17.ref_sha(lr.model) == rs0
    chk["update_trainables_unchanged"] = lr.trainable_names() == names0
    chk["policy_moved_after_update"] = lr.policy_sha() != st0["policy_sha"]
    mm = st.get("behav_mismatch_mean")
    chk["token_alignment_within_tis_tol"] = (mm <= a.behav_mismatch_abort) if mm is not None else NM
    # ---- Task 1 turn 1 (§3.1.5) and the components
    t1rows = [r for r in T.read_jsonl(tr.p_roll_t1) if r["update"] == 1]
    t1s = [x for r in t1rows if r["t"] == 1 for x in r["samples"]]
    ok_vm = [x for x in t1s if x.get("status") == "valid"]
    chk["t1_value_mask_located"] = bool(ok_vm) and all(x["planner_gen"].get("value_mask") and 1 in x["planner_gen"]["value_mask"]
                                                       for x in ok_vm)
    logs["t1_samples"] = {"n": len(t1s), "valid": len(ok_vm),
                          "dropped_value_mismatch": sum(1 for x in t1s if x.get("drop_reason") == "t1_value_mismatch"),
                          "invalid": sum(1 for x in t1s if x.get("status") == "invalid")}
    aux_pts = row["task1_stats"].get("aux_points") or []
    chk["t1_no_pend_no_aux"] = all(x.get("p_end") is None for x in t1s) and not any(p[1] == 1 for p in aux_pts)
    pm_ok = []
    for x in ok_vm:
        _, _, plm = V18.task1_sample_masks(x, 1)
        pm_ok.append(all(not (p and v) for p, v in zip(plm, x["planner_gen"]["value_mask"])))
    chk["t1_plan_mask_excludes_value"] = all(pm_ok) if pm_ok else NM
    cn = row["task1_stats"].get("components_n") or {}
    chk["components_task1"] = all((cn.get(k) or 0) > 0 for k in ("stop", "act", "len", "fmt"))
    chk["components_task2"] = row["task2_stats"].get("n_groups", 0) > 0
    # P_end: batch vs single on the logged t >= 2 valid samples
    items = [{"prompt_ids": x["planner_gen"]["prompt_ids"], "prefix_ids": x["prefix_ids"], "target_true": x["target_true"],
              "target_false": x["target_false"]} for r in t1rows if r["t"] >= 2 for x in r["samples"]
             if x.get("status") == "valid"][:8]
    if items:
        b = lr.p_end_batch(items)
        s1 = [lr.end_prob(x) for x in items]
        chk["pend_batch_equals_single"] = max(abs(x - y) for x, y in zip(b, s1)) < 1e-9
    else:
        chk["pend_batch_equals_single"] = NM
    # ---- scales / kappa / tau (§4.1, §4.3)
    sc = json.load(open(tr.p_scales)) if os.path.exists(tr.p_scales) else None
    chk["scales_written"] = sc is not None and sc["kappa"]["task1"] > 0 and sc["kappa"]["task2"] > 0
    chk["tau_u1_equals_targets"] = (abs(row["tau"]["task1"]["tau"] - a.adv_target_task1) < 1e-6
                                    and abs(row["tau"]["task2"]["tau"] - a.adv_target_task2) < 1e-6)
    logs["tau_v17_range"] = {"task1": [0.0167, 0.0248], "task2": [0.0410, 0.1194]}
    # ---- the verifier's recomputation on the logged rows
    import verify_pipeline as VP
    import verify_v18
    rep = VP.Report()
    verify_v18.check_v18(a.out, rep, a.splits)
    for name, key in (("recompute_reward", "rl.v18_reward"), ("recompute_task1", "rl.v18_task1"),
                      ("recompute_advantage", "rl.v18_advantage"), ("recompute_tau", "rl.v18_tau"),
                      ("recompute_scales", "rl.v18_scales")):
        c = rep.checks.get(key) or {"n": 0, "fail": 0, "examples": []}
        chk[name] = c["n"] > 0 and c["fail"] == 0
        if c["fail"]:
            logs.setdefault("recompute_failures", {})[key] = c["examples"]
    # ---- the reranker's selections (§6.4)
    n_sel, bad = 0, []
    for r in T.read_jsonl(tr.p_roll):
        for s in r["episode"].get("trace") or []:
            app, ok, why = verify_v18.selection_ok(s)
            if app:
                n_sel += 1
                if not ok:
                    bad.append(why)
    logs["reranked_selections"] = {"n": n_sel, "bad": bad[:3]}
    chk["rerank_selections_recomputed"] = (n_sel > 0 and not bad) if a.selector == "borda_rerank" else NM
    # ---- ref = u0 before the update (a fresh ref forward on a u0 Task 2 sample vs the u0 checkpoint reloaded)
    ep_rows = T.read_jsonl(tr.p_roll)
    smp = RA.episode_samples(ep_rows[0]["episode"], policy_version=0)[0] if ep_rows else None
    if smp is not None:
        cur = lr.policy_sha()
        ref_lp = lr._logp(dict(smp), grad=False, reference=True)[0].float().cpu()
        lr.load_policy(tr.ckpt_dir(0))
        u0_lp = lr._logp(dict(smp), grad=False)[0].float().cpu()
        lr.load_policy(tr.ckpt_dir(1))
        assert lr.policy_sha() == cur
        logs["ref_vs_u0_maxdiff"] = float((ref_lp - u0_lp).abs().max())
        chk["ref_equals_policy_at_u0"] = logs["ref_vs_u0_maxdiff"] < 1e-4
    else:
        chk["ref_equals_policy_at_u0"] = NM
    chk["no_ref_dir_in_checkpoints"] = S17.ref_dirs(tr.ckpt_root) == []
    # ---- resume: the scales are read back (u1 exists), the fingerprint of the logged update-1 rows matches
    try:
        sc2 = tr.scales_v18(2, [], [], [], [])
        chk["resume_reads_scales"] = sc2 == sc
    except SystemExit as e:
        logs["resume_error"] = str(e)
        chk["resume_reads_scales"] = False
    # ---- GPU / budget (as smoke_v17)
    og = S17.other_gpu_bytes(gpu)
    logs["other_gpu_bytes_end"] = og
    chk["no_other_gpu_memory_end"] = all(v == 0 for v in og.values())
    on, mine = S17.gpus_with_my_pid(), S17.my_gpu_uuid(gpu)
    logs["nvidia_smi_gpu_uuids_of_this_pid"], logs["training_gpu_uuid"] = on, mine
    chk["no_context_on_other_gpu"] = (on == [mine]) if on else NM
    held_end = torch.cuda.memory_reserved(gpu)
    chk["budget_still_reserved_at_end"] = (held_end >= target) if budget_on else NM
    tr.keep_budget("smoke end")
    phase("end")
    chk["memory_under_budget"] = all(v < S17.BUDGET_GIB for v in mem.values())
    res["total_s"] = round(time.time() - t0, 1)
    ess_fail = sorted(k for k in ESSENTIAL if chk.get(k) is not True)
    other_fail = sorted(k for k, v in chk.items() if k not in ESSENTIAL and v is False)
    res["not_measurable"] = sorted(k for k, v in chk.items() if v == NM)
    res["essential"] = list(ESSENTIAL)
    res["passed"] = not ess_fail and not other_fail
    res["failed_checks"] = ess_fail + other_fail
    json.dump(res, open(os.path.join(a.out, "smoke_v18.json"), "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    print("SMOKE V18 %s" % ("PASSED" if res["passed"] else "FAILED %s" % res["failed_checks"]))
    return res


if __name__ == "__main__":
    main()
