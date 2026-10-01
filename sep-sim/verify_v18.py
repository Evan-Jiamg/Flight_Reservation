"""SPEC v18 branch of verify_pipeline.py (ops/SPEC_v18_multiobj_rerank.md §11): everything recomputed from the run's logs
with the trainer's own rule code (v18_rules, rl_reward, rl_algos, style_select) -- never a second copy of a rule.

  rl.v18_settings     the SPEC v18 values (or a named ablation), epochs x minibatches, no SFT artifacts
  rl.v18_init         u0 = the init adapter (sha locked, a byte copy, plain files), ref = u0, the run_meta init row
  rl.v18_labels       label cids in train_all (reward) / validation_all (validation), shas, the 8-gram check, opened files,
                      no benchmark instrument / data path in the reward / label / reranker code, coverage not a reward
  rl.v18_task1        Task 1 points t = 1..n (minus capped suffixes), G samples, states (§3.1.5), turn-1 value masks
  rl.v18_reward       r_stop / r_act / r_len / r_fmt recomputed per sample (labels + the logged act_distribution)
  rl.v18_scales       adv_scales.json = the measurement on the fingerprinted update-1 batch (scales, frozen, tau, kappa);
                      aborted measurements renamed and never used
  rl.v18_advantage    c, z (clipped), A_pre / A_plan / A_ep per group recomputed; skip kinds; masks (value tokens 0)
  rl.v18_tau          the token-weighted tau of every update recomputed; the §4.3 alarms
  rl.v18_task2        clean_v18 groups, exclusions, drift
  rl.v18_update       optimizer steps 8, kl / lr / aux, KL > 0 from u2 (real runs)
  rl.v18_validation   Task 1 metrics (validation labels) recomputed, Task 2 only at u0 / the checked candidates
  rl.v18_selection    candidates, J, guards, order, the Task 2 check, final.json, the stop rule
  rl.v18_reranker     every selection recomputed from the Borda scores, the top-k rule and the logged rerank scores;
                      the reranker json (train cids, per-fold PCA, gate) and its sha
  rl.v18_test         the test evaluation holds the final policy only and names the v17 comparison file's sha
"""
from __future__ import annotations

import json
import math
import os
import re

import rl_algos as RA
import rl_reward as RR
import style_select as SS
import v18_rules as V18
import verify_pipeline as VP

TOL = 1e-6
CODE_NO_BENCH = ("label_acts.py", "train_reranker.py", "rl_reward.py", "v18_rules.py", "style_select.py")
BENCH = "/tmp2/hchsu/trec2026-usersim-benchmark"
SPEC_V18_CHECK = {"lr": 1e-5, "kl": 0.01, "task1_G": 4, "task1_convs": 8, "task1_positions": "all_t1", "aux_weight": 0.5,
                  "updates": 5, "val_every": 1, "length_drift_margin": 1.0, "controller": "fixed", "stop_credit": 0,
                  "reward_task1": "multi", "w_stop": 1.0, "w_act": 1.0, "w_len": 0.5, "w_fmt": 1.0, "len_band": 0.6667,
                  "reward_task2": "v5", "w_turn": 1.0, "w_fmt2": 1.0, "w_cov": 0.0, "adv_norm": "fixed_u0_tokenw",
                  "adv_z_clip": 3.0, "adv_target_task1": 0.0197, "adv_target_task2": 0.0762, "adv_min_spread_groups": 3,
                  "adv_min_spread_groups_task2": 2, "adv_min_scale": 0.01, "fmt_scale": 0.5, "rerank_topk": 2,
                  "init_policy_sha": "f67d643796951c6bac4aeec7ae1e826fb3f34d6139edaaf167d52e57d81e1ac6"}


def close(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def dclose(a, b, tol=TOL):
    """Two dicts of numbers / None equal within tol."""
    if set(a or {}) != set(b or {}):
        return False
    return all(close(a[k], b[k], tol) for k in a)


def bench_paths_in(text):
    """Benchmark instrument / data paths named in a code file (comments included: a path in a comment is refused too)."""
    return re.findall(r"(?:instruments/[A-Za-z0-9_./-]+|data/req_shards[A-Za-z0-9_./-]*)", text)


def check_code_no_bench(rep, here):
    """§1.2 / §11: the reward / label / reranker code names no benchmark instrument or data file; the one allowed name is
    the act codebook in label_acts.py, read ONLY for the 8-gram overlap refusal (§1.1)."""
    for f in CODE_NO_BENCH:
        p = os.path.join(here, f)
        if not os.path.exists(p):
            rep.ok("rl.v18_labels", False, f, "code file missing")
            continue
        hits = bench_paths_in(open(p, encoding="utf-8").read())
        allowed = {"instruments/prompts/act_codebook_system.txt"} if f == "label_acts.py" else set()
        bad = sorted(set(hits) - allowed)
        rep.ok("rl.v18_labels", not bad, f, "names benchmark instrument / data paths %s" % bad)


def opened_bench(paths, allow=()):
    """Opened files under the benchmark's instruments/ or data/ (minus the allowed ones)."""
    out = []
    for p in paths or []:
        q = str(p).replace("\\", "/")
        if (q.startswith(BENCH + "/instruments/") or q.startswith(BENCH + "/data/")) and q not in allow:
            out.append(q)
    return out


def latest_rows(path, key):
    out = {}
    for r in VP._jl(path):
        out[key(r)] = r
    return out


def selection_ok(step):
    """§6.4 recomputed for one Speaker step with a reranker selection: the Borda choice (borda_order[0]) == borda_index,
    rerank_choice(info, scores, k) == the selected index, rerank_changed consistent. -> (applicable, ok, why)."""
    sel = step.get("selection") or {}
    if "rerank_scores" not in sel:
        return False, True, ""
    scores = {c: s for c, s in zip(sel["candidates"], sel["rerank_scores"]) if s is not None}
    b = SS.borda_order(sel)[0]
    ch, top = SS.rerank_choice(sel, scores, int(sel.get("rerank_k", 2)))
    ok = (b == sel.get("borda_index") and ch == step.get("selected_index") and top == sel.get("topk")
          and bool(sel.get("rerank_changed")) == (ch != b))
    return True, ok, "borda %r/%r, choice %r/%r, top %r/%r" % (b, sel.get("borda_index"), ch, step.get("selected_index"),
                                                             top, sel.get("topk"))


def t1_samples_from_records(r, used, kind_where):
    """Rebuild the learner samples of one used Task 1 group from its row and advantage record (tau recompute)."""
    by_rep = {x["replicate"]: x for x in r["samples"]}
    out = []
    for rec in used:
        x = by_rep[rec[0]]
        where, pm, plm = V18.task1_sample_masks(x, r["t"])
        g = dict(x["planner_gen"], prefix_mask=pm, plan_mask=plm)
        pseudo = {"conversation_id": r["conversation_id"], "replicate": x["replicate"],
                  "trace": [{"t": x["t"], "planner_gen": g}]}
        for s_ in RA.episode_samples(pseudo):
            s_.update(adv=0.0, adv_stop=0.0, adv_prefix=rec[5], adv_plan=rec[6], source="task1")
            out.append(s_)
    return out


def check_v18(rl_dir, rep, splits_path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    meta = VP._jl(os.path.join(rl_dir, "run_meta.jsonl"))
    if not rep.ok("rl.v18_settings", bool(meta), rl_dir, "run_meta.jsonl missing"):
        return
    cfg0 = meta[0].get("config") or {}
    args, acfg = cfg0.get("args") or {}, cfg0.get("algo_cfg") or {}
    abl, dry = bool(args.get("ablation")), bool(args.get("dry_run"))
    real_vllm = args.get("planner_backend") == "vllm" and not dry
    v18c = cfg0.get("v18") or {}
    # ---------------------------------------------------------------- settings (§10)
    rep.ok("rl.v18_settings", all((m.get("config") or {}).get("spec_version") == "v18" for m in meta), "run_meta",
           "a run_meta row of another spec version")
    for k, want in SPEC_V18_CHECK.items():
        rep.ok("rl.v18_settings", args.get(k) == want or abl, "run_meta", "%s = %r (spec v18: %r)" % (k, args.get(k), want))
    rep.ok("rl.v18_settings", sorted(args.get("val_seeds") or []) == list(range(8)) or abl, "run_meta", "val_seeds")
    rep.ok("rl.v18_settings", (acfg.get("epochs"), acfg.get("minibatches")) == (2, 4) or abl, "run_meta", "epochs x mb")
    rep.ok("rl.adv_norm", acfg.get("grpo_std_norm") is False, "run_meta", "grpo_std_norm %r" % acfg.get("grpo_std_norm"))
    rep.ok("rl.v18_settings", args.get("stop_credit") == 0 and args.get("w_cov") == 0.0, "run_meta",
           "v18 runs with stop credit 0 and w_cov 0 (coverage is not a reward)")
    left = [p for p in ("sft_examples.jsonl", "sft_examples_meta.json", "sft.jsonl", "base_pend_train.jsonl")
            if os.path.exists(os.path.join(rl_dir, p))] + \
        [d for d in sorted(os.listdir(os.path.join(rl_dir, "ckpt"))) if d.startswith("sft_e")] \
        if os.path.isdir(os.path.join(rl_dir, "ckpt")) else []
    rep.ok("rl.v18_settings", not left, rl_dir, "SFT artifacts in a v18 run: %s" % left)
    sp_path = splits_path or args.get("splits")
    try:
        f = {int(x["fold"]): x for x in json.load(open(sp_path, encoding="utf-8"))["folds"]}[int(args["fold"])]
        rep.ok("rl.v18_settings", VP.sha256_file(sp_path) == meta[0].get("splits_sha256"), "splits", "split file sha")
    except Exception as e:                       # never skipped silently
        rep.ok("rl.v18_settings", False, "splits", "cannot read the split file %r: %r" % (sp_path, e))
        return
    train_all, forb = set(f.get("train_all", f["train"])), set(f.get("forbidden_for_training", []))
    val, val_all = set(f["validation"]), set(f.get("validation_all", f["validation"]))
    # ---------------------------------------------------------------- Ditto architecture (as v17 S12)
    env = cfg0.get("env")
    rep._c("rl.v18_ditto")
    if env is None:
        rep.ok("rl.v18_ditto", dry, "run_meta", "a real run without config.env")
    else:
        ae, v2 = env.get("arm_env") or {}, env.get("v2fix") or {}
        rep.ok("rl.v18_ditto", env.get("arm") == "pend" and "Ditto-8B" in str(env.get("speaker_path"))
               and env.get("speaker_endconv_is_none") is True and "e1r_cf19400" in str(env.get("tree"))
               and any(c.endswith("DittoSpeaker") for c in env.get("speaker_class") or []), "env", "Ditto / pend / tree")
        for k in ("SEPSIM_ENDMASK_RETRY", "SEPSIM_ENDGATE", "SEPSIM_KEEPEND", "SEPSIM_ENDSCORE"):
            rep.ok("rl.v18_ditto", ae.get(k) == "0", "env", "%s = %r" % (k, ae.get(k)))
        rep.ok("rl.v18_ditto", v2.get("SEPSIM_END_PROBE") == "0", "env", "SEPSIM_END_PROBE")
        rr = env.get("reranker")
        if args.get("selector") == "borda_rerank":
            rep.ok("rl.v18_reranker", isinstance(rr, dict) and rr.get("sha256") == v18c.get("reranker", {}).get("sha256")
                   and env.get("selector") == "borda_rerank" and env.get("rerank_topk") == args.get("rerank_topk"), "env",
                   "the env's reranker %r is not the run's" % (rr,))
        else:
            rep.ok("rl.v18_reranker", rr is None and env.get("selector") == "borda", "env", "a reranker without the gate")
    # ---------------------------------------------------------------- init / ref (§2)
    st0p = os.path.join(rl_dir, "ckpt", "u00000", "state.json")
    st0 = json.load(open(st0p, encoding="utf-8")) if os.path.exists(st0p) else None
    if rep.ok("rl.v18_init", st0 is not None, st0p, "no ckpt u00000"):
        rep.ok("rl.v18_init", st0["policy_sha"] == args.get("init_policy_sha"), st0p,
               "u0 policy sha %s != --init-policy-sha" % str(st0["policy_sha"])[:12])
        info = st0.get("init") or {}
        ad0 = os.path.join(rl_dir, "ckpt", "u00000", "adapter")
        sha0 = VP_sha_dir(ad0)
        rep.ok("rl.v18_init", info.get("copied") is True and info.get("policy_sha") == st0["policy_sha"]
               and info.get("adapter_sha256") == sha0 and all(m.get("init_adapter_sha256") == sha0 for m in meta),
               ad0, "u0 adapter is not the recorded copy of the init adapter")
        for root, _, files in os.walk(ad0):
            for fn in files:
                fp_ = os.path.join(root, fn)
                rep.ok("rl.v18_init", not os.path.islink(fp_) and os.stat(fp_).st_nlink == 1, fp_,
                       "u0 adapter file is a link")
        src = os.path.join(args.get("init_adapter") or "", "adapter")
        if os.path.isdir(src):
            rep.ok("rl.v18_init", VP_sha_dir(src) == sha0, src, "the init adapter changed / differs from u0's copy")
        rep.ok("rl.v18_init", any(m.get("kind") == "init" for m in meta), "run_meta", "no init row")
        for m in meta:
            if m.get("ref_policy_sha") is not None:
                rep.ok("rl.v18_init", m["ref_policy_sha"] == st0["policy_sha"], "run_meta %s" % m.get("kind"), "ref != u0")
    import glob
    for d in sorted(glob.glob(os.path.join(rl_dir, "ckpt", "*", "adapter"))):
        rep.ok("rl.v18_init", not os.path.exists(os.path.join(d, "ref")), d, "a checkpoint holds the ref adapter")
    # ---------------------------------------------------------------- labels (§1)
    labels = vlabels = lmeta = None
    try:
        labels, lmeta, lsha = V18.load_labels(args["act_labels"])
        vlabels, _, vsha = V18.load_labels(args["act_labels_val"])
        rep.ok("rl.v18_labels", lsha == v18c.get("labels_train", {}).get("sha256") == meta[0].get("labels_train_sha256")
               and vsha == v18c.get("labels_val", {}).get("sha256"), "labels", "label files changed since the run")
        rep.ok("rl.v18_labels", not V18.check_label_cids(labels, train_all, forb), "labels",
               "reward labels outside train_all / in forbidden")
        rep.ok("rl.v18_labels", not V18.check_label_cids(vlabels, val_all, set()), "labels",
               "validation labels outside validation_all")
        rep.ok("rl.v18_labels", not ({c for c, _ in labels} & val_all), "labels", "reward labels on a validation id")
    except Exception as e:
        rep.ok("rl.v18_labels", False, "labels", "cannot read the labels: %r" % (e,))
    vmeta = None
    try:
        vmeta = V18.load_labels(args["act_labels_val"])[1]
    except Exception:
        pass
    # audit A: the validation labels' meta gets the same 8-gram / opened-files checks as the train labels'
    for lmeta_, which in ((lmeta, "train"), (vmeta, "validation")):
        if lmeta_ is None and dry:
            continue
        lm = lmeta_ or {}
        rep.ok("rl.v18_labels", lm.get("ngram8_overlap", 0) == 0 or dry, "%s labels meta" % which,
               "the label prompt shares %r 8-grams with the benchmark codebook" % lm.get("ngram8_overlap"))
        rep.ok("rl.v18_labels", not opened_bench(lm.get("opened_files"), allow=(BENCH + "/instruments/prompts/act_codebook_system.txt",)),
               "%s labels meta" % which, "the labeller opened benchmark files %s" % opened_bench(lm.get("opened_files")))
        if not dry:
            rep.ok("rl.v18_labels", lm.get("ngram8_overlap") == 0 and lm.get("ngram8_checked") is True,
                   "%s labels meta" % which, "no recorded 8-gram check")
    check_code_no_bench(rep, here)
    # ---------------------------------------------------------------- reranker json (§6)
    rpath = args.get("reranker")
    rj = None
    try:
        rj = json.load(open(rpath, encoding="utf-8"))
        rep.ok("rl.v18_reranker", VP.sha256_file(rpath) == v18c.get("reranker", {}).get("sha256"), rpath, "reranker sha")
        g = rj.get("gate") or {}
        rep.ok("rl.v18_reranker", (args.get("selector") == "borda_rerank") == bool(g.get("passed")) or abl, rpath,
               "selector %r but gate passed %r" % (args.get("selector"), g.get("passed")))
        if not dry:
            rep.ok("rl.v18_reranker", set(rj.get("train_cids") or []) <= train_all and rj.get("train_cids"), rpath,
                   "reranker trained outside train_all")
            for fo in (rj.get("loco") or {}).get("folds") or []:
                rep.ok("rl.v18_reranker", fo.get("held_out") not in (fo.get("pca_fit_cids") or [])
                       and set(fo.get("pca_fit_cids") or []) <= train_all, rpath,
                       "fold %s: PCA fitted on the held-out conversation" % fo.get("held_out"))
            want = bool(g.get("loco_acc") is not None and g.get("val_acc") is not None
                        and g["loco_acc"] >= 0.60 and g["val_acc"] >= 0.55)
            rep.ok("rl.v18_reranker", bool(g.get("passed")) == want, rpath, "gate %r inconsistent with its accuracies" % g)
            rep.ok("rl.v18_reranker", not opened_bench(rj.get("opened_files")), rpath,
                   "the reranker fit opened benchmark files %s" % opened_bench(rj.get("opened_files")))
    except Exception as e:
        rep.ok("rl.v18_reranker", False, str(rpath), "cannot read the reranker json: %r" % (e,))
    # ---------------------------------------------------------------- per update
    upd = {u["update"]: u for u in VP._jl(os.path.join(rl_dir, "updates.jsonl"))}
    t1p, t2p = os.path.join(rl_dir, "rollouts_task1.jsonl"), os.path.join(rl_dir, "rollouts.jsonl")
    t1_latest = latest_rows(t1p, lambda r: (r["update"], r["conversation_id"], r["t"]))
    t2_latest = latest_rows(t2p, lambda r: (r["update"], r["slot"], r["replicate"]))
    sp_ = os.path.join(rl_dir, "adv_scales.json")
    sc = json.load(open(sp_, encoding="utf-8")) if os.path.exists(sp_) else None
    band = float(args.get("len_band", 0.6667))
    w_eff = {"stop": args.get("w_stop"), "act": args.get("w_act"), "len": args.get("w_len"), "fmt": args.get("w_fmt"),
             "turn": args.get("w_turn"), "fmt2": args.get("w_fmt2")}
    kap = (lmeta or {}).get("fleiss_kappa_coarse")
    if kap is not None and kap < 0.4:
        w_eff["act"] = min(float(args.get("w_act")), 0.5)            # §1.1, recomputed (audit B NIT 3)
    rep.ok("rl.v18_settings", (v18c.get("weights_effective") or {}) == w_eff, "run_meta",
           "weights_effective %r, recomputed from the args and the label meta's kappa %r" % (v18c.get("weights_effective"), w_eff))
    n_of = {}
    for (c, t) in (labels or {}):
        n_of[c] = max(n_of.get(c, 0), t)
    G1, n_convs = int(args.get("task1_G", 4)), int(args.get("task1_convs", 8))
    epochs, mbs = int(acfg.get("epochs", 2)), int(acfg.get("minibatches", 4))
    tm = int((cfg0.get("selection_cfg") or {}).get("t_max", 10))
    drift = {}
    if upd:
        rep.ok("rl.v18_scales", sc is not None, sp_, "adv_scales.json missing")
    for u, row in sorted(upd.items()):
        w = "update %s" % u
        try:
            w = "update %s" % u
            ts, t2s, st = row.get("task1_stats") or {}, row.get("task2_stats") or {}, row.get("learner_stats") or {}
            cfg, ag = row.get("cfg_used") or {}, row.get("train_aggregate") or {}
            # settings of the update (§5)
            rep.ok("rl.v18_update", close(cfg.get("kl_coef"), args.get("kl")) and close(cfg.get("lr"), args.get("lr"))
                   and float(ag.get("aux_weight", -1)) == float(args.get("aux_weight", 0.5)), w, "kl / lr / aux")
            if args.get("controller") == "fixed":
                rep.ok("rl.v18_update", cfg == cfg0.get("cfg0"), w, "the fixed controller's cfg moved")
            n_s = int(row.get("n_samples") or 0)
            want = epochs * min(mbs, n_s) if n_s else (1 if st.get("aux_n") else 0)
            rep.ok("rl.v18_update", st.get("optimizer_steps") == want, w, "%r optimizer steps, expected %d" % (st.get("optimizer_steps"), want))
            if u >= 2 and not dry and n_s:
                rep.ok("rl.v18_update", float(st.get("kl") or 0.0) > 0.0, w, "KL to the ref is %r" % st.get("kl"))
            if sc is not None:
                rep.ok("rl.v18_scales", row.get("adv_scales_sha256") == VP.sha256_file(sp_) and dclose(row.get("kappa"), sc["kappa"])
                       and dclose(row.get("scales"), sc["scales"]) and row.get("frozen") == sc["frozen"], w,
                       "the update did not use adv_scales.json")
            # ---- Task 2 groups (clean_v18), drift
            slots = {}
            for (uu, slot, g), r in t2_latest.items():
                if uu == u:
                    slots.setdefault(slot, []).append(r)
            excl_want, groups2 = [], []
            for slot in sorted(slots):
                rs = sorted(slots[slot], key=lambda r: r["replicate"])
                keep = []
                for r in rs:
                    ep = r["episode"]
                    rep.ok("rl.v18_task2", ep.get("clean_v18") is RR.clean_v18(ep), w, "clean_v18 flag recomputed differs")
                    (keep if RR.clean_v18(ep) else excl_want).append(r if RR.clean_v18(ep) else [slot, r["replicate"]])
                if len(keep) >= 2:
                    groups2.append(keep)
            rep.ok("rl.v18_task2", sorted([e[0], e[1]] for e in (t2s.get("excluded") or [])) == sorted(excl_want), w,
                   "excluded episodes %r, recomputed %r" % ([e[:2] for e in t2s.get("excluded") or []], excl_want))
            eps_u = [r["episode"] for g in groups2 for r in g]
            if rep.ok("rl.v18_task2", bool(eps_u), w, "no clean_v18 group"):
                dv = V18.drift_of(eps_u, tm)
                drift[u] = dv
                rep.ok("rl.v18_task2", close(ag.get("drift_stat"), dv, 1e-9), w, "drift %r != %r" % (ag.get("drift_stat"), dv))
            # ---- Task 1 rows, states, components
            convs = set(ts.get("convs") or [])
            rep.ok("rl.v18_task1", convs <= train_all and not convs & forb and len(convs) == min(n_convs, len(train_all)), w,
                   "Task 1 conversations")
            rows1 = {}
            for cid, t in ts.get("groups") or []:
                r = t1_latest.get((u, cid, t))
                if rep.ok("rl.v18_task1", r is not None and cid in convs, w, "group %s t%s has no row" % (str(cid)[:10], t)):
                    rows1[(cid, t)] = r
            skipped = {(c, t) for c, t in ts.get("skipped_capped") or []}
            for cid in convs:
                n = n_of.get(cid)
                got = {t for (c, t) in rows1 if c == cid}
                sk = {t for (c, t) in skipped if c == cid}
                rep.ok("rl.v18_task1", n is not None and (got | sk) == set(range(1, n + 1)) and not (got & sk)
                       and VP._skips_form_suffix(sorted(sk), n), w,
                       "%s: points %s + skipped %s are not t = 1..%s" % (str(cid)[:10], sorted(got), sorted(sk), n))
            for (cid, t), r in sorted(rows1.items()):
                wg = "%s %s t%s" % (w, str(cid)[:10], t)
                smp = r.get("samples") or []
                lab = V18.label_of(labels or {}, cid, t)
                hw = (labels or {}).get((cid, t), {}).get("n_words")
                rep.ok("rl.v18_task1", len(smp) == G1 and r["n_real"] == n_of.get(cid) and r["real_final"] == (t == r["n_real"])
                       and r.get("label") == lab and (hw is None or r.get("human_words") == hw), wg,
                       "samples / n / label snapshot / human words disagree with the labels")
                for x in smp:
                    stv, why = V18.t1_status(x, t)
                    ok = x.get("status") == stv and x.get("drop_reason") == why
                    g_ = x.get("planner_gen") or {}
                    if stv == "valid" and t >= 2:
                        ok = ok and x.get("p_end") is not None and 0.0 <= x["p_end"] <= 1.0 and g_.get("stop_mask") \
                            and 1 in g_["stop_mask"] and g_["stop_mask"].index(1) > 0
                    elif t == 1:
                        ok = ok and x.get("p_end") is None and g_.get("stop_mask") is None
                        if stv == "valid":
                            vm = g_.get("value_mask")
                            ok = ok and vm is not None and len(vm) == len(g_.get("gen_ids") or []) and 1 in vm
                        if x.get("valid_t1"):
                            ok = ok and bool(x.get("value_mask_ok")) == (g_.get("value_mask") is not None)
                    else:
                        ok = ok and x.get("p_end") is None
                    rep.ok("rl.v18_task1", ok, wg, "sample r%s: state %r / %r recomputed %r (§3.1.5)" % (
                        x.get("replicate"), x.get("status"), x.get("drop_reason"), stv))
                    if stv != "dropped":
                        m = V18.t1_member(x, t, r["n_real"], lab, r.get("human_words"), band)
                        lc = x.get("components") or {}
                        rep.ok("rl.v18_reward", all(close(lc.get(k), m[k], 1e-12) for k in V18.T1_KEYS), wg,
                               "r%s components %r, recomputed %r" % (x.get("replicate"), {k: lc.get(k) for k in V18.T1_KEYS},
                                                                     {k: m[k] for k in V18.T1_KEYS}))
            rep.ok("rl.v18_task1", sorted(ts.get("aux_points") or []) == sorted(
                [c, t] for (c, t), r in rows1.items() if t >= 2 and any(x.get("status") == "valid" for x in r["samples"]))
                   and st.get("aux_n", 0) == len(ts.get("aux_points") or []), w,
                   "aux examples are not one per t >= 2 point with a valid sample (none at turn 1)")
            # ---- advantages recomputed (Task 1 and Task 2) and tau
            if sc is None:
                continue
            frozen, scales, zc = set(sc["frozen"]), sc["scales"], float(args.get("adv_z_clip", 3.0))
            k1, k2 = sc["kappa"]["task1"], sc["kappa"]["task2"]
            logged = {(a_[0], a_[1]): a_ for a_ in ts.get("advantages") or []}
            rep.ok("rl.v18_advantage", set(logged) == set(rows1), w, "advantage records != Task 1 groups")
            smp_all = []
            for (cid, t), r in sorted(rows1.items()):
                wg = "%s %s t%s" % (w, str(cid)[:10], t)
                kept, members = V18.t1_members(r, labels or {}, band)
                lg = logged.get((cid, t))
                if len(kept) < 2:
                    rep.ok("rl.v18_advantage", lg is not None and lg[2] == "lt2", wg, "lt2 group logged as %r" % (lg and lg[2]))
                    continue
                rows_z, _ = V18.advantages_t1(members, scales, frozen, zc, w_eff, k1)
                if rows_z is None:
                    rep.ok("rl.v18_advantage", lg is not None and lg[2] == "zero_spread", wg, "zero-spread group logged %r" % (lg and lg[2]))
                    continue
                if not rep.ok("rl.v18_advantage", lg is not None and lg[2] == "used" and len(lg[3]) == len(kept), wg,
                              "used group not logged as used"):
                    continue
                for x, z, rec in zip(kept, rows_z, lg[3]):
                    where, _, _ = V18.task1_sample_masks(x, t)
                    a_pre = z["A_pre"] if where == "prefix+plan" else 0.0
                    rep.ok("rl.v18_advantage", rec[0] == x["replicate"] and rec[1] == x["status"] and close(rec[5], a_pre)
                           and close(rec[6], z["A_plan"]) and rec[7] == where and dclose(rec[4], z["z"]), wg,
                           "r%s: logged A_pre %r A_plan %r (%s), recomputed %r %r (%s)" % (
                               x["replicate"], rec[5], rec[6], rec[7], a_pre, z["A_plan"], where))
                smp_all += t1_samples_from_records(r, lg[3], None)
            logged2 = {a_[0]: a_ for a_ in t2s.get("advantages") or []}
            tm_ = tm
            for g in groups2:
                slot = g[0]["slot"]
                members = [V18.t2_member(r["episode"], tm_) for r in g]
                rows_z, _ = V18.advantages_t2(members, scales, frozen, zc, w_eff, k2)
                lg = logged2.get(slot)
                if rows_z is None:
                    rep.ok("rl.v18_advantage", lg is not None and lg[1] == "zero_spread", w, "Task 2 slot %s zero spread" % slot)
                    continue
                if not rep.ok("rl.v18_advantage", lg is not None and lg[1] == "used" and len(lg[2]) == len(g), w,
                              "Task 2 slot %s not logged as used" % slot):
                    continue
                for r, z, rec in zip(g, rows_z, lg[2]):
                    rep.ok("rl.v18_advantage", rec[0] == r["replicate"] and close(rec[4], z["A_ep"]), w,
                           "Task 2 slot %s r%s: A_ep %r, recomputed %r" % (slot, r["replicate"], rec[4], z["A_ep"]))
                    for s_ in RA.episode_samples(r["episode"]):
                        s_.update(adv=z["A_ep"], adv_stop=0.0, adv_prefix=0.0, adv_plan=0.0, source="task2")
                        smp_all.append(s_)
            rep.ok("rl.v18_advantage", set(logged2) == {g[0]["slot"] for g in groups2}, w, "Task 2 advantage records != groups")
            # value tokens never get a Task 1 advantage; note tokens nothing; stop credit 0
            for s_ in smp_all:
                a_tok = RA.token_advantages(s_, len(s_["gen_ids"]))
                if s_["source"] == "task1":
                    for key in ("stop_mask", "value_mask"):
                        m_ = s_.get(key)
                        if m_ is not None:
                            rep.ok("rl.v18_advantage", all(v == 0.0 for v, mm in zip(a_tok, m_) if mm), w,
                                   "a Task 1 value token got an advantage")
                nm = s_.get("note_mask")
                if nm is not None:
                    rep.ok("rl.v18_advantage", all(v == 0.0 for v, mm in zip(a_tok, nm) if mm), w, "a note token got an advantage")
            tau = RA.token_weighted_tau(smp_all)
            lt = row.get("tau") or {}
            rep.ok("rl.v18_tau", all(close((lt.get(s) or {}).get("tau"), tau[s]["tau"]) and (lt.get(s) or {}).get("n_tokens")
                                     == tau[s]["n_tokens"] for s in ("task1", "task2")), w,
                   "tau %r, recomputed %r" % ({s: (lt.get(s) or {}).get("tau") for s in lt}, {s: tau[s]["tau"] for s in tau}))
            al = V18.tau_alarms(tau, st.get("rl_grad_norm"), st.get("aux_grad_norm"))
            rep.ok("rl.v18_tau", al == row.get("alarms"), w, "alarms %r, recomputed %r" % (row.get("alarms"), al))
            if al:
                rep.warn("rl.v18_tau_alarm", "%s: %s" % (w, "; ".join(al)))
            if u == 1:
                # the measurement batch = update 1's rows (fingerprint), the scales / frozen / kappa recomputed from it
                fp = RA.batch_fingerprint([r for (uu, _, _), r in t2_latest.items() if uu == 1], list(rows1.values()))
                rep.ok("rl.v18_scales", sc.get("batch_sha256") == fp, sp_, "batch fingerprint %s, update-1 rows %s"
                       % (str(sc.get("batch_sha256"))[:12], fp[:12]))
                t1m = [V18.t1_members(r, labels or {}, band)[1] for r in rows1.values()]
                t1m = [m for m in t1m if len(m) >= 2]
                mins1 = {k: int(args.get("adv_min_spread_groups", 3)) for k in ("stop", "act", "len")}
                m1 = RA.measure_scales(t1m, V18.T1_KEYS, mins1, float(args.get("adv_min_scale", 0.01)),
                                       fixed={"fmt": float(args.get("fmt_scale", 0.5))})
                t2m = [[V18.t2_member(r["episode"], tm) for r in g] for g in groups2]
                m2 = RA.measure_scales(t2m, V18.T2_KEYS, {"turn": int(args.get("adv_min_spread_groups_task2", 2))},
                                       float(args.get("adv_min_scale", 0.01)), fixed={"fmt2": float(args.get("fmt_scale", 0.5))})
                meas = dict(m1, **m2)
                rep.ok("rl.v18_scales", all(close(sc["scales"][k], v["scale"]) for k, v in meas.items())
                       and sorted(k for k, v in meas.items() if v["frozen"]) == sc["frozen"]
                       and all(sc["spread_groups"][k] == v["n_spread_groups"] for k, v in meas.items()), sp_,
                       "scales / frozen / spread groups differ from the recomputed measurement")
                rep.ok("rl.v18_scales", not (set(sc["frozen"]) & set(V18.MUST_NOT_FREEZE)), sp_,
                       "training ran with a frozen r_stop / r_act / r_turn")
                # tau(kappa = 1) and kappa
                s1 = []
                for (cid, t), r in rows1.items():
                    kept, members = V18.t1_members(r, labels or {}, band)
                    if len(kept) < 2:
                        continue
                    rz, _ = V18.advantages_t1(members, scales, frozen, zc, w_eff, 1.0)
                    if rz is None:
                        continue
                    recs = [[x["replicate"], x["status"], None, None, z["z"],
                             z["A_pre"] if V18.task1_sample_masks(x, t)[0] == "prefix+plan" else 0.0, z["A_plan"],
                             V18.task1_sample_masks(x, t)[0]] for x, z in zip(kept, rz)]
                    s1 += t1_samples_from_records(r, recs, None)
                for g in groups2:
                    rz, _ = V18.advantages_t2([V18.t2_member(r["episode"], tm) for r in g], scales, frozen, zc, w_eff, 1.0)
                    if rz is None:
                        continue
                    for r, z in zip(g, rz):
                        for s_ in RA.episode_samples(r["episode"]):
                            s_.update(adv=z["A_ep"], adv_stop=0.0, adv_prefix=0.0, adv_plan=0.0, source="task2")
                            s1.append(s_)
                tk = RA.token_weighted_tau(s1)
                kk = RA.calibrate_kappa(tk["task1"]["tau"], tk["task2"]["tau"], args.get("adv_target_task1"),
                                        args.get("adv_target_task2"))
                rep.ok("rl.v18_scales", close(kk[0], k1) and close(kk[1], k2)
                       and sc.get("kappa2_low_confidence") == (meas["turn"]["n_spread_groups"] == 2), sp_,
                       "kappa %r, recomputed %r" % (sc["kappa"], kk))
        except Exception as e:                   # a log the rules cannot even be applied to is a failure, never a crash
            rep.ok("rl.v18_advantage", False, w, "the recomputation raised %r" % (e,))
    # aborted measurements: renamed, never used by a gradient step
    for p in sorted(glob.glob(os.path.join(rl_dir, "adv_scales.aborted_*.json"))):
        a_ = json.load(open(p, encoding="utf-8"))
        rep.ok("rl.v18_scales", os.path.exists(os.path.join(rl_dir, "ABORTED_u00001.json"))
               and sc is not None and a_.get("batch_sha256") != sc.get("batch_sha256")
               and all(r.get("adv_scales_sha256") != VP.sha256_file(p) for r in upd.values()), p,
               "an aborted measurement without an abort, or used by an update")
        # audit B NIT 2: the file holds the identities of the rows it was measured on; they must hash to its fingerprint
        # and every one must be a logged update-1 row (of any attempt)
        logged = {RA.json_key(x) for x in RA.batch_ids([r for r in VP._jl(t2p) if r["update"] == 1],
                                                     [r for r in VP._jl(t1p) if r["update"] == 1])}
        ids = a_.get("batch_ids")
        rep.ok("rl.v18_scales", isinstance(ids, list) and RA.fingerprint_of_ids(ids) == a_.get("batch_sha256")
               and all(RA.json_key(x) in logged for x in ids), p,
               "the aborted measurement's rows (batch_ids) do not hash to its fingerprint or are not logged update-1 rows")
    # ---------------------------------------------------------------- reranker selections (§6.4)
    n_sel = 0
    for path in (t2p, os.path.join(rl_dir, "validation.jsonl")):
        for r in VP._jl(path):
            ep = r.get("episode")
            if not isinstance(ep, dict):
                continue
            for s in ep.get("trace") or []:
                app, ok, why = selection_ok(s)
                if app:
                    n_sel += 1
                    rep.ok("rl.v18_reranker", ok, "%s t%s" % (str(r.get("conversation_id"))[:10], s.get("t")), why)
    rep.note("rl.v18_reranker", "%d reranked selections recomputed" % n_sel)
    if args.get("selector") == "borda_rerank" and not dry:
        rep.ok("rl.v18_reranker", n_sel > 0, rl_dir, "selector borda_rerank but no reranked selection logged")
    # ---------------------------------------------------------------- validation + selection (§7)
    vrows = VP._jl(os.path.join(rl_dir, "validation.jsonl"))
    summ = [v for v in vrows if v.get("kind") == "summary"]
    hwv = {}
    for (c, t), lr in (vlabels or {}).items():
        hwv.setdefault(c, {})[t] = lr.get("n_words")
    hw_by = {c: [d_[t] for t in sorted(d_)] for c, d_ in hwv.items()}
    metrics = {}
    for v in summ:
        u, w = v["update"], "validation u%s%s" % (v["update"], " task2" if v.get("task2") else "")
        rep.ok("rl.v18_validation", sorted(v.get("task1_ids") or []) == sorted(val_all), w, "Task 1 ids != validation_all")
        cp_ = os.path.join(rl_dir, "ckpt", "u%05d" % u, "state.json")
        rep.ok("rl.v18_validation", os.path.exists(cp_) and json.load(open(cp_, encoding="utf-8"))["policy_sha"] == v.get("policy_sha"),
               w, "summary policy sha is not ckpt u%s's" % u)
        rows_ = {}
        for r in vrows:
            if r.get("kind") == "task1" and r.get("update") == u and r.get("policy_sha") == v.get("policy_sha"):
                rows_[r["conversation_id"]] = r
        rep.ok("rl.v18_validation", sorted(rows_) == sorted(val_all), w, "Task 1 rows != validation_all")
        pts = [p for r in rows_.values() for p in (r.get("end_probs") or [])]
        rep.ok("rl.v18_validation", all(VP._unscored_ok(p) for p in pts) and len(pts) == sum(
            len(r["task1"]["turns"]) - 1 for r in rows_.values()), w, "probe points")
        if vlabels is not None and rows_ and set(hw_by) >= set(rows_):
            import task1_stop as T1
            m = V18.validation_metrics(list(rows_.values()), vlabels, hw_by, band)
            m["nll"] = T1.task1_prob_metrics(pts)["nll"] if pts else None
            lm = v.get("v18") or {}
            rep.ok("rl.v18_validation", all(close(lm.get(k), m[k]) for k in m), w,
                   "v18 metrics %r, recomputed %r" % ({k: lm.get(k) for k in m}, m))
            metrics.setdefault(u, m)
        if real_vllm:
            want_ad = VP.adapter_name(rl_dir, u)
            rep.ok("rl.v18_validation", all(p.get("gen_adapter") == want_ad for p in pts), w, "probes not by ckpt u%s" % u)
        if v.get("task2"):
            ep = {}
            for r in vrows:
                if r.get("kind") == "episode" and r.get("update") == u and r.get("policy_sha") == v.get("policy_sha"):
                    ep[(r["conversation_id"], r["seed"])] = r
            seeds = sorted(args.get("val_seeds") or list(range(8)))
            rep.ok("rl.v18_validation", sorted(ep) == sorted((c, s) for c in val for s in seeds), w, "Task 2 episodes")
            eps = [r["episode"] for r in ep.values() if r["episode"].get("clean_v18")]
            rep.ok("rl.v18_validation", v.get("n_episodes") == len(eps) and v.get("n_unclean_episodes") == len(ep) - len(eps),
                   w, "episode counts (clean_v18)")
            rep.ok("rl.v18_validation", close(v.get("task2_drift"), V18.drift_of(eps, tm)), w, "task2_drift")
            covs = [e.get("coverage_diag") for e in eps]
            ts_ = v.get("turn_stats") or {}
            have = [c for c in covs if c is not None]
            rep.ok("rl.v18_validation", ts_.get("coverage_missing") == len(covs) - len(have)
                   and close(ts_.get("coverage_mean"), (sum(have) / len(have)) if have else None), w,
                   "coverage diagnostic (mean over non-null, missing count)")
    # selection, Task 2 check, final.json, the stop rule
    fp = os.path.join(rl_dir, "final.json")
    fin = json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else None
    if upd:
        j = {0: 0.0}
        for u in sorted(upd):
            j[u] = V18.j_of(metrics[u], metrics[0], {"stop": args.get("w_stop"), "act": args.get("w_act"),
                                                      "len": args.get("w_len")}) if (u in metrics and 0 in metrics) else None
        su, reason = V18.stop_rule(drift, j, float(args.get("length_drift_margin", 1.0)), int(args.get("updates", 5)), max(upd))
        if fin is not None:
            rep.ok("rl.v18_selection", fin.get("last_update") == su == max(upd) and fin.get("stop_reason") == reason, fp,
                   "final.json last u%r (%r), recomputed u%r (%r)" % (fin.get("last_update"), fin.get("stop_reason"), su, reason))
            sel = V18.selection(metrics, drift, float(args.get("length_drift_margin", 1.0)), max(upd),
                                {"stop": args.get("w_stop"), "act": args.get("w_act"), "len": args.get("w_len")})
            fs = fin.get("selection") or {}
            rep.ok("rl.v18_selection", fs.get("candidates") == sel["candidates"] and fs.get("order") == sel["order"]
                   and all(close(fs["J"].get(k), v) for k, v in sel["J"].items()) and fs.get("guards") == sel["guards"], fp,
                   "selection %r, recomputed %r" % ({k: fs.get(k) for k in ("candidates", "order")},
                                                    {k: sel[k] for k in ("candidates", "order")}))
            t2s_ = {v["update"]: v for v in summ if v.get("task2")}
            final, checks = 0, []
            for u in sel["order"][:2]:
                s2 = t2s_.get(u)
                if s2 is None:
                    break
                ok = s2.get("task2_drift") is not None and s2["task2_drift"] >= -float(args.get("length_drift_margin", 1.0))
                checks.append({"update": u, "task2_drift": s2.get("task2_drift"), "passed": ok})
                if ok:
                    final = u
                    break
            rep.ok("rl.v18_selection", fin.get("final_update") == final and len(fs.get("task2_check") or []) == len(checks)
                   and all(a_["update"] == b_["update"] and a_["passed"] == b_["passed"]
                           for a_, b_ in zip(fs.get("task2_check") or [], checks)), fp,
                   "final u%r / checks %r, recomputed u%r / %r" % (fin.get("final_update"), fs.get("task2_check"), final, checks))
            st_f = os.path.join(rl_dir, "ckpt", "u%05d" % int(fin.get("final_update", 0)), "state.json")
            rep.ok("rl.v18_selection", os.path.exists(st_f) and json.load(open(st_f, encoding="utf-8"))["policy_sha"] == fin.get("policy_sha")
                   and fin.get("validated") is True and fin.get("reranker_sha256") == v18c.get("reranker", {}).get("sha256")
                   and fin.get("adv_scales_sha256") == (VP.sha256_file(sp_) if os.path.exists(sp_) else None), fp,
                   "final.json policy / validated / reranker / scales")
            stops = [m for m in meta if m.get("kind") == "stop"]
            rep.ok("rl.v18_selection", len(stops) == 1 and stops[0].get("final_update") == fin.get("final_update")
                   and stops[0].get("stop_reason") == fin.get("stop_reason"), "run_meta", "stop rows")
            t2u = {v["update"] for v in summ if v.get("task2")}
            rep.ok("rl.v18_validation", t2u <= {0} | {c["update"] for c in checks}, "validation.jsonl",
                   "Task 2 validations at %s (only u0 and the checked candidates)" % sorted(t2u))
        rep.ok("rl.v18_validation", all(u in metrics for u in [0] + sorted(upd)), "validation.jsonl",
               "an update without its Task 1 validation")
        rep.ok("rl.v18_validation", 0 in {v["update"] for v in summ if v.get("task2")}, "validation.jsonl",
               "u0 has no Task 2 validation")
    # ---------------------------------------------------------------- test (§8)
    tm_rows = VP._jl(os.path.join(rl_dir, "test_meta.jsonl"))
    if tm_rows:
        t_ = tm_rows[-1]
        rep.ok("rl.v18_test", fin is not None and t_.get("updates") == [fin.get("final_update")] and not t_.get("include_base"),
               "test_meta", "tested updates %r (v18: the final policy only)" % t_.get("updates"))
        test_rows = VP._jl(os.path.join(rl_dir, "test.jsonl"))
        if fin is not None:
            for msg in check_test_rows(rl_dir, test_rows, fin.get("final_update"), set(f.get("test") or []),
                                       set(f.get("test_all") or []), list(t_.get("seeds") or []), real_vllm):
                rep.ok("rl.v18_test", False, "test.jsonl", msg)
            rep.ok("rl.v18_test", {r.get("update") for r in test_rows if r.get("kind")} <= {fin.get("final_update")},
                   "test.jsonl", "test rows of another update than the final")
        ref = t_.get("v17_reference") or {}
        rep.ok("rl.v18_test", bool(ref.get("test_jsonl_sha256")) and (not os.path.exists(ref.get("test_jsonl", ""))
                                                                      or VP.sha256_file(ref["test_jsonl"]) == ref["test_jsonl_sha256"]),
               "test_meta", "the v17 comparison file's sha is not recorded / changed")
    rep._c("rl.v18_test")


def check_test_rows(run_dir, rows, u, test, test_all, seeds, real_vllm):
    """SPEC v18 §8 (audit B SHOULD 2; v17 eval_test_boot's checks): the test rows of the final policy u -- the policy sha
    is ckpt u's state.json (not a summary row's), every row of u carries it, split "test"; the Task 2 episodes are exactly
    test x seeds, the Task 1 rows exactly test_all; on a real vLLM run every Task 2 Planner step, Task 1 turn and Task 1
    probe was generated by ckpt u's served adapter; the unscored-point rule; the probes are the decision points t >= 2,
    sum(n - 1) (task1_eval_row probes t >= 2; the v18 turn-1 positions are training groups, not stop decisions).
    -> list of failure strings."""
    out = []
    st = os.path.join(run_dir, "ckpt", "u%05d" % int(u), "state.json")
    if not os.path.exists(st):
        return ["ckpt u%s has no state.json" % u]
    psha = json.load(open(st, encoding="utf-8"))["policy_sha"]
    want_ad = VP.adapter_name(run_dir, int(u)) if real_vllm else None
    eps, t1 = {}, {}
    for r in rows:
        if r.get("update") != u:
            continue
        if r.get("policy_sha") != psha:
            out.append("u%s %s row of another policy sha %s" % (u, r.get("kind"), str(r.get("policy_sha"))[:12]))
            continue
        if r.get("kind") in ("episode", "task1") and r.get("split") != "test":
            out.append("u%s row with split %r" % (u, r.get("split")))
        if r.get("kind") == "episode":
            if r["conversation_id"] not in test or r["seed"] not in seeds:
                out.append("u%s episode %s s%s outside the test plan" % (u, r["conversation_id"], r["seed"]))
            eps[(r["conversation_id"], r["seed"])] = r
            fits = [s.get("planner_fit") for s in r["episode"].get("trace") or [] if s.get("planner_fit")]
            if want_ad is not None and (not fits or any(f.get("gen_adapter") != want_ad for f in fits)):
                out.append("u%s %s s%s: Task 2 Planner steps not served by %s" % (u, r["conversation_id"], r["seed"], want_ad))
        elif r.get("kind") == "task1":
            if r["conversation_id"] not in test_all:
                out.append("u%s Task 1 id %s outside test_all" % (u, r["conversation_id"]))
            t1[r["conversation_id"]] = r
    summ = [r for r in rows if r.get("kind") == "summary" and r.get("update") == u]
    if not summ or summ[-1].get("policy_sha") != psha:
        out.append("u%s: no test summary with ckpt u%s's policy sha" % (u, u))
    if sorted(eps) != sorted((c, s) for c in test for s in seeds):
        out.append("u%s: Task 2 episodes != test x seeds" % u)
    if sorted(t1) != sorted(test_all):
        out.append("u%s: Task 1 conversations != test_all" % u)
    pts = []
    for cid, r in t1.items():
        fits = [x.get("planner_fit") for x in r["task1"]["turns"] if x.get("planner_fit")]
        if want_ad is not None and (not fits or any(f.get("gen_adapter") != want_ad for f in fits)):
            out.append("u%s %s: Task 1 turns not served by %s" % (u, cid, want_ad))
        pts += [dict(p, cid=cid) for p in r.get("end_probs") or []]
    if want_ad is not None and any(p.get("gen_adapter") != want_ad for p in pts):
        out.append("u%s: Task 1 probes not generated by %s" % (u, want_ad))
    n_exp = sum(len(r["task1"]["turns"]) - 1 for r in t1.values())
    if len(pts) != n_exp or any(int(p["t"]) < 2 for p in pts):
        out.append("u%s: %d probe points, expected sum(n - 1) = %d at t >= 2" % (u, len(pts), n_exp))
    for p in pts:
        if not VP._unscored_ok(p):
            out.append("u%s %s t%s: the unscored-point rule is broken (p_end %r)" % (u, p["cid"], p["t"], p["p_end"]))
    return out


def VP_sha_dir(p):
    """The trainer's sha_path of a directory (relative path + file sha, sorted)."""
    import hashlib
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(p)):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            h.update(os.path.relpath(fp, p).replace("\\", "/").encode())
            h.update(VP.sha256_file(fp).encode())
    return h.hexdigest()
