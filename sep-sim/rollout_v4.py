# -*- coding: utf-8 -*-
"""Evaluation rollouts on one split of one fold, through the shared Task2Env.

  --arm a0|a2 --fold F --split validation|test|train --splits splits_v1.json
  --planner-path P [--planner-nf4] [--planner-dtype bfloat16|float16|auto] [--planner-adapter DIR]
  --judge-base B --judge-adapter DIR   (a2)
  --replicate R --gpu G --out-dir D [--limit N] [--final]

Leak gates (refuse to run on violation):
  * scenarios = splits[F][split]; split "test" requires --final (method frozen);
  * a2 judge: its train_manifest.json must name fold F, its train_scenarios must be a subset of
    splits[F]["train"] and disjoint from splits[F]["forbidden_for_training"];
  * planner adapter (RL/SFT-trained): if it carries a manifest with a fold, it must be F, and its
    training scenarios (if listed) must satisfy the same two conditions.
Writes <arm>.jsonl (append-only, resumable) and run_meta_<arm>_rep<R>.json with env.describe(),
code SHA256s, split SHA256 and the gate report.
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


SPLITS_SHA = {"sha": None}


def check_manifest(path, fold, split_fold, what):
    if not os.path.exists(path):
        return None
    m = json.load(open(path))
    if "fold" in m and int(m["fold"]) != fold:
        raise SystemExit("LEAK GATE: %s trained for fold %s, evaluating fold %d" % (what, m["fold"], fold))
    if m.get("splits_sha256") and SPLITS_SHA.get("sha") and m["splits_sha256"] != SPLITS_SHA["sha"]:
        raise SystemExit("LEAK GATE: %s was trained with another split file (sha differs)" % what)
    train = set(m.get("train_scenarios") or [])
    if train and not train <= set(split_fold["train"]):
        raise SystemExit("LEAK GATE: %s trained on non-train scenarios %s" % (what, sorted(train - set(split_fold["train"]))[:3]))
    bad = train & set(split_fold["forbidden_for_training"])
    if bad:
        raise SystemExit("LEAK GATE: %s trained on validation/test scenarios %s" % (what, sorted(bad)[:3]))
    # RL manifest: Task 1 conversations, few-shot pool (train_all) must avoid validation/test too
    other = set(m.get("train_conversations") or []) | set(m.get("fewshot_pool") or [])
    bad = other & set(split_fold["forbidden_for_training"])
    if bad:
        raise SystemExit("LEAK GATE: %s used validation/test conversations %s" % (what, sorted(bad)[:3]))
    return {"manifest": path, "sha256": sha_file(path), "n_train_scenarios": len(train)}


def check_rl_settings(adapter, settings):
    """An RL adapter is evaluated only under the settings it was trained with, and only when no init adapter
    was merged into its base (this CLI loads the LoRA on the plain base)."""
    for d in (adapter, os.path.dirname(os.path.abspath(adapter.rstrip("/\\")))):
        p = os.path.join(d, "rl_manifest.json")
        if os.path.exists(p):
            m = json.load(open(p, encoding="utf-8"))
            if m.get("init_adapter"):
                raise SystemExit("adapter was trained on top of init adapter %s: evaluating it on the plain base is wrong" % m["init_adapter"])
            norm = lambda k, x: os.path.normpath(str(x)) if (k == "planner_path" and x) else x
            diff = {k: (m.get(k), v) for k, v in settings.items() if k in m and norm(k, m.get(k)) != norm(k, v)}
            if diff:
                raise SystemExit("adapter trained with other settings than this evaluation: %r" % diff)
            return m
    raise SystemExit("LEAK GATE: RL adapter without rl_manifest.json (next to it or in its directory)")


def make_planner(PlannerLM, path, gpu, adapter, backend, url, **hf_kw):
    """HF: the model on this GPU (legacy / ablation). vLLM (spec): only the tokenizer here; generation on the vLLM
    server, the adapter (if any) loaded there under a name carrying its sha."""
    if backend == "hf":
        return PlannerLM(path, gpu, adapter=adapter or None, **hf_kw)
    import vllm_planner
    planner = PlannerLM(path, gpu, load_model=False)
    vllm_planner.VLLMPlanner(url).attach(planner)
    if adapter:
        planner.remote.use_adapter(adapter, "eval")
    planner.adapter = adapter or None
    return planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("a0", "a2", "e16", "final", "pend"), required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--split", choices=("train", "validation", "test"), required=True)
    ap.add_argument("--splits", default="/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json")
    ap.add_argument("--planner-path", required=True)
    ap.add_argument("--planner-nf4", action="store_true")
    ap.add_argument("--planner-dtype", default="bfloat16")
    ap.add_argument("--planner-adapter", default="")
    ap.add_argument("--planner-backend", choices=("vllm", "hf"), default=None,
                    help="pend spec: vllm (the same generation backend as training); other arms: hf")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8031/v1")
    ap.add_argument("--judge-base", default="")
    ap.add_argument("--judge-adapter", default="")
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="required for the test split")
    ap.add_argument("--implicit-profile", type=int, choices=(0, 1), default=None, help="pend spec: 1")
    ap.add_argument("--fewshot", choices=("off", "fold"), default=None,
                    help="fold: Speaker few-shot examples from splits[fold].train_all only (pend arm; pend spec)")
    ap.add_argument("--selector", choices=("length", "borda"), default=None, help="pend spec: borda")
    ap.add_argument("--planner-temperature", type=float, default=None,
                    help="pend spec (D5): 0.7, the sampled Planner, one episode per seed 0 and 1; other arms: 0")
    ap.add_argument("--planner-top-p", type=float, default=1.0)
    ap.add_argument("--ablation", default=None, help="name of a declared ablation; required for any non-spec pend setting")
    ap.add_argument("--batch", type=int, choices=(0, 1), default=1)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1, help="episodes in threads (useful with --batch 1)")
    ap.add_argument("--smoke", action="store_true",
                    help="pipeline smoke only: judge may be the untrained base (no adapter); never on test; "
                         "out-dir must contain 'smoke'; rows are not results")
    args = ap.parse_args()
    spec = {"implicit_profile": 1, "fewshot": "fold", "selector": "borda", "planner_temperature": 0.7,
            "planner_backend": "vllm"} \
        if args.arm == "pend" else {"implicit_profile": 0, "fewshot": "off", "selector": "length", "planner_temperature": 0.0,
                                    "planner_backend": "hf"}
    for k, v in spec.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    off = {k: getattr(args, k) for k in spec if getattr(args, k) != spec[k]}
    if args.arm == "pend" and "Qwen3-4B-Instruct-2507" not in args.planner_path:
        off["planner_path"] = args.planner_path          # the spec's Planner
    if off and not args.ablation:
        raise SystemExit("settings %r differ from the %s spec; name the ablation with --ablation" % (off, args.arm))
    if args.arm != "pend" and (args.implicit_profile or args.fewshot != "off" or args.selector != "length"):
        raise SystemExit("Implicit Profile / few-shot / Borda selector exist only for the pend arm")
    if args.split == "test" and not args.final:
        raise SystemExit("LEAK GATE: the test split needs --final (method frozen)")
    if args.smoke and (args.split == "test" or "smoke" not in args.out_dir):
        raise SystemExit("--smoke: not on test, and --out-dir must contain 'smoke'")
    splits = json.load(open(args.splits))
    sf = [f for f in splits["folds"] if f["fold"] == args.fold][0]
    scen = list(sf[args.split])
    gate = {"fold": args.fold, "split": args.split, "n_scenarios": len(scen), "splits_sha256": sha_file(args.splits)}
    SPLITS_SHA["sha"] = gate["splits_sha256"]
    if args.arm in ("a2", "final") and args.smoke and not args.judge_adapter:
        if not args.judge_base:
            raise SystemExit("--smoke still needs --judge-base")
        gate["judge"] = {"smoke_untrained_base": args.judge_base}
    elif args.arm in ("a2", "final"):
        if not (args.judge_base and args.judge_adapter):
            raise SystemExit("%s needs --judge-base and --judge-adapter" % args.arm)
        g = check_manifest(os.path.join(args.judge_adapter, "train_manifest.json"), args.fold, sf, "goal judge")
        if g is None:
            raise SystemExit("LEAK GATE: judge adapter has no train_manifest.json")
        gate["judge"] = g
    if args.planner_adapter:
        found = None
        # train_planner_rl writes ckpt/uNNNNN/rl_manifest.json next to ckpt/uNNNNN/adapter
        for d in (args.planner_adapter, os.path.dirname(os.path.abspath(args.planner_adapter.rstrip("/\\")))):
            for name in ("train_manifest.json", "rl_manifest.json", "run_manifest.json"):
                found = check_manifest(os.path.join(d, name), args.fold, sf, "planner adapter") or found
        if found is None:
            raise SystemExit("LEAK GATE: planner adapter has no manifest; cannot verify its training data")
        gate["planner_adapter"] = found
        if args.arm == "pend":           # pend adapters come from train_planner_rl: rl_manifest.json required
            check_rl_settings(args.planner_adapter, {"arm": args.arm, "implicit_profile": args.implicit_profile,
                                                     "fewshot": args.fewshot, "selector": args.selector,
                                                     "planner_path": args.planner_path,
                                                     **({} if args.ablation else {"planner_backend": args.planner_backend})})
    print("leak gate OK", json.dumps(gate), flush=True)

    from task2_env import Task2Env, PlannerLM, setup_environment, make_fewshot_pool
    setup_environment(args.arm)
    import goal_judge as GJ
    planner = make_planner(PlannerLM, args.planner_path, args.gpu, args.planner_adapter, args.planner_backend,
                           args.vllm_url, nf4=args.planner_nf4, dtype=args.planner_dtype)
    judge = GJ.GoalJudge(args.judge_base, adapter=args.judge_adapter or None, gpu=args.gpu).load() if args.arm in ("a2", "final") else None
    env = Task2Env(args.arm, args.gpu, planner, judge=judge, batch=bool(args.batch), max_batch=args.max_batch,
                   implicit_profile=bool(args.implicit_profile), selector=args.selector)
    if args.fewshot == "fold":
        pool_ids = list(sf["train_all"])
        assert not set(pool_ids) & set(sf["forbidden_for_training"]), "few-shot pool intersects validation/test"
        assert not set(pool_ids) & set(scen) or args.split == "train", "few-shot pool contains scored scenarios"
        env.fewshot = make_fewshot_pool(env.recs, pool_ids)
        gate["fewshot_pool"] = {"source": "train_all", "n": len(pool_ids)}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "%s.jsonl" % args.arm)
    done = set()
    if os.path.exists(out):
        lines = [l for l in open(out, encoding="utf-8") if l.strip()]
        for i, l in enumerate(lines):
            try:
                r = json.loads(l)
            except ValueError:
                if i != len(lines) - 1:
                    raise SystemExit("%s: undecodable line %d (not the last one): the file is corrupt" % (out, i + 1))
                # a torn last line after a crash: cut it off (the next row must not merge into it); regenerated
                with open(out, "rb+") as f:
                    data = f.read()
                    cut = data.rstrip(b"\n").rfind(b"\n")
                    f.seek(0)
                    f.truncate(cut + 1 if cut >= 0 else 0)
                continue
            done.add((r["conversation_id"], r["seed"], r.get("replicate", 0)))
    code = {n: sha_file(os.path.join(HERE, n)) for n in ("rollout_v4.py", "task2_env.py", "task2_episode.py", "ditto_e16.py", "task1_stop.py",
                                                         "fit_prompts.py", "planner_prompt_v3.py", "goal_judge.py",
                                                         "implicit_profile.py", "style_select.py", "batching.py")}
    meta = {"env": env.describe(), "gate": gate, "code_sha256": code, "replicate": args.replicate, "smoke": args.smoke,
            "planner_nf4": args.planner_nf4, "planner_dtype": args.planner_dtype,
            "settings": {"implicit_profile": args.implicit_profile, "fewshot": args.fewshot, "selector": args.selector,
                         "planner_temperature": args.planner_temperature, "planner_top_p": args.planner_top_p,
                         "planner_backend": args.planner_backend,
                         "ablation": args.ablation},
            "goal_judge_system_sha256": hashlib.sha256(GJ.SYSTEM.encode()).hexdigest() if judge else None,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(os.path.join(args.out_dir, "run_meta_%s_rep%d.json" % (args.arm, args.replicate)), "w"), indent=1)
    jobs = [(c, s) for c in scen for s in (0, 1)]
    if args.limit:
        jobs = jobs[: args.limit]
    t0 = time.time()
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()

    errors = []

    def one(job):
        cid, seed = job
        te = time.time()
        try:
            row = env.run_episode(cid, seed, replicate=args.replicate, planner_temperature=args.planner_temperature,
                                  planner_top_p=args.planner_top_p)
        except Exception as e:                      # recorded, never swallowed: the run exits non-zero
            import traceback
            with lock:
                errors.append({"conversation_id": cid, "seed": seed, "error": repr(e), "trace": traceback.format_exc()[-2000:]})
                with open(out + ".errors.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(errors[-1]) + "\n")
            print("  ERROR %s s%d %r" % (cid[:8], seed, e), flush=True)
            return
        row.update(fold=args.fold, split=args.split, wall_seconds=round(time.time() - te, 1))
        with lock:
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("  %s %s s%d emitted=%d human=%s end=%s cov=%.3f (%.0fs)" % (
                args.arm, cid[:8], seed, row["emitted_user_turns"], row.get("human_turns"), row["end_kind"],
                row["coverage"], time.time() - t0), flush=True)

    todo = [j for j in jobs if (j[0], j[1], args.replicate) not in done]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        list(ex.map(one, todo))
    if errors:
        print("ERRORS", len(errors), flush=True)
    print("DONE", json.dumps({"episodes": len(jobs), "errors": len(errors), "planner_calls": planner.n_calls,
                              "judge_unparsed": judge.n_unparsed if judge else 0,
                              "speaker_regen": env.speaker.n_regen}), flush=True)
    if errors:
        raise SystemExit("%d episode(s) failed; see %s.errors.jsonl" % (len(errors), out))


if __name__ == "__main__":
    main()
