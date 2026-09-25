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


def check_manifest(path, fold, split_fold, what):
    if not os.path.exists(path):
        return None
    m = json.load(open(path))
    if "fold" in m and int(m["fold"]) != fold:
        raise SystemExit("LEAK GATE: %s trained for fold %s, evaluating fold %d" % (what, m["fold"], fold))
    train = set(m.get("train_scenarios") or [])
    if train and not train <= set(split_fold["train"]):
        raise SystemExit("LEAK GATE: %s trained on non-train scenarios %s" % (what, sorted(train - set(split_fold["train"]))[:3]))
    bad = train & set(split_fold["forbidden_for_training"])
    if bad:
        raise SystemExit("LEAK GATE: %s trained on validation/test scenarios %s" % (what, sorted(bad)[:3]))
    return {"manifest": path, "sha256": sha_file(path), "n_train_scenarios": len(train)}


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
    ap.add_argument("--judge-base", default="")
    ap.add_argument("--judge-adapter", default="")
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="required for the test split")
    ap.add_argument("--implicit-profile", type=int, choices=(0, 1), default=0)
    ap.add_argument("--fewshot", choices=("off", "fold"), default="off",
                    help="fold: Speaker few-shot examples from splits[fold].train_all only (pend arm)")
    ap.add_argument("--selector", choices=("length", "borda"), default="length")
    ap.add_argument("--batch", type=int, choices=(0, 1), default=0)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1, help="episodes in threads (useful with --batch 1)")
    ap.add_argument("--smoke", action="store_true",
                    help="pipeline smoke only: judge may be the untrained base (no adapter); never on test; "
                         "out-dir must contain 'smoke'; rows are not results")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("LEAK GATE: the test split needs --final (method frozen)")
    if args.smoke and (args.split == "test" or "smoke" not in args.out_dir):
        raise SystemExit("--smoke: not on test, and --out-dir must contain 'smoke'")
    splits = json.load(open(args.splits))
    sf = [f for f in splits["folds"] if f["fold"] == args.fold][0]
    scen = list(sf[args.split])
    gate = {"fold": args.fold, "split": args.split, "n_scenarios": len(scen), "splits_sha256": sha_file(args.splits)}
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
        for name in ("train_manifest.json", "rl_manifest.json", "run_manifest.json"):
            found = check_manifest(os.path.join(args.planner_adapter, name), args.fold, sf, "planner adapter") or found
        if found is None:
            raise SystemExit("LEAK GATE: planner adapter has no manifest; cannot verify its training data")
        gate["planner_adapter"] = found
    print("leak gate OK", json.dumps(gate), flush=True)

    from task2_env import Task2Env, PlannerLM, setup_environment, make_fewshot_pool
    setup_environment(args.arm)
    import goal_judge as GJ
    planner = PlannerLM(args.planner_path, args.gpu, nf4=args.planner_nf4, dtype=args.planner_dtype,
                        adapter=args.planner_adapter or None)
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
        done = {(json.loads(l)["conversation_id"], json.loads(l)["seed"]) for l in open(out)}
    code = {n: sha_file(os.path.join(HERE, n)) for n in ("rollout_v4.py", "task2_env.py", "task2_episode.py", "ditto_e16.py", "task1_stop.py",
                                                         "fit_prompts.py", "planner_prompt_v3.py", "goal_judge.py",
                                                         "implicit_profile.py", "style_select.py", "batching.py")}
    meta = {"env": env.describe(), "gate": gate, "code_sha256": code, "replicate": args.replicate, "smoke": args.smoke,
            "planner_nf4": args.planner_nf4, "planner_dtype": args.planner_dtype,
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

    def one(job):
        cid, seed = job
        te = time.time()
        row = env.run_episode(cid, seed, replicate=args.replicate)
        row.update(fold=args.fold, split=args.split, wall_seconds=round(time.time() - te, 1))
        with lock:
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("  %s %s s%d emitted=%d human=%s end=%s cov=%.3f (%.0fs)" % (
                args.arm, cid[:8], seed, row["emitted_user_turns"], row.get("human_turns"), row["end_kind"],
                row["coverage"], time.time() - t0), flush=True)

    todo = [j for j in jobs if j not in done]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        list(ex.map(one, todo))
    print("DONE", json.dumps({"episodes": len(jobs), "planner_calls": planner.n_calls,
                              "judge_unparsed": judge.n_unparsed if judge else 0,
                              "speaker_regen": env.speaker.n_regen}), flush=True)


if __name__ == "__main__":
    main()
