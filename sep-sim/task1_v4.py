#!/usr/bin/env python3
"""Task 1 (teacher-forced) generations of the pend architecture, in the benchmark generations schema.

Runs Task2Env.task1_generate (the same Planner + Ditto + Selector code as Task 2) over real finished
sessions and writes one row per real user turn, ready for tools/score_method.py. Resumable (a
conversation whose rows are all present is skipped), threaded (GPU calls are serialised inside Task2Env).

  --sessions all              every finished session of the corpus (no training involved: pre-RL use)
  --sessions fold-validation  splits[fold].validation
  --sessions fold-test        splits[fold].test, needs --final (method frozen)
A trained --planner-adapter must carry a manifest whose training scenarios avoid the sessions scored here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def sha_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", choices=("all", "fold-validation", "fold-test"), required=True)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--splits", default="/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json")
    ap.add_argument("--planner-path", required=True)
    ap.add_argument("--planner-adapter", default="")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--implicit-profile", type=int, choices=(0, 1), default=0)
    ap.add_argument("--selector", choices=("length", "borda"), default="length",
                    help="borda: length + style rank sum (style_select.py); length: the E1.6 rule")
    ap.add_argument("--fewshot", choices=("off", "loo", "fold"), default="off",
                    help="loo: all finished sessions (leave-one-out: never the same conversation, goal or persona); "
                         "fold: splits[fold].train_all only")
    ap.add_argument("--batch", type=int, choices=(0, 1), default=1)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--out", required=True, help="generations .jsonl")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    if a.sessions == "fold-test" and not a.final:
        raise SystemExit("LEAK GATE: the test sessions need --final (method frozen)")
    from task2_env import Task2Env, PlannerLM, setup_environment, make_fewshot_pool
    setup_environment("pend")
    from sepsim import pipeline
    planner = PlannerLM(a.planner_path, a.gpu, adapter=a.planner_adapter or None)
    env = Task2Env("pend", a.gpu, planner, judge=None, batch=bool(a.batch), max_batch=a.max_batch,
                   implicit_profile=bool(a.implicit_profile), selector=a.selector)
    finished = [cid for cid, r in env.recs.items()
                if any(m.get("is_final") is True for m in r.get("chat_messages", []))]
    if a.fewshot == "loo":
        if a.sessions != "all":
            raise SystemExit("--fewshot loo is for --sessions all; fold runs use --fewshot fold")
        env.fewshot = make_fewshot_pool(env.recs, finished)
    elif a.fewshot == "fold":
        if a.fold < 0:
            raise SystemExit("--fewshot fold needs --fold")
        spf = [x for x in json.load(open(a.splits, encoding="utf-8"))["folds"] if x["fold"] == a.fold][0]
        env.fewshot = make_fewshot_pool(env.recs, spf["train_all"])
        forb = set(spf["forbidden_for_training"])
        assert not set(spf["train_all"]) & forb, "few-shot pool intersects validation/test"
    if a.sessions == "all":
        cids = sorted(finished)
        if a.planner_adapter:
            raise SystemExit("LEAK GATE: a trained adapter cannot be scored on ALL sessions (train included)")
    else:
        sp = json.load(open(a.splits, encoding="utf-8"))
        f = [x for x in sp["folds"] if x["fold"] == a.fold][0]
        cids = sorted(f["validation" if a.sessions == "fold-validation" else "test"])
        if a.planner_adapter:
            man = None
            for name in ("rl_manifest.json", "train_manifest.json", "run_manifest.json"):
                p = os.path.join(a.planner_adapter, name)
                if os.path.exists(p):
                    man = json.load(open(p, encoding="utf-8"))
            if man is None:
                raise SystemExit("LEAK GATE: adapter has no manifest")
            used = set(man.get("train_scenarios", [])) | set(man.get("train_conversations", []))
            if used & set(cids):
                raise SystemExit("LEAK GATE: adapter was trained on sessions scored here: %s" % sorted(used & set(cids))[:5])
    if a.limit:
        cids = cids[: a.limit]
    done = {}
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done.setdefault(r["conversation_id"], set()).add(r["turn_index"])
    k1_path = a.out + ".k1.jsonl"
    k1_done = set()
    if os.path.exists(k1_path):
        k1_done = {json.loads(l)["conversation_id"] for l in open(k1_path, encoding="utf-8") if l.strip()}
    todo = []
    for cid in cids:
        users, _ = pipeline.split_messages(env.recs[cid])
        if done.get(cid) != set(range(1, len(users) + 1)) or cid not in k1_done:
            todo.append(cid)
    lock = threading.Lock()
    meta = {"describe": env.describe(), "sessions": a.sessions, "fold": a.fold, "n_sessions": len(cids),
            "code_sha256": {n: sha_file(os.path.join(HERE, n)) for n in
                            ("task1_v4.py", "task2_env.py", "planner_prompt_v3.py", "fit_prompts.py", "ditto_e16.py",
                             "implicit_profile.py", "batching.py", "style_select.py")},
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
    t0 = time.time()

    def one(cid):
        rows, k1 = env.task1_generate(cid)
        with lock:
            # a partial conversation from an interrupted run is replaced as a whole
            if cid in done and os.path.exists(a.out):
                keep = [l for l in open(a.out, encoding="utf-8") if l.strip() and json.loads(l)["conversation_id"] != cid]
                open(a.out, "w", encoding="utf-8").writelines(keep)
            with open(a.out, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if cid in k1_done:
                keep = [l for l in open(k1_path, encoding="utf-8") if l.strip() and json.loads(l)["conversation_id"] != cid]
                open(k1_path, "w", encoding="utf-8").writelines(keep)
            with open(k1_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(k1, ensure_ascii=False) + "\n")
            print("  %s turns=%d ends=%s k1_end=%s (%.0fs)" % (cid[:10], len(rows),
                  [r["turn_index"] for r in rows if r["greedy_ended"]], k1["ended"], time.time() - t0), flush=True)

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        list(ex.map(one, todo))
    print("DONE", json.dumps({"sessions": len(cids), "generated": len(todo)}), flush=True)


if __name__ == "__main__":
    main()
