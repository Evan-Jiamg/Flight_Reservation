"""Test-split evaluation of a finished pend GRPO run (user 2026-09-28: "先測 u5" / "u5 和 u0，u10 先不要", 8 seeds):
the re-selected checkpoint (reselect_best.json) and the untrained policy u0, on splits[fold].test (Task 2, the ids with
requirement shards) and splits[fold].test_all (Task 1, as the benchmark / task1_v4.py), with exactly the validation
procedure of train_planner_rl.validate(): Task 2 = sampled Planner (--val-temperature, one replicate per seed, an
unclean episode re-run up to VAL_RETRIES times), Task 1 = greedy run + teacher-forced end probabilities
(Trainer.task1_eval_row). Writes RUN/test.jsonl (episode / task1 / summary rows, split "test") and RUN/test_meta.jsonl;
never writes a training file (checkpoints, validation.jsonl, best.json, reselect*).

Gates: --final; the trainer arguments are the run's own (check_provenance: same code, splits, arguments); the updates are
exactly {0, re-selected best}; every test id is in forbidden_for_training and in no training / validation / few-shot /
p_h list; an update whose test summary exists is never evaluated again (test once), an unfinished one is resumed.

Usage: python eval_test_rl.py --final --test-seeds 0 1 2 3 4 5 6 7 --test-updates 0 5  <the trainer arguments of the run>
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import train_planner_rl as TP
import task1_stop as T1

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--final", action="store_true", help="required: this reads the test split")
    ap.add_argument("--test-seeds", type=int, nargs="+", required=True)
    ap.add_argument("--test-updates", type=int, nargs="+", required=True)
    own, rest = ap.parse_known_args(argv)
    if not own.final:
        ap.error("the test split is read only with --final")
    a = TP.parse_args(rest)
    if a.reselect_seeds or a.resume:
        ap.error("pass the trainer arguments without --reselect-* / --resume")
    return own, a


def test_ids(a, split):
    """splits[fold].test / test_all, read here only (load_split never keeps them) and checked against every
    training-side list."""
    d = json.load(open(a.splits, encoding="utf-8"))
    assert TP.sha_file(a.splits) == split["sha256"], "splits file changed while loading"
    f = {int(x["fold"]): x for x in d["folds"]}[int(a.fold)]
    test, test_all = sorted(f["test"]), sorted(f["test_all"])
    for k, ids in (("test", test), ("test_all", test_all)):
        n = (f.get("sizes") or {}).get(k)
        assert n is None or n == len(ids), "splits fold %d: %s has %d ids, declared %d" % (a.fold, k, len(ids), n)
        assert ids and len(set(ids)) == len(ids), "%s is empty or has duplicates" % k
    assert set(test) <= set(test_all) <= split["forbidden"], "test must lie in test_all, test_all in forbidden_for_training"
    for name in ("train", "train_all", "validation"):
        assert not set(test_all) & set(split[name]), "test_all intersects %s" % name
    assert not set(test_all) & set(f.get("validation_all") or []), "test_all intersects validation_all"
    return test, test_all


def summarize(tr, u, psha, seeds, vrows, t1rows, t0):
    """Summary of one update: validate()'s turn_stats / Task 1 metrics (no selection, best or D2 fields)."""
    t_max = int(tr.selection_cfg["t_max"])

    def turn_stats(rows):
        eps = [r["episode"] for r in rows if r["episode"]["clean"]]
        if not eps or not all("human_turns" in e for e in eps):
            return None
        d = [e["emitted_user_turns"] - e["human_turns"] for e in eps]
        return {"n_episodes": len(eps),
                "sim_turns_mean": sum(e["emitted_user_turns"] for e in eps) / len(eps),
                "human_turns_mean": sum(e["human_turns"] for e in eps) / len(eps),
                "abs_diff_mean": sum(abs(x) for x in d) / len(d),
                "coverage_mean": sum(float(e["coverage"]) for e in eps) / len(eps),
                "turn_w1": TP.turn_w1([e["emitted_user_turns"] for e in eps],
                                      [min(int(e["human_turns"]), t_max) for e in eps]),
                "end_kinds": {k: sum(e["end_kind"] == k for e in eps) for k in sorted({e["end_kind"] for e in eps})}}

    n_unclean = sum(1 for r in vrows if not r["episode"]["clean"])
    t1 = T1.task1_stop_metrics([r["task1"] for r in t1rows])
    t1.update(T1.task1_prob_metrics([p for r in t1rows for p in r["end_probs"]]))
    return {"kind": "summary", "update": u, "policy_sha": psha, "split": "test", "final": True,
            "n_episodes": len(vrows) - n_unclean, "n_unclean_episodes": n_unclean,
            "unclean_after_retries": n_unclean > 0, "val_temperature": tr.a.val_temperature, "seeds": list(seeds),
            "task2_ids": sorted({r["conversation_id"] for r in vrows}),
            "task1_ids": sorted(r["conversation_id"] for r in t1rows),
            "turn_stats": turn_stats(vrows),
            # the spec's D5 seeds (0 and 1) alone: comparable with a two-seed evaluation
            "turn_stats_seeds01": turn_stats([r for r in vrows if r["seed"] in (0, 1)]),
            "task1": t1, "test_s": round(time.time() - t0, 1), "time": time.time()}


def evaluate(tr, u, seeds, test, test_all, p_out):
    t0 = time.time()
    prev = TP.read_jsonl(p_out)
    for r in prev:
        if r.get("kind") == "summary" and r["update"] == u:
            print("u%d: test summary exists - not evaluated again" % u, flush=True)
            return r
    d = tr.ckpt_dir(u)
    cst = json.load(open(os.path.join(d, "state.json")))
    tr.learner.load_policy(d)
    psha = tr.learner.policy_sha()
    if psha != cst["policy_sha"]:
        raise AssertionError("checkpoint u%d: loaded policy sha differs from its record" % u)
    tr.sync_generation_policy(u)            # vLLM generates with this checkpoint's adapter (name p<u>-<sha>)
    a = tr.a
    done = {(r["conversation_id"], r["seed"]): r for r in prev if r.get("kind") == "episode" and r["update"] == u}
    done_t1 = {r["conversation_id"]: r for r in prev if r.get("kind") == "task1" and r["update"] == u}

    def run_ep(job):
        cid, s = job
        assert cid in test, "Task 2 id %r not in the test split" % cid
        r = done.get((cid, s))
        if r is not None and r["policy_sha"] == psha and (r["episode"]["clean"] or r.get("attempt", 0) >= TP.VAL_RETRIES):
            return r
        attempt = r.get("attempt", 0) + 1 if (r is not None and r["policy_sha"] == psha) else 0
        while True:
            ep = tr.env.run_episode(cid, seed=s, replicate=0, planner_temperature=a.val_temperature,
                                    planner_top_p=a.val_top_p, record_generation=False)
            r = {"kind": "episode", "update": u, "policy_sha": psha, "conversation_id": cid, "seed": s,
                 "attempt": attempt, "split": "test", "episode": ep, "time": time.time()}
            with tr.io_lock:
                TP.append_jsonl(p_out, r)
            if ep["clean"] or attempt >= TP.VAL_RETRIES:
                return r
            attempt += 1                    # an infrastructure incident, not the policy: run it again

    def run_t1(cid):
        assert cid in test_all, "Task 1 id %r not in test_all" % cid
        r = done_t1.get(cid)
        if r is None or r["policy_sha"] != psha or "end_probs" not in r:
            r = tr.task1_eval_row(u, psha, cid)
            r["split"] = "test"
            with tr.io_lock:
                TP.append_jsonl(p_out, r)
        return r

    with ThreadPoolExecutor(max_workers=max(1, a.rollout_workers)) as ex:
        vrows = list(ex.map(run_ep, [(c, s) for c in test for s in seeds]))
        t1rows = list(ex.map(run_t1, test_all))
    s = summarize(tr, u, psha, seeds, vrows, t1rows, t0)
    TP.append_jsonl(p_out, s)
    return s


def main(argv=None):
    own, a = parse(argv)
    best = json.load(open(os.path.join(a.out, "reselect_best.json")))
    bu = (best.get("best") or {}).get("update")
    if bu is None:
        raise SystemExit("reselect_best.json has no re-selected checkpoint")
    if sorted(set(own.test_updates)) != sorted({0, bu}) or len(own.test_updates) != len(set(own.test_updates)):
        raise SystemExit("--test-updates must be exactly u0 and the re-selected u%d, got %s" % (bu, own.test_updates))
    validated = {r["update"] for r in TP.read_jsonl(os.path.join(a.out, "validation.jsonl")) if r.get("kind") == "summary"}
    if set(own.test_updates) - validated:
        raise SystemExit("updates %s were never validated" % sorted(set(own.test_updates) - validated))
    if not TP.read_jsonl(os.path.join(a.out, "run_meta.jsonl")):
        raise SystemExit("%s has no run_meta.jsonl: provenance cannot be checked" % a.out)
    tr = TP.Trainer(a)
    test, test_all = test_ids(a, tr.split)
    for u in own.test_updates:
        man = json.load(open(os.path.join(tr.ckpt_dir(u), "rl_manifest.json")))
        assert man["fold"] == a.fold and man["splits_sha256"] == tr.split["sha256"], "u%d manifest: other fold / splits" % u
        for k in ("train_scenarios", "train_conversations", "fewshot_pool"):
            assert not set(man[k]) & set(test_all), "u%d manifest %s intersects the test split" % (u, k)
    tr.build()                              # few-shot pool and p_h from train_all (asserted disjoint from forbidden)
    tr.update_done = json.load(open(os.path.join(tr.ckpt_root, "LATEST.json")))["update"]
    row = tr.meta("test")
    tr.check_provenance(row)                # same code, splits and arguments as the training run
    TP.append_jsonl(os.path.join(a.out, "test_meta.jsonl"),
                    {**row, "final": True, "updates": list(own.test_updates), "seeds": list(own.test_seeds),
                     "reselect_best_sha256": TP.sha_file(os.path.join(a.out, "reselect_best.json")),
                     "reselect_best_update": bu, "task2_ids": test, "task1_ids": test_all, "argv": sys.argv,
                     "eval_code_sha256": {f: TP.sha_file(os.path.join(HERE, f)) for f in ("eval_test_rl.py", "eval_test_boot.py")}})
    p_out = os.path.join(a.out, "test.jsonl")
    for u in own.test_updates:
        s = evaluate(tr, u, own.test_seeds, test, test_all, p_out)
        ts, t1 = s["turn_stats"] or {}, s["task1"] or {}
        print(json.dumps({"test_update": u, "n_episodes": s["n_episodes"], "unclean": s["n_unclean_episodes"],
                          "sim_turns": ts.get("sim_turns_mean"), "human_turns": ts.get("human_turns_mean"),
                          "turn_w1": ts.get("turn_w1"), "coverage": ts.get("coverage_mean"),
                          "term_f1": t1.get("term_f1"), "bal_p": t1.get("bal_p"), "auc": t1.get("auc")}), flush=True)


if __name__ == "__main__":
    main()
