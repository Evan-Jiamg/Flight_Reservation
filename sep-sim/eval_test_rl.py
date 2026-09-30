"""Test-split evaluation of a finished pend run, SPEC v17 §5 (user 2026-09-30): exactly SFT (u0) and the final update
(final.json: the last completed GRPO update, max_updates or length_drift) on splits[fold].test (Task 2, the ids with
requirement shards) and splits[fold].test_all (Task 1, as the benchmark / task1_v4.py), with exactly the validation
procedure of train_planner_rl.validate(): Task 2 = sampled Planner (--val-temperature, one replicate per seed, an
unclean episode re-run up to VAL_RETRIES times), Task 1 = greedy run + teacher-forced end probabilities
(Trainer.task1_eval_row). Writes RUN/test.jsonl (episode / task1 / summary rows, split "test") and RUN/test_meta.jsonl;
never writes a training file (checkpoints, validation.jsonl, final.json, sft*).
--include-base (v17 B6; fix round 2: EVERY fold, fold 2 included -- the base is re-scored under the v17 definitions):
also the untrained start policy "base" (ckpt/sft_e0, served as sft_e0-<sha>) with the same procedure ->
RUN/test_base.jsonl. The v16 run's u0 test rows (bf16 P_end) are a reference only, never the base.

Gates: --final; the trainer arguments are the run's own (check_provenance: same code, splits, arguments); final.json is
validated and the updates are exactly {0, final}, both with a validation summary that includes Task 2 (S13); every test id
is in forbidden_for_training and in no training / validation / few-shot / p_h list; an update whose test summary exists is
never evaluated again (test once), an unfinished one is resumed.

Usage: python eval_test_rl.py --final --test-seeds 0 1 2 3 4 5 6 7 --test-updates 0 5 [--include-base] <the trainer arguments>
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
    ap.add_argument("--include-base", action="store_true",
                    help="v17 B6: also evaluate the untrained start policy (ckpt/sft_e0) -> test_base.jsonl")
    own, rest = ap.parse_known_args(argv)
    if not own.final:
        ap.error("the test split is read only with --final")
    a = TP.parse_args(rest)
    if a.resume:
        ap.error("pass the trainer arguments without --resume")
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
        return TP.turn_stats_of(eps, t_max)          # human turns capped at t_max (fix round 1, D-N5), as validate()

    n_unclean = sum(1 for r in vrows if not r["episode"]["clean"])
    t1 = T1.task1_stop_metrics([r["task1"] for r in t1rows])
    t1.update(T1.task1_prob_metrics([p for r in t1rows for p in r["end_probs"]]))
    return {"kind": "summary", "update": u, "policy_sha": psha, "split": "test", "final": True,
            "n_episodes": len(vrows) - n_unclean, "n_unclean_episodes": n_unclean,
            "unclean_after_retries": n_unclean > 0, "val_temperature": tr.a.val_temperature, "seeds": list(seeds),
            "task2_ids": sorted({r["conversation_id"] for r in vrows}),
            "task1_ids": sorted(r["conversation_id"] for r in t1rows),
            "turn_stats": turn_stats(vrows),
            # the seeds 0/1 subset alone: comparable with a two-seed evaluation
            "turn_stats_seeds01": turn_stats([r for r in vrows if r["seed"] in (0, 1)]),
            "task1": t1, "test_s": round(time.time() - t0, 1), "time": time.time()}


def evaluate(tr, u, seeds, test, test_all, p_out):
    """u: an update (int) -> ckpt/u<u>; "base" -> the start policy ckpt/sft_e0 (v17 B6)."""
    t0 = time.time()
    prev = TP.read_jsonl(p_out)
    for r in prev:
        if r.get("kind") == "summary" and r["update"] == u:
            print("u%s: test summary exists - not evaluated again" % u, flush=True)
            return r
    d = tr.sft_dir(0) if u == "base" else tr.ckpt_dir(u)
    cst = json.load(open(os.path.join(d, "state.json")))
    tr.learner.load_policy(d)
    psha = tr.learner.policy_sha()
    if psha != cst["policy_sha"]:
        raise AssertionError("checkpoint %s: loaded policy sha differs from its record" % d)
    if u == "base":
        tr.sync_generation_policy(path=os.path.join(d, "adapter"), tag="sft_e0")
    else:
        tr.sync_generation_policy(u)        # vLLM generates with this checkpoint's adapter (name p<u>-<sha>)
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
    fp = os.path.join(a.out, "final.json")
    if not os.path.exists(fp):
        raise SystemExit("%s has no final.json: the run is not finished" % a.out)
    fin = json.load(open(fp, encoding="utf-8"))
    bu = fin.get("final_update")
    if bu is None or not fin.get("validated"):
        raise SystemExit("final.json has no validated final update: %r" % fin)
    # fix round 1 (A-7 / C-N9): final.json names the checkpoint that exists; the spec's test seeds are 0..7
    st_f = json.load(open(os.path.join(a.out, "ckpt", "u%05d" % bu, "state.json"), encoding="utf-8"))
    if st_f["policy_sha"] != fin.get("policy_sha"):
        raise SystemExit("final.json policy sha %s != ckpt u%d's %s" % (str(fin.get("policy_sha"))[:12], bu,
                                                                     st_f["policy_sha"][:12]))
    if sorted(own.test_seeds) != list(range(8)) and not a.ablation:
        raise SystemExit("--test-seeds must be 0..7 (spec v17 §5) unless the run is a named --ablation")
    if sorted(set(own.test_updates)) != sorted({0, bu}) or len(own.test_updates) != len(set(own.test_updates)):
        raise SystemExit("--test-updates must be exactly u0 (SFT) and the final u%d, got %s" % (bu, own.test_updates))
    # S13: both need a validation summary that INCLUDES Task 2 (a Task-1-only summary does not count)
    validated = {r["update"] for r in TP.read_jsonl(os.path.join(a.out, "validation.jsonl"))
                 if r.get("kind") == "summary" and r.get("task2")}
    if set(own.test_updates) - validated:
        raise SystemExit("updates %s have no validation with Task 2" % sorted(set(own.test_updates) - validated))
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
    if own.include_base:
        b0 = json.load(open(os.path.join(tr.sft_dir(0), "state.json")))
        assert b0["epoch"] == 0, "ckpt/sft_e0 is not the start policy"
    TP.append_jsonl(os.path.join(a.out, "test_meta.jsonl"),
                    {**row, "final": True, "updates": list(own.test_updates), "seeds": list(own.test_seeds),
                     "final_json_sha256": TP.sha_file(fp), "final_update": bu, "stop_reason": fin.get("stop_reason"),
                     "include_base": bool(own.include_base), "task2_ids": test, "task1_ids": test_all, "argv": sys.argv,
                     "eval_code_sha256": {f: TP.sha_file(os.path.join(HERE, f)) for f in ("eval_test_rl.py", "eval_test_boot.py")}})
    p_out = os.path.join(a.out, "test.jsonl")
    jobs = [(u, p_out) for u in own.test_updates] + ([("base", os.path.join(a.out, "test_base.jsonl"))]
                                                     if own.include_base else [])
    for u, p in jobs:
        s = evaluate(tr, u, own.test_seeds, test, test_all, p)
        ts, t1 = s["turn_stats"] or {}, s["task1"] or {}
        print(json.dumps({"test_update": u, "n_episodes": s["n_episodes"], "unclean": s["n_unclean_episodes"],
                          "sim_turns": ts.get("sim_turns_mean"), "human_turns": ts.get("human_turns_mean"),
                          "turn_w1": ts.get("turn_w1"), "coverage": ts.get("coverage_mean"),
                          "term_f1": t1.get("term_f1"), "bal_p": t1.get("bal_p"), "auc": t1.get("auc"),
                          "nll": t1.get("nll")}), flush=True)


if __name__ == "__main__":
    main()
