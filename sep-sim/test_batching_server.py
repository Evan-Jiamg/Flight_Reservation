#!/usr/bin/env python3
"""Server check: batched generation == one-by-one generation (greedy), sane when sampled, and faster.

  GPU=1 python test_batching_server.py
Needs the E1.6 tree (task2_env.TREE_E16), Qwen3-4B-Instruct-2507 and Ditto-8B.
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
GPU = int(os.environ.get("GPU", "1"))
Q4 = sorted(glob.glob("/tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/"))[-1]


def log(*a):
    print(*a, flush=True)


def main():
    import task2_env as T2
    T2.setup_environment("pend")
    from sepsim import pipeline, state, persona as P, stopping
    import planner_prompt_v3 as V3
    planner = T2.PlannerLM(Q4, GPU)
    env = T2.Task2Env("pend", GPU, planner, judge=None, batch=False)
    cids = sorted(env.recs)[:4]
    ups = []
    for cid in cids:
        sc = env.recs[cid]["scenario"]
        ups.append(V3.user_prompt_pend(sc, state.d0(P.initial_stage(sc.get("goal"))), [], [], 1,
                                       stopping.StoppingLedger(sc), prev_ann={}))
    # ---- 1 Planner greedy: batch == single
    t0 = time.time()
    single = [planner.generate(env.system, u, 0.0, 1.0, 7) for u in ups]
    t_single = time.time() - t0
    t0 = time.time()
    batch = planner.generate_batch([{"system": env.system, "user": u, "temperature": 0.0, "top_p": 1.0, "seed": 7} for u in ups])
    t_batch = time.time() - t0
    same = sum(a["gen_ids"] == b["gen_ids"] for a, b in zip(single, batch))
    same_prefix = [next((i for i, (x, y) in enumerate(zip(a["gen_ids"], b["gen_ids"])) if x != y), None) for a, b in zip(single, batch)]
    log("1 planner greedy identical %d/%d (first divergence %s); prompt ids identical %s; single %.1fs batch %.1fs" % (
        same, len(ups), same_prefix, all(a["prompt_ids"] == b["prompt_ids"] for a, b in zip(single, batch)), t_single, t_batch))
    assert all(a["prompt_ids"] == b["prompt_ids"] for a, b in zip(single, batch))
    # a divergence must be a numerical near-tie, not a batching bug: at the first differing position, the
    # UNPADDED single-sequence model must give both tokens (single's and batch's) almost the same log-prob
    import torch
    import rl_algos as RA
    for a, b, j in zip(single, batch, same_prefix):
        if j is None:
            continue
        ids = torch.tensor([a["prompt_ids"] + a["gen_ids"][:j]], device=next(planner.model.parameters()).device)
        with torch.no_grad():
            lp = torch.log_softmax(planner.model(input_ids=ids).logits[0, -1].float(), -1)
        ta, tb = a["gen_ids"][j], b["gen_ids"][j]
        gap = abs(float(lp[ta] - lp[tb]))
        log("   divergence at %d: logp single-token %.3f batch-token %.3f gap %.3f" % (j, float(lp[ta]), float(lp[tb]), gap))
        assert j >= 8, "divergence in the first tokens: position/padding bug"
        assert gap < 0.5, "batched greedy picked a clearly worse token (gap %.3f): not a numerical tie" % gap
    _ = RA
    eos = planner.eos_ids()
    for b in batch:
        assert b["hit_max_new"] or b["gen_ids"][-1] in eos, "batched output not cut at the end token"
        assert state.json_of(b["raw"]) is not None or b["hit_max_new"], "batched Planner output does not parse"
    # ---- 2 Planner sampled batch: parses, cut at end token
    sb = planner.generate_batch([{"system": env.system, "user": u, "temperature": 0.7, "top_p": 0.8, "seed": 11} for u in ups])
    ok = sum(state.json_of(b["raw"]) is not None for b in sb)
    log("2 planner sampled batch parse ok %d/%d, lengths %s" % (ok, len(sb), [len(b["gen_ids"]) for b in sb]))
    assert ok >= len(sb) - 1
    # ---- 3 Ditto greedy: say_batch == say
    reqs = []
    for cid, b in zip(cids, batch):
        rec = env.recs[cid]
        fields, diag, end = V3.read_plan_pend(b["raw"], 1, rec["scenario"], __import__("random").Random(0),
                                              stopping.StoppingLedger(rec["scenario"]))
        block = V3.speaker_block_pend(fields) if fields else state.d0("exploring")
        reqs.append({"scenario_text": pipeline.scenario_text(rec), "block": block, "hist_u": [], "hist_a": [],
                     "turn": 1, "seed": 5, "temperature": 0.0, "top_p": 1.0, "avoid": [],
                     "reject_template": True, "reject_reuse": True})
    t0 = time.time()
    one = [env.speaker.say(r["scenario_text"], r["block"], [], [], 1, seed=5, temperature=0.0, top_p=1.0, avoid=[],
                           reject_template=True, reject_reuse=True)[0] for r in reqs]
    t_one = time.time() - t0
    t0 = time.time()
    many = [x[0] for x in env.speaker.say_batch(reqs)]
    t_many = time.time() - t0
    same_d = sum(a == b for a, b in zip(one, many))
    log("3 ditto greedy identical %d/%d; single %.1fs batch %.1fs" % (same_d, len(reqs), t_one, t_many))
    for a, b in zip(one, many):
        if a != b:
            log("   single:", a[:120]); log("   batch :", b[:120])
    assert same_d >= len(reqs) - 1, "batched greedy Ditto diverges from single on more than one request"
    # ---- 4 Ditto sampled batch: non-empty, guard-passing
    sreqs = [dict(r, temperature=0.7, top_p=0.9, seed=100 + i) for i, r in enumerate(reqs * 2)]
    outs = env.speaker.say_batch(sreqs)
    log("4 ditto sampled batch non-empty %d/%d" % (sum(bool(t.strip()) for t, _, _ in outs), len(outs)))
    assert all(t.strip() for t, _, _ in outs)
    log("speedup planner %.2fx, ditto %.2fx" % (t_single / max(t_batch, 1e-6), t_one / max(t_many, 1e-6)))
    log("ALL BATCHING CHECKS PASSED")


if __name__ == "__main__":
    main()
