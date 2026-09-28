# -*- coding: utf-8 -*-
"""Task 2 rollout for SFT stop gates, with corrected terminal semantics.

Differences from rollout_stop_grpo.py (which stays untouched for provenance):
  * The gate is a SEPARATE Qwen2.5-7B NF4 model carrying the Stage A/B LoRA
    adapters; the frozen JSON Planner (32B NF4) and UserLM Speaker are unchanged.
  * Gate prompts go through stop_prompt.encode_stop_prompt with max_length 2048,
    byte-identical to train_stop_sft.py / eval_stop_sft.py (same system prompt,
    chat template, TREC goal/state-preserving compaction).
  * Terminal order is task2_episode.run_episode: gate stop and empty draws are not
    emitted turns; a non-empty Speaker end-token utterance is an emitted turn but
    gets no R0 reply and no ledger update.
  * Several arms (adapters and/or a no-gate arm) run in ONE process, interleaved
    per (scenario, seed) in an order drawn from (conversation_id, seed, replicate),
    so R0 drift over wall-clock time does not align with arm identity.
  * Scenario IDs are asserted against the nested manifest for the declared fold
    and side before any model loads.

Writes one JSONL per arm under --out-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

V2FIX = {
    "SEPSIM_ANTILEAK": "1", "SEPSIM_NEARCOPY": "1", "SEPSIM_COPY_SCOPE": "both",
    "SEPSIM_NSAMP": "3", "SEPSIM_SELECTOR": "length", "SEPSIM_LENGTH_SELECT": "1",
    "SEPSIM_POSITION": "system", "SEPSIM_END_PROBE": "0", "SEPSIM_GUARDRAILS": "0",
}
for _k, _v in V2FIX.items():
    os.environ[_k] = _v
os.environ.pop("SEPSIM_REDRAW", None)
os.environ["SEPSIM_ARM"] = "sepsim_v2fix"
os.environ["SEPSIM_ACT_PRIOR"] = "off"
os.environ["SEPSIM_PLANNER_END"] = "0"

BENCH = "/tmp2/hchsu/trec2026-usersim-benchmark"
TREE = "/home/mzjiang/Sep-Simulator"
G = "/tmp2/mzjiang_usersim/grpo_planner"
WORK = "/tmp2/mzjiang_usersim/task2"
CORPUS = os.environ.get("CORPUS", "/home/mzjiang/v5-latency/data.jsonl")
GATE_BASE = ("/tmp2/hf_shared/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
             "a09a35458c702b33eeacc393d103063234e8bc28")
T_MAX = 10
GATE_MAX_LENGTH = 2048

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "scripts"))
sys.path.insert(0, os.path.join(BENCH, "tools"))

from stop_prompt import encode_stop_prompt            # noqa: E402
from task2_episode import run_episode                 # noqa: E402


def log(*a):
    print(*a, flush=True)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_arm(spec):
    """name=nogate  |  name=/path/to/adapter@threshold"""
    name, rest = spec.split("=", 1)
    if rest == "nogate":
        return {"name": name, "adapter": None, "threshold": None}
    path, thr = rest.rsplit("@", 1)
    return {"name": name, "adapter": path, "threshold": float(thr)}


def arm_order(arms, cid, seed, replicate):
    order = list(arms)
    key = hashlib.sha256(("%s|%d|%d" % (cid, seed, replicate)).encode()).hexdigest()
    random.Random(int(key[:16], 16)).shuffle(order)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True, help="t2_scenarios-format JSON (ID-only)")
    ap.add_argument("--fold", type=int, required=True,
                    help="-1 = no-gate union run (no trained component touches any scenario)")
    ap.add_argument("--side", choices=("inner_train", "inner_validation", "inner_union"),
                    required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="episodes per arm (smoke)")
    ap.add_argument("--r0-crn", action="store_true",
                    help="common random numbers: arms of the same (seed, replicate) share R0 "
                         "replies for identical histories (per-seed cache under out-dir)")
    ap.add_argument("--speaker", choices=("userlm", "ditto"), default="userlm",
                    help="frozen Speaker; ditto signals its end with a blank message")
    ap.add_argument("--ditto-path", default="/tmp2/mzjiang_usersim/models/Ditto-8B")
    ap.add_argument("--log-prompts", action="store_true",
                    help="store the exact gate prompt (PP.user_prompt) of every step")
    args = ap.parse_args()

    arms = [parse_arm(s) for s in args.arm]
    if len({a["name"] for a in arms}) != len(arms):
        raise ValueError("duplicate arm names")
    for a in arms:
        if a["adapter"]:
            a["adapter_sha256"] = sha_file(os.path.join(a["adapter"], "adapter_model.safetensors"))

    # ---- leakage gate: scenario IDs must be on the declared nested side --------
    manifest = json.load(open(os.path.join(G, "nested/nested_manifest.json")))
    if args.fold < 0:
        # A no-gate run has no trained component; any scenario that is inner train or
        # inner validation in SOME fold may be generated. Fold-specific adapters are
        # applied later, offline, only to that fold's own inner scenarios.
        if any(a["adapter"] for a in arms) or args.side != "inner_union":
            raise SystemExit("fold -1 is only for no-gate inner_union runs")
        allowed = set()
        for f in manifest["folds"]:
            allowed |= set(f["inner_train_ids"]) | set(f["inner_validation_ids"])
    else:
        if args.side == "inner_union":
            raise SystemExit("inner_union requires --fold -1")
        fold = [f for f in manifest["folds"] if f["fold"] == args.fold][0]
        allowed = set(fold["%s_ids" % args.side])
    sc_file = json.load(open(args.scenarios))
    assert sc_file["t_max"] == T_MAX and sc_file["seeds"] == [0, 1]
    cids = [s["conversation_id"] for s in sc_file["scenarios"]]
    bad = [c for c in cids if c not in allowed]
    if bad:
        raise SystemExit("LEAK GATE: %d scenario(s) not on fold%d %s: %s"
                         % (len(bad), args.fold, args.side, bad[:3]))
    reqs = json.load(open(os.path.join(BENCH, "data/req_shards_v1.json")))
    if any(c not in reqs for c in cids):
        raise SystemExit("scenario without requirement shards")
    log("leak gate OK: %d scenarios on fold%d %s" % (len(cids), args.fold, args.side))

    from sepsim import (agenda as AG, models, persona as P, pipeline,  # noqa: E402
                        planner_prompt as PP, state, stopping)
    import run_v2                                                        # noqa: E402
    from r0_client import R0Client, Ledger                               # noqa: E402
    from metrics.judge import Judge                                      # noqa: E402
    NSAMP, POSITION = run_v2.NSAMP, run_v2.POSITION
    GUARDRAILS, GUARDS_ON = run_v2.GUARDRAILS, run_v2.GUARDS_ON
    COPY_SCOPE, REDRAW = run_v2.COPY_SCOPE, run_v2.REDRAW
    ANTILEAK, NEARCOPY = run_v2.ANTILEAK, run_v2.NEARCOPY
    assert (ANTILEAK, NEARCOPY, COPY_SCOPE, REDRAW, NSAMP, run_v2.SELECTOR,
            run_v2.LENGTH_SELECT, POSITION, GUARDRAILS, run_v2.END_PROBE) == \
           (True, True, "both", 4, 3, "length", True, "system", False, False), \
           "v2fix configuration did not bind"

    recs = [json.loads(l) for l in open(CORPUS, encoding="utf-8") if l.strip()]
    by_cid = {r["conversation_id"]: r for r in recs}
    jobs = [(s, sd) for s in sorted(sc_file["scenarios"], key=lambda x: x["order"])
            for sd in sc_file["seeds"]]
    if args.limit:
        jobs = jobs[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = {a["name"]: os.path.join(args.out_dir, "%s.jsonl" % a["name"]) for a in arms}
    done = {a["name"]: set() for a in arms}
    for a in arms:
        if os.path.exists(out_path[a["name"]]):
            for line in open(out_path[a["name"]], encoding="utf-8"):
                r = json.loads(line)
                done[a["name"]].add((r["conversation_id"], r["seed"]))

    judge = Judge(reasoning_effort="minimal", verbose=False,
                  cache_dir=os.path.join(WORK, "judge_cache"))
    r0 = R0Client()   # cache OFF (faithful setting) unless --r0-crn
    r0_by_seed = {}

    def r0_for(seed):
        if not args.r0_crn:
            return r0
        if seed not in r0_by_seed:
            r0_by_seed[seed] = R0Client(cache_dir=os.path.join(
                args.out_dir, "r0_crn", "rep%d" % args.replicate, "s%d" % seed))
        return r0_by_seed[seed]

    def r0_calls():
        return r0.n_calls + sum(c.n_calls for c in r0_by_seed.values())
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    gpu = args.gpu
    dev = "cuda:%d" % gpu
    planner = models.Planner(gpu=gpu)
    planner._tok = AutoTokenizer.from_pretrained(planner.path)
    planner._model = AutoModelForCausalLM.from_pretrained(planner.path,
        quantization_config=quant, device_map={"": gpu}, low_cpu_mem_usage=True).eval()
    speaker = (models.DittoSpeaker(path=args.ditto_path, gpu=gpu, position=POSITION).load()
               if args.speaker == "ditto" else models.Speaker(gpu=gpu, position=POSITION).load())

    gate_model, gtok = None, None
    gated = [a for a in arms if a["adapter"]]
    if gated:
        gtok = AutoTokenizer.from_pretrained(GATE_BASE)
        gbase = AutoModelForCausalLM.from_pretrained(GATE_BASE, quantization_config=quant,
                                                     device_map={"": gpu}, low_cpu_mem_usage=True)
        for a in gated:
            if gate_model is None:
                gate_model = PeftModel.from_pretrained(gbase, a["adapter"], adapter_name=a["name"])
            else:
                gate_model.load_adapter(a["adapter"], adapter_name=a["name"])
        gate_model.eval()
        yes_no = [gtok.encode(x, add_special_tokens=False) for x in ("NO", "YES")]
        assert all(len(x) == 1 for x in yes_no)
        yes_no = [x[0] for x in yes_no]

    meta = {"runner_sha256": sha_file(os.path.abspath(__file__)),
            "task2_episode_sha256": sha_file(os.path.join(HERE, "task2_episode.py")),
            "stop_prompt_sha256": sha_file(os.path.join(HERE, "stop_prompt.py")),
            "scenarios_sha256": sha_file(args.scenarios), "fold": args.fold, "side": args.side,
            "replicate": args.replicate, "arms": arms, "t_max": T_MAX,
            "gate_base": GATE_BASE, "gate_max_length": GATE_MAX_LENGTH,
            "planner": planner.path, "speaker": speaker.path, "speaker_kind": args.speaker,
            "r0_model": r0.model, "r0_effort": r0.reasoning_effort,
            "judge_model": judge.model, "judge_effort": judge.reasoning_effort,
            "v2fix": V2FIX, "planner_end": False,
            "r0_cache": "per-(replicate,seed) CRN cache shared by arms" if args.r0_crn else "off",
            "log_prompts": args.log_prompts,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(args.out_dir, "run_meta_rep%d.json" % args.replicate), "w") as f:
        json.dump(meta, f, indent=1)
    log("META", json.dumps(meta))

    SYS = PP.system_prompt()
    t0 = time.time()
    n_done = 0

    def one_episode(arm, sc, seed):
        cid, rid = sc["conversation_id"], sc["record_id"]
        rec = by_cid[cid]
        scenario = rec["scenario"]
        sc_text = pipeline.scenario_text(rec)
        sid = "%s#s%d" % (rid, seed)
        rng = random.Random(pipeline.seed_for(sid, 0))
        led = stopping.StoppingLedger(scenario)
        ag = AG.Agenda(AG.terms_of(
            " ".join([str((scenario.get("goal") or {}).get("topic", "")),
                      str((scenario.get("goal") or {}).get("context", ""))]), 8))
        ledger = Ledger(reqs[cid]["req"], judge=judge)
        S = {"hist_u": [], "hist_a": [], "prev_block": state.d0(P.initial_stage(scenario.get("goal"))),
             "up": None, "block": None, "coverage_trace": []}

        def build_up(t):
            S["led_before"] = {"candidates": list(led.candidates()), "frustrated": led.frustrated(),
                               "satisfied": led.satisfied(), "first_met": led.first_met(),
                               "contig_unhelpful": led.contiguous_unhelpful,
                               "best_quality": led.best_quality_seen,
                               "repeated_offer": led.repeated_offer, "n_gain": len(led.gain_trace)}
            S["up"] = PP.user_prompt(scenario, S["prev_block"], S["hist_u"], S["hist_a"], t,
                                     ledger=led, prev_ann={}, agenda_view=ag.render(), p_end=None)
            S["up_t"] = t

        def gate(t):
            build_up(t)
            gate_model.set_adapter(arm["name"])
            ids, info = encode_stop_prompt(gtok, S["up"], GATE_MAX_LENGTH)
            S["gate_info"] = info
            with torch.no_grad():
                logits = gate_model(input_ids=torch.tensor([ids], device=dev),
                                    use_cache=False).logits[0, -1, yes_no].float()
            return round(torch.softmax(logits, dim=-1)[1].item(), 6)

        def speak(t):
            if S.get("up_t") != t:
                build_up(t)
            up = S["up"]
            hist_u, hist_a = S["hist_u"], S["hist_a"]
            raw = planner.raw_with(SYS, up)
            fields, diag = PP.read_plan(raw, t, scenario, rng, led)
            unparsed = fields is None
            if unparsed:
                block, fields = S["prev_block"], {"move": "Other", "act": "other"}
            else:
                block = PP.render_block(fields, ag.render())
            S["block"] = block
            avoid = hist_u if GUARDRAILS else None
            prior = list(hist_u) + (list(hist_a) if COPY_SCOPE == "both" else [])
            if GUARDS_ON:
                avoid = prior
            greedy, ge = speaker.say(sc_text, block, hist_u, hist_a, t,
                                     seed=pipeline.seed_for(sid, t), temperature=0.0,
                                     avoid=avoid, reject_template=ANTILEAK, reject_reuse=NEARCOPY)
            cands, flags = [greedy], [ge]
            for k in range(NSAMP):
                s, e = speaker.say(sc_text, block, hist_u, hist_a, t,
                                   seed=pipeline.seed_for(sid, t, k + 1), temperature=0.7,
                                   top_p=0.9, avoid=avoid, reject_template=ANTILEAK,
                                   reject_reuse=NEARCOPY)
                cands.append(s)
                flags.append(e)
            reasons, n_extra, no_surv, eligible = None, 0, False, None
            if GUARDS_ON:
                reasons = [run_v2.guard_reason(c, prior) for c in cands]
                while all(reasons) and n_extra < REDRAW:
                    k = NSAMP + n_extra
                    sx, ex = speaker.say(sc_text, block, hist_u, hist_a, t,
                                         seed=pipeline.seed_for(sid, t, k + 1), temperature=0.7,
                                         top_p=0.9, avoid=avoid, reject_template=ANTILEAK,
                                         reject_reuse=NEARCOPY)
                    cands.append(sx)
                    flags.append(ex)
                    reasons.append(run_v2.guard_reason(sx, prior))
                    n_extra += 1
                eligible = [i for i, r in enumerate(reasons) if not r] or None
                no_surv = eligible is None
            idx = run_v2.choose(cands, fields.get("length_words"), None, eligible)
            return {"user": cands[idx], "ended_speaker": bool(flags[idx]),
                    "ended_planner": bool(state.ends_session(fields)),
                    "move": fields.get("move", ""), "act": fields.get("act", ""),
                    "stop_rule": fields.get("stop_rule", "none"), "planner_unparsed": unparsed,
                    "guard_reasons": reasons, "guard_extra": n_extra, "no_survivor": no_surv,
                    "selected_index": idx, "n_candidates": len(cands),
                    "candidate_end_flags": [bool(x) for x in flags],
                    "ledger_before": S["led_before"], "gate_encoding": S.get("gate_info"),
                    **({"gate_prompt": up} if args.log_prompts else {})}

        def respond(t, text):
            S["hist_u"].append(text)
            msgs = []
            for i, u in enumerate(S["hist_u"]):
                msgs.append({"role": "user", "content": u})
                if i < len(S["hist_a"]):
                    msgs.append({"role": "assistant", "content": S["hist_a"][i]})
            reply = r0_for(seed).reply(msgs)
            prev_reply = S["hist_a"][-1] if S["hist_a"] else ""
            S["hist_a"].append(reply)
            ledger.update(t, text, reply)
            led.observe({}, reply, prev_reply)
            ag.retire_satisfied(reply, {})
            if AG.looks_like_new_offer(reply, prev_reply):
                ag.reset_on_new_offer()
            S["prev_block"] = S["block"]
            S["coverage_trace"].append((t, (round(ledger.coverage(), 4), bool(ledger.complete()))))
            return reply

        g = gate if arm["adapter"] else None
        ep = run_episode(T_MAX, g, speak, respond,
                         threshold=arm["threshold"] if arm["adapter"] else 0.5)
        # Coverage/complete are latched, so a step without an exchange (gate stop,
        # empty, terminal utterance) carries the value after the last exchange.
        after = dict(S["coverage_trace"])
        last = (0.0, False)
        for step in ep["trace"]:
            last = after.get(step["t"], last)
            step["coverage_after"], step["complete_after"] = last
        return {"conversation_id": cid, "record_id": rid, "seed": seed, "arm": arm["name"],
                "speaker_kind": args.speaker,
                "replicate": args.replicate, "adapter": arm["adapter"],
                "adapter_sha256": arm.get("adapter_sha256"), "threshold": arm["threshold"],
                "fold": args.fold, "side": args.side,
                "emitted_user_turns": ep["emitted_user_turns"],
                "decision_steps": ep["decision_steps"], "end_kind": ep["end_kind"],
                "turns": ep["emitted_user_turns"], "stop_kind": ep["end_kind"],
                "ended_by_token": ep["end_kind"] != "t_max",
                "coverage": round(ledger.coverage(), 4), "complete": ledger.complete(),
                "n_req": len(reqs[cid]["req"]), "ledger": ledger.as_dict(), "trace": ep["trace"]}

    for sc, seed in jobs:
        for rank, arm in enumerate(arm_order(arms, sc["conversation_id"], seed, args.replicate)):
            if (sc["conversation_id"], seed) in done[arm["name"]]:
                continue
            t_ep = time.time()
            row = one_episode(arm, sc, seed)
            row["arm_rank"] = rank
            row["wall_seconds"] = round(time.time() - t_ep, 1)
            with open(out_path[arm["name"]], "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done += 1
            log("  [%d] %s %s s%d emitted=%d steps=%d end=%s cov=%.3f complete=%s (%.0fs r0=%d)"
                % (n_done, arm["name"], sc["conversation_id"][:8], seed, row["emitted_user_turns"],
                   row["decision_steps"], row["end_kind"], row["coverage"], row["complete"],
                   time.time() - t0, r0_calls()))
    stats = {"r0": r0.stats(), "r0_crn": {s: c.stats() for s, c in r0_by_seed.items()},
             "judge_calls": judge.n_calls, "judge_cached": judge.n_cached,
             "speaker_rejects": {"template": speaker.n_reject_template,
                                 "reuse": speaker.n_reject_reuse, "regen": speaker.n_regen},
             "episodes_written": n_done, "seconds": round(time.time() - t0)}
    with open(os.path.join(args.out_dir, "runstats_rep%d.json" % args.replicate), "w") as f:
        json.dump(stats, f, indent=1)
    log("DONE", json.dumps(stats))


if __name__ == "__main__":
    main()
