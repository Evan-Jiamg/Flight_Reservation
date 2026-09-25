# -*- coding: utf-8 -*-
"""Task 2 rollout, Ditto Speaker, after the architecture audit (AUDIT_AND_PLAN_v2).

Arms
  a0  no-gate reference: ORIGINAL Planner prompt and Speaker block (ledger + lexical agenda),
      Planner end acts are recorded but do not end the episode (A1 is derived offline).
  a2  Planner prompt v3 + goal-satisfaction judge + silent Planner exit.
Both arms: D5 truncation fix (Planner and Speaker), Ditto under the consistent-test stack,
frozen JSON Planner (Qwen2.5-32B NF4), v2fix guards, R0 gpt-5-mini (cache off), T_max 10.

Stop decision (D1, stated explicitly): the Planner's final act after sepsim read_plan is a
session-ending act (state.ends_session). With SEPSIM_ACT_PRIOR=off, read_plan overwrites the
sampled act with Complete/settle|abandon whenever the Planner names a non-weak stop_rule; the
sampler may also choose a Complete act itself. Both paths are logged per step (diag).

Leak gates
  a0: every scenario is inner train or inner validation of SOME outer fold (no trained part).
  a2: scenarios are exactly crossfit group g of outer fold f; the judge adapter's
      train_manifest.json lists its training scenarios, which must be disjoint from group g
      and inside fold f's cross-fit set.
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
os.environ["SEPSIM_ACT_PRIOR"] = "off"      # stop override ON (D1), as in every earlier run
os.environ["SEPSIM_PLANNER_END"] = "0"

BENCH = "/tmp2/hchsu/trec2026-usersim-benchmark"
TREE = "/home/mzjiang/Sep-Simulator"
G = "/tmp2/mzjiang_usersim/grpo_planner"
WORK = "/tmp2/mzjiang_usersim/task2"
CORPUS = "/home/mzjiang/v5-latency/data.jsonl"
DITTO = "/tmp2/mzjiang_usersim/models/Ditto-8B"
JUDGE_BASE = ("/tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/")
T_MAX = 10

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "scripts"))
sys.path.insert(0, os.path.join(BENCH, "tools"))
sys.modules.setdefault("torchvision", None)
sys.modules.setdefault("torchaudio", None)

from task2_episode import run_episode   # noqa: E402
import fit_prompts as F                 # noqa: E402


def log(*a):
    print(*a, flush=True)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve_judge_base(path):
    if os.path.isdir(os.path.join(path, "")) and not os.path.exists(os.path.join(path, "config.json")):
        snaps = sorted(os.listdir(path))
        if len(snaps) != 1:
            raise SystemExit("ambiguous judge base snapshots: %s" % snaps)
        return os.path.join(path, snaps[0])
    return path


def leak_gate(args):
    nested = json.load(open(os.path.join(G, "nested/nested_manifest.json")))["folds"]
    union_inner = set()
    for f in nested:
        union_inner |= set(f["inner_train_ids"]) | set(f["inner_validation_ids"])
    if args.arm == "a0":
        sc = json.load(open(args.scenarios))["scenarios"]
        cids = [s["conversation_id"] for s in sc]
        bad = [c for c in cids if c not in union_inner]
        if bad:
            raise SystemExit("LEAK GATE a0: not inner in any fold: %s" % bad[:3])
        return sc, {"arm": "a0", "scenarios": len(cids)}
    cf = json.load(open(args.crossfit_manifest))
    fold = [f for f in cf["folds"] if f["fold"] == args.fold][0]
    group = fold["groups"][args.group]
    tm = json.load(open(os.path.join(args.judge_adapter, "train_manifest.json")))
    train = set(tm["train_scenarios"])
    if tm.get("fold") != args.fold or tm.get("held_out_group") != args.group:
        raise SystemExit("LEAK GATE a2: judge adapter was trained for fold %s group %s"
                         % (tm.get("fold"), tm.get("held_out_group")))
    if train & set(group):
        raise SystemExit("LEAK GATE a2: judge trained on evaluation scenarios %s" % sorted(train & set(group))[:3])
    if not train <= set(fold["scenarios"]):
        raise SystemExit("LEAK GATE a2: judge trained outside fold %d cross-fit set" % args.fold)
    corpus = {json.loads(l)["conversation_id"]: json.loads(l)["record_id"] for l in open(CORPUS) if l.strip()}
    sc = [{"order": i, "conversation_id": c, "record_id": corpus[c]} for i, c in enumerate(group)]
    return sc, {"arm": "a2", "fold": args.fold, "group": args.group, "scenarios": len(group),
                "judge_train_scenarios": len(train), "judge_train_manifest_sha256":
                sha_file(os.path.join(args.judge_adapter, "train_manifest.json"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("a0", "a2"), required=True)
    ap.add_argument("--scenarios", help="a0: scenario list (ID-only)")
    ap.add_argument("--crossfit-manifest", default=os.path.join(G, "crossfit_manifest_v1.json"))
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--group", type=int, default=-1)
    ap.add_argument("--judge-adapter", default="")
    ap.add_argument("--judge-base", default=JUDGE_BASE)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="episodes (smoke)")
    args = ap.parse_args()
    if args.arm == "a2" and not (args.judge_adapter and args.fold >= 0 and args.group >= 0):
        raise SystemExit("a2 needs --judge-adapter, --fold, --group")
    if args.arm == "a0" and not args.scenarios:
        raise SystemExit("a0 needs --scenarios")
    scen, gate_info = leak_gate(args)
    log("leak gate OK", json.dumps(gate_info))
    # a0 keeps the original Planner (stop override ON); a2 turns it OFF and decides by end_session
    os.environ["SEPSIM_ACT_PRIOR"] = "nostopclobber" if args.arm == "a2" else "off"

    from sepsim import (agenda as AG, models, persona as P, pipeline,  # noqa: E402
                        planner_prompt as PP, state, stopping)
    import run_v2                                                        # noqa: E402
    from r0_client import R0Client, Ledger                               # noqa: E402
    from metrics.judge import Judge                                      # noqa: E402
    import planner_prompt_v3 as V3                                       # noqa: E402
    import goal_judge as GJ                                              # noqa: E402
    assert (run_v2.ANTILEAK, run_v2.NEARCOPY, run_v2.COPY_SCOPE, run_v2.REDRAW, run_v2.NSAMP,
            run_v2.SELECTOR, run_v2.LENGTH_SELECT, run_v2.POSITION, run_v2.GUARDRAILS, run_v2.END_PROBE) == \
           (True, True, "both", 4, 3, "length", True, "system", False, False), "v2fix did not bind"
    from sepsim import acts
    if args.arm == "a0":
        assert acts.prior_mode() == frozenset(), "a0 must run the original Planner (override ON)"
    else:
        assert acts.prior_mode() == frozenset({"nostopclobber"}), "a2 must run with the override OFF"

    recs = {}
    for l in open(CORPUS, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            recs[r["conversation_id"]] = r
    reqs = json.load(open(os.path.join(BENCH, "data/req_shards_v1.json")))
    jobs = [(s, sd) for s in scen for sd in (0, 1)]
    if args.limit:
        jobs = jobs[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "%s.jsonl" % args.arm)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["conversation_id"], r["seed"]))

    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    gpu = args.gpu
    ledger_judge = Judge(reasoning_effort="minimal", verbose=False, cache_dir=os.path.join(WORK, "judge_cache"))
    r0 = R0Client()
    planner = models.Planner(gpu=gpu)
    planner_raw_tok = AutoTokenizer.from_pretrained(planner.path)
    planner._tok = F.TokProxy(planner_raw_tok, F.PLANNER_BUDGET)
    planner._model = AutoModelForCausalLM.from_pretrained(planner.path, quantization_config=quant,
                                                          device_map={"": gpu}, low_cpu_mem_usage=True).eval()
    FitDitto = F.make_fit_ditto_speaker(models.DittoSpeaker)
    speaker = FitDitto(path=DITTO, gpu=gpu, position=run_v2.POSITION).load()
    judge = None
    if args.arm == "a2":
        judge = GJ.GoalJudge(resolve_judge_base(args.judge_base), adapter=args.judge_adapter, gpu=gpu).load()

    code = {n: sha_file(os.path.join(HERE, n)) for n in
            ("rollout_ditto_v3.py", "task2_episode.py", "fit_prompts.py", "planner_prompt_v3.py", "goal_judge.py")}
    meta = {"code_sha256": code, "gate": gate_info, "arm": args.arm, "replicate": args.replicate,
            "t_max": T_MAX, "planner": planner.path, "speaker": DITTO,
            "judge_base": resolve_judge_base(args.judge_base) if judge else None,
            "judge_adapter": args.judge_adapter or None,
            "judge_adapter_sha256": sha_file(os.path.join(args.judge_adapter, "adapter_model.safetensors"))
            if judge else None,
            "r0_model": r0.model, "r0_effort": r0.reasoning_effort, "r0_cache": "off",
            "ledger_judge_model": ledger_judge.model, "v2fix": V2FIX,
            "act_prior": ("nostopclobber: override off; stop = Planner end_session; length unclamped"
                          if args.arm == "a2" else "off: original read_plan (override on, length clamp)"),
            "planner_exit": "silent" if args.arm == "a2" else "not executed (logged only)",
            "planner_budget": F.PLANNER_BUDGET, "speaker_budget": F.SPEAKER_BUDGET,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "peft": peft.__version__},
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    SYS = V3.system_prompt_v3() if args.arm == "a2" else PP.system_prompt()
    meta["system_prompt_sha256"] = hashlib.sha256(SYS.encode()).hexdigest()
    meta["goal_judge_system_sha256"] = hashlib.sha256(GJ.SYSTEM.encode()).hexdigest() if judge else None
    with open(os.path.join(args.out_dir, "run_meta_%s_rep%d.json" % (args.arm, args.replicate)), "w") as f:
        json.dump(meta, f, indent=1)
    log("META", json.dumps(meta))
    t0, n_done = time.time(), 0

    def one_episode(sc, seed):
        cid, rid = sc["conversation_id"], sc["record_id"]
        rec = recs[cid]
        scenario = rec["scenario"]
        sc_text = pipeline.scenario_text(rec)
        sid = "%s#s%d" % (rid, seed)
        rng = random.Random(pipeline.seed_for(sid, 0))
        led = stopping.StoppingLedger(scenario)
        ag = AG.Agenda(AG.terms_of(" ".join([str((scenario.get("goal") or {}).get("topic", "")),
                                             str((scenario.get("goal") or {}).get("context", ""))]), 8))
        ledger = Ledger(reqs[cid]["req"], judge=ledger_judge)
        S = {"hist_u": [], "hist_a": [], "prev_block": state.d0(P.initial_stage(scenario.get("goal"))),
             "block": None, "cov": []}

        def speak(t):
            hist_u, hist_a = S["hist_u"], S["hist_a"]
            gs = None
            if args.arm == "a2":
                gs = judge.assess(sc_text, hist_u, hist_a) if hist_a else {"status": "NOT ASSESSED", "unmet": []}
                up = V3.user_prompt_v3(scenario, S["prev_block"], hist_u, hist_a, t, led,
                                       {"status": gs["status"], "unmet": gs.get("unmet", [])}, prev_ann={})
            else:
                up = PP.user_prompt(scenario, S["prev_block"], hist_u, hist_a, t, ledger=led,
                                    prev_ann={}, agenda_view=ag.render(), p_end=None)
            up_fit, pfit = F.fit_planner_user(planner_raw_tok, SYS, up)
            raw = planner.raw_with(SYS, up_fit)
            if args.arm == "a2":
                fields, diag, end_session = V3.read_plan_v3(raw, t, scenario, rng, led)
            else:
                fields, diag = PP.read_plan(raw, t, scenario, rng, led)
                end_session = None
            unparsed = fields is None
            if unparsed:
                fields = {"move": "Other", "act": "other"}
            # a2: the Planner's own end_session decision; a0: the original act-based signal (logged only)
            ended = bool(end_session) if args.arm == "a2" else bool(state.ends_session(fields))
            base = {"planner_prompt": up_fit, "planner_fit": pfit, "planner_raw": raw,
                    "planner_diag": diag, "planner_unparsed": unparsed, "ended_planner": ended,
                    "move": fields.get("move", ""), "act": fields.get("act", ""),
                    "stop_rule": fields.get("stop_rule", "none"), "goal_status": gs,
                    "ledger_before": {"turns": led.turns, "gain_trace": list(led.gain_trace)}}
            if args.arm == "a2" and ended:
                return {**base, "planner_stop": True, "user": ""}
            if unparsed:
                block = S["prev_block"]
            elif args.arm == "a2":
                block = V3.speaker_block_v3(fields, gs)
            else:
                block = PP.render_block(fields, ag.render())
            S["block"] = block
            # ---- generation block: line for line the same as rollout_stop_sft.py / run_v2 ----
            ANTILEAK, NEARCOPY = run_v2.ANTILEAK, run_v2.NEARCOPY
            avoid = hist_u if run_v2.GUARDRAILS else None
            prior = list(hist_u) + (list(hist_a) if run_v2.COPY_SCOPE == "both" else [])
            if run_v2.GUARDS_ON:
                avoid = prior
            greedy, ge = speaker.say(sc_text, block, hist_u, hist_a, t, seed=pipeline.seed_for(sid, t),
                                     temperature=0.0, avoid=avoid, reject_template=ANTILEAK,
                                     reject_reuse=NEARCOPY)
            fits = [speaker.last_fit]
            cands, flags = [greedy], [ge]
            for k in range(run_v2.NSAMP):
                s_, e_ = speaker.say(sc_text, block, hist_u, hist_a, t, seed=pipeline.seed_for(sid, t, k + 1),
                                     temperature=0.7, top_p=0.9, avoid=avoid, reject_template=ANTILEAK,
                                     reject_reuse=NEARCOPY)
                cands.append(s_)
                flags.append(e_)
            reasons, n_extra, eligible = None, 0, None
            if run_v2.GUARDS_ON:
                reasons = [run_v2.guard_reason(c, prior) for c in cands]
                while all(reasons) and n_extra < run_v2.REDRAW:
                    k = run_v2.NSAMP + n_extra
                    sx, ex = speaker.say(sc_text, block, hist_u, hist_a, t,
                                         seed=pipeline.seed_for(sid, t, k + 1), temperature=0.7, top_p=0.9,
                                         avoid=avoid, reject_template=ANTILEAK, reject_reuse=NEARCOPY)
                    cands.append(sx)
                    flags.append(ex)
                    reasons.append(run_v2.guard_reason(sx, prior))
                    n_extra += 1
                eligible = [i for i, r in enumerate(reasons) if not r] or None
            idx = run_v2.choose(cands, fields.get("length_words"), None, eligible)
            return {**base, "user": cands[idx], "ended_speaker": bool(flags[idx]), "block": block,
                    "guard_reasons": reasons, "guard_extra": n_extra,
                    "no_survivor": reasons is not None and eligible is None,
                    "selected_index": idx, "n_candidates": len(cands), "speaker_fit": fits[0]}

        def respond(t, text):
            S["hist_u"].append(text)
            msgs = []
            for i, u in enumerate(S["hist_u"]):
                msgs.append({"role": "user", "content": u})
                if i < len(S["hist_a"]):
                    msgs.append({"role": "assistant", "content": S["hist_a"][i]})
            reply = r0.reply(msgs)
            prev_reply = S["hist_a"][-1] if S["hist_a"] else ""
            S["hist_a"].append(reply)
            ledger.update(t, text, reply)
            led.observe({}, reply, prev_reply)
            ag.retire_satisfied(reply, {})
            if AG.looks_like_new_offer(reply, prev_reply):
                ag.reset_on_new_offer()
            S["prev_block"] = S["block"]
            S["cov"].append((t, (round(ledger.coverage(), 4), bool(ledger.complete()))))
            return reply

        ep = run_episode(T_MAX, None, speak, respond)
        after, last = dict(S["cov"]), (0.0, False)
        for step in ep["trace"]:
            last = after.get(step["t"], last)
            step["coverage_after"], step["complete_after"] = last
        return {"conversation_id": cid, "record_id": rid, "seed": seed, "arm": args.arm,
                "replicate": args.replicate, "fold": args.fold, "group": args.group, "speaker_kind": "ditto",
                "emitted_user_turns": ep["emitted_user_turns"], "decision_steps": ep["decision_steps"],
                "end_kind": ep["end_kind"], "turns": ep["emitted_user_turns"], "stop_kind": ep["end_kind"],
                "ended_by_token": ep["end_kind"] != "t_max",
                "coverage": round(ledger.coverage(), 4), "complete": ledger.complete(),
                "n_req": len(reqs[cid]["req"]), "ledger": ledger.as_dict(), "trace": ep["trace"]}

    for sc, seed in jobs:
        if (sc["conversation_id"], seed) in done:
            continue
        t_ep = time.time()
        row = one_episode(sc, seed)
        row["wall_seconds"] = round(time.time() - t_ep, 1)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_done += 1
        log("  [%d] %s %s s%d emitted=%d steps=%d end=%s cov=%.3f complete=%s (%.0fs)"
            % (n_done, args.arm, sc["conversation_id"][:8], seed, row["emitted_user_turns"],
               row["decision_steps"], row["end_kind"], row["coverage"], row["complete"], time.time() - t0))
    stats = {"r0": r0.stats(), "ledger_judge_calls": ledger_judge.n_calls,
             "goal_judge_calls": judge.n_calls if judge else 0,
             "goal_judge_unparsed": judge.n_unparsed if judge else 0,
             "speaker_rejects": {"template": speaker.n_reject_template, "reuse": speaker.n_reject_reuse,
                                 "regen": speaker.n_regen},
             "planner_legacy_tok_calls": planner._tok.n_legacy_calls,
             "episodes_written": n_done, "seconds": round(time.time() - t0)}
    with open(os.path.join(args.out_dir, "runstats_%s_rep%d.json" % (args.arm, args.replicate)), "w") as f:
        json.dump(stats, f, indent=1)
    log("DONE", json.dumps(stats))


if __name__ == "__main__":
    main()
