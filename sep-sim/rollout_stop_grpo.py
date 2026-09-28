# -*- coding: utf-8 -*-
"""Task 2 free-running rollout for the sepsim_v2fix ARM CONFIGURATION.

Protocol: /tmp2/hchsu/trec2026-usersim-benchmark/protocols/t2_rollout.md
Environment contract: tools/r0_client.py in that same benchmark repo.

There is no run_task2.py.  The benchmark ships the PROTOCOL and the ENVIRONMENT
(R0 + Ledger) but explicitly cannot ship a harness: "F10 is the one family that
needs YOUR model in the loop".  So the driver is ours; the environment is not.

The v2fix guards (ANTILEAK / NEARCOPY / COPY_SCOPE / REDRAW / LENGTH_SELECT) are
read ONLY in scripts/run_v2.py, never in sepsim/.  As in probe_k1_v2fix.py, the
guard logic is not re-typed here: we CALL run_v2.guard_reason and run_v2.choose.
The v2fix environment is exported before importing run_v2 because run_v2 binds
its flags at module import time.

pipeline.prior_annotations is "Present in Task 1, empty in Task 2".  R0 replies
carry no annotations, so ann = {} everywhere below.  Nothing is fabricated to
fill it; the per-turn ledger state is recorded so the loss can be quantified.

Writes only under /tmp2/mzjiang_usersim/task2/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

# ---- v2fix configuration, verbatim, BEFORE importing run_v2 -----------------
V2FIX = {
    "SEPSIM_ANTILEAK": "1",
    "SEPSIM_NEARCOPY": "1",
    "SEPSIM_COPY_SCOPE": "both",
    "SEPSIM_NSAMP": "3",
    "SEPSIM_SELECTOR": "length",
    "SEPSIM_LENGTH_SELECT": "1",
    "SEPSIM_POSITION": "system",
    "SEPSIM_END_PROBE": "0",
    "SEPSIM_GUARDRAILS": "0",
    # SEPSIM_REDRAW deliberately unset -> run_v2 default 4 (per-turn cap)
}
for _k, _v in V2FIX.items():
    os.environ[_k] = _v
os.environ.pop("SEPSIM_REDRAW", None)
os.environ["SEPSIM_ARM"] = "sepsim_v2fix"
os.environ["SEPSIM_ACT_PRIOR"] = "off"
os.environ["SEPSIM_PLANNER_END"] = "0"

BENCH = "/tmp2/hchsu/trec2026-usersim-benchmark"
TREE = "/home/mzjiang/Sep-Simulator"
WORK = "/tmp2/mzjiang_usersim/task2"
CORPUS = os.environ.get("CORPUS", "/home/mzjiang/v5-latency/data.jsonl")
ARM = "sepsim_v2fix"
T_MAX = 10
#: Credit the planner's own stop decision, matching run_v2.py's SEPSIM_PLANNER_END.
#: Default OFF so the published 64-episode run reproduces byte for byte.
PLANNER_END = os.environ.get("SEPSIM_PLANNER_END", "0") == "1"

sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "scripts"))
sys.path.insert(0, os.path.join(BENCH, "tools"))

from sepsim import (agenda as AG, models, persona as P, pipeline,  # noqa: E402
                    planner_prompt as PP, state, stopping)
import run_v2                          # noqa: E402  (binds the v2fix flags above)
from r0_client import R0Client, Ledger  # noqa: E402
from train_stop_head_grpo import DECISION_SYSTEM  # noqa: E402
from metrics.judge import Judge         # noqa: E402

NSAMP, POSITION = run_v2.NSAMP, run_v2.POSITION
GUARDRAILS, GUARDS_ON = run_v2.GUARDRAILS, run_v2.GUARDS_ON
COPY_SCOPE, REDRAW = run_v2.COPY_SCOPE, run_v2.REDRAW
ANTILEAK, NEARCOPY = run_v2.ANTILEAK, run_v2.NEARCOPY


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="episodes (smoke); 0 = all 64")
    ap.add_argument("--out", default=os.path.join(WORK, "episodes_v2fix.jsonl"))
    ap.add_argument("--tag", default="grpo")
    ap.add_argument("--adapter", default="", help="PEFT adapter; omit for paired base")
    ap.add_argument("--speaker", choices=("userlm", "ditto"), default="userlm")
    args = ap.parse_args()

    log("CONFIG antileak=%s nearcopy=%s scope=%s redraw=%d nsamp=%d selector=%s "
        "length_select=%s position=%s guardrails=%s end_probe=%s"
        % (ANTILEAK, NEARCOPY, COPY_SCOPE, REDRAW, NSAMP, run_v2.SELECTOR,
           run_v2.LENGTH_SELECT, POSITION, GUARDRAILS, run_v2.END_PROBE))
    assert (ANTILEAK, NEARCOPY, COPY_SCOPE, REDRAW, NSAMP, run_v2.SELECTOR,
            run_v2.LENGTH_SELECT, POSITION, GUARDRAILS, run_v2.END_PROBE) == \
           (True, True, "both", 4, 3, "length", True, "system", False, False), \
           "v2fix configuration did not bind"

    sc_file = json.load(open(os.path.join(BENCH, "data/t2_scenarios_v1.json")))
    reqs = json.load(open(os.path.join(BENCH, "data/req_shards_v1.json")))
    assert sc_file["t_max"] == T_MAX and sc_file["seeds"] == [0, 1]
    recs = [json.loads(l) for l in open(CORPUS, encoding="utf-8") if l.strip()]
    by_cid = {r["conversation_id"]: r for r in recs}

    # 64 episodes = 32 scenarios x seeds [0,1], in the frozen file order.
    jobs = [(s, sd) for s in sorted(sc_file["scenarios"], key=lambda x: x["order"])
            for sd in sc_file["seeds"]]
    if args.limit:
        jobs = jobs[: args.limit]
    log("episodes queued: %d" % len(jobs))

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["conversation_id"], r["seed"]))
            except Exception:
                pass
    log("already present: %d" % len(done))

    judge = Judge(reasoning_effort="minimal", verbose=False,
                  cache_dir=os.path.join(WORK, "judge_cache"))
    r0 = R0Client()   # cache OFF: the protocol calls that the faithful setting
    log("R0: model=%s effort=%s | Judge: model=%s effort=%s"
        % (r0.model, r0.reasoning_effort, judge.model, judge.reasoning_effort))

    pgpu = 0
    log("planner on cuda:%d, quantized; adapter=%s" % (pgpu, args.adapter or "base"))
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    planner = models.Planner(gpu=pgpu)
    planner._tok = AutoTokenizer.from_pretrained(planner.path)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(planner.path,
        quantization_config=quant, device_map={"": pgpu}, low_cpu_mem_usage=True)
    planner._model = (PeftModel.from_pretrained(base, args.adapter).eval()
                      if args.adapter else base.eval())
    sgpu = pgpu
    log("speaker on cuda:%d (position=%s)" % (sgpu, POSITION))
    speaker = (models.DittoSpeaker(path="/tmp2/mzjiang_usersim/models/Ditto-8B", gpu=sgpu, position=POSITION).load()
               if args.speaker == "ditto" else models.Speaker(gpu=sgpu, position=POSITION).load())

    SYS = PP.system_prompt()
    t0 = time.time()
    n_done = 0
    G = {"blocked": 0, "extra": 0, "nosurv": 0, "cands": 0,
         "empty": 0, "template": 0, "reuse": 0, "turns": 0, "unparsed": 0}

    for sc, seed in jobs:
        cid, rid = sc["conversation_id"], sc["record_id"]
        if (cid, seed) in done:
            continue
        rec = by_cid[cid]
        scenario = rec["scenario"]
        sc_text = pipeline.scenario_text(rec)
        req_items = reqs[cid]["req"]
        # independent seed stream per (scenario, seed); the published run seeded a
        # different policy stack (seed*1000+turn) that cannot be transplanted.
        sid = "%s#s%d" % (rid, seed)
        rng = random.Random(pipeline.seed_for(sid, 0))

        led = stopping.StoppingLedger(scenario)
        ag = AG.Agenda(AG.terms_of(
            " ".join([str((scenario.get("goal") or {}).get("topic", "")),
                      str((scenario.get("goal") or {}).get("context", ""))]), 8))
        ledger = Ledger(req_items, judge=judge)

        hist_u, hist_a = [], []
        prev_block = state.d0(P.initial_stage(scenario.get("goal")))
        turns, ended_by_token, stop_kind = 0, False, "t_max"
        trace = []

        for t in range(1, T_MAX + 1):
            turns = t
            # --- Task 2: no platform annotations exist. Nothing is invented. ---
            prev_ann = {}
            led_before = {"candidates": list(led.candidates()),
                          "frustrated": led.frustrated(), "satisfied": led.satisfied(),
                          "first_met": led.first_met(),
                          "contig_unhelpful": led.contiguous_unhelpful,
                          "best_quality": led.best_quality_seen,
                          "repeated_offer": led.repeated_offer,
                          "n_gain": len(led.gain_trace)}

            up = PP.user_prompt(scenario, prev_block, hist_u, hist_a, t,
                                ledger=led, prev_ann=prev_ann,
                                agenda_view=ag.render(), p_end=None)
            # The separately trained Planner gate acts before any user utterance.
            # NO continues into the original frozen JSON Planner and Speaker.
            gate_prompt = planner._tok.apply_chat_template([
                {"role": "system", "content": DECISION_SYSTEM},
                {"role": "user", "content": up}],
                tokenize=False, add_generation_prompt=True)
            gate_ids = planner._tok(gate_prompt, return_tensors="pt",
                                    truncation=True, max_length=3072).to("cuda:%d" % pgpu)
            no_yes = [planner._tok.encode(x, add_special_tokens=False)[0]
                      for x in ("NO", "YES")]
            with torch.no_grad():
                gate_logits = planner._model(**gate_ids, use_cache=False).logits[
                    0, -1, no_yes].float()
            p_stop = torch.softmax(gate_logits, dim=-1)[1].item()
            if p_stop >= 0.5:
                ended_by_token, stop_kind = True, "stop_gate"
                trace.append({"t": t, "stop_gate": True, "p_stop": round(p_stop, 6),
                              "ended_speaker": False, "ended_planner": False,
                              "empty": True, "coverage_after": round(ledger.coverage(), 4),
                              "user": "", "agent": None})
                break
            if args.adapter:
                with planner._model.disable_adapter():
                    raw = planner.raw_with(SYS, up)
            else:
                raw = planner.raw_with(SYS, up)
            fields, diag = PP.read_plan(raw, t, scenario, rng, led)
            if fields is None:
                G["unparsed"] += 1
                block, fields = prev_block, {"move": "Other", "act": "other"}
            else:
                block = PP.render_block(fields, ag.render())

            # ---- generation block of run_v2.py, same order, same seeds -------
            avoid = hist_u if GUARDRAILS else None
            prior = list(hist_u) + (list(hist_a) if COPY_SCOPE == "both" else [])
            if GUARDS_ON:
                avoid = prior
            greedy, ge = speaker.say(sc_text, block, hist_u, hist_a, t,
                                     seed=pipeline.seed_for(sid, t), temperature=0.0,
                                     avoid=avoid, reject_template=ANTILEAK,
                                     reject_reuse=NEARCOPY)
            samples, ends = [], []
            for k in range(NSAMP):
                s, e = speaker.say(sc_text, block, hist_u, hist_a, t,
                                   seed=pipeline.seed_for(sid, t, k + 1),
                                   temperature=0.7, top_p=0.9, avoid=avoid,
                                   reject_template=ANTILEAK, reject_reuse=NEARCOPY)
                samples.append(s)
                ends.append(e)

            cands, flags = [greedy] + samples, [ge] + ends
            reasons, n_extra, no_surv = None, 0, False
            if GUARDS_ON:
                reasons = [run_v2.guard_reason(c, prior) for c in cands]
                while all(reasons) and n_extra < REDRAW:
                    k = NSAMP + n_extra
                    sx, ex = speaker.say(sc_text, block, hist_u, hist_a, t,
                                         seed=pipeline.seed_for(sid, t, k + 1),
                                         temperature=0.7, top_p=0.9, avoid=avoid,
                                         reject_template=ANTILEAK, reject_reuse=NEARCOPY)
                    cands.append(sx)
                    flags.append(ex)
                    reasons.append(run_v2.guard_reason(sx, prior))
                    n_extra += 1
                    G["extra"] += 1
            target = fields.get("length_words")
            eligible = None
            if GUARDS_ON:
                eligible = [i for i, r in enumerate(reasons) if not r]
                G["cands"] += len(reasons)
                for r in reasons:
                    if r:
                        G["blocked"] += 1
                        G[r] = G.get(r, 0) + 1
                if not eligible:
                    no_surv = True
                    G["nosurv"] += 1
                    eligible = None
            idx = run_v2.choose(cands, target, None, eligible)
            text, ended_speaker = cands[idx], flags[idx]
            ended_planner = bool(state.ends_session(fields))
            G["turns"] += 1

            hist_u.append(text)
            empty = not (text or "").strip()
            # protocols/t2_rollout.md step 3: utterance -> R0 reply -> ledger update
            reply = None
            if not empty:
                hist_msgs = []
                for i, u in enumerate(hist_u):
                    hist_msgs.append({"role": "user", "content": u})
                    if i < len(hist_a):
                        hist_msgs.append({"role": "assistant", "content": hist_a[i]})
                reply = r0.reply(hist_msgs)
                prev_reply = hist_a[-1] if hist_a else ""
                hist_a.append(reply)
                ledger.update(t, text, reply)
                # StoppingLedger sees the assistant turn with EMPTY annotations (T2)
                led.observe({}, reply, prev_reply)
                ag.retire_satisfied(reply, {})
                if AG.looks_like_new_offer(reply, prev_reply):
                    ag.reset_on_new_offer()

            trace.append({"t": t, "stop_gate": False, "p_stop": round(p_stop, 6),
                          "ended_speaker": bool(ended_speaker),
                          "ended_planner": ended_planner, "empty": empty,
                          "move": fields.get("move", ""), "act": fields.get("act", ""),
                          "stop_rule": fields.get("stop_rule", "none"),
                          "guard_reasons": reasons, "guard_extra": n_extra,
                          "no_survivor": no_surv, "selected_index": idx,
                          "ledger_before": led_before,
                          "coverage_after": round(ledger.coverage(), 4),
                          "user": text, "agent": reply})

            prev_block = block
            # episode ends on the method stop signal (or an empty message), else T_max
            #
            # protocols/t2_rollout.md ends an episode when THE METHOD emits its
            # stop signal. The planner's act IS the method's signal: ended_planner
            # is computed above from state.ends_session and was, until now,
            # recorded in the trace and then discarded, so only the frozen
            # speaker's own token could end an episode. That is why 51 of 64
            # episodes ran to T_max.
            #
            # Gated by SEPSIM_PLANNER_END so the published run reproduces exactly
            # with the flag unset. Offline truncation of episodes_full.jsonl at
            # the first ended_planner gives the exact counterfactual, because the
            # rollout is causal and coverage is latched monotonically:
            # rollout_turns 8.4375 -> 7.4844, ended_by_token 0.2031 -> 0.5312,
            # coverage 0.2759 -> 0.2679.
            if (PLANNER_END and ended_planner) or ended_speaker or empty:
                ended_by_token = True
                stop_kind = ("empty" if empty
                             else ("end_token" if ended_speaker else "planner"))
                break

        row = {"conversation_id": cid, "record_id": rid, "seed": seed, "arm": args.tag, "adapter": args.adapter or None, "speaker": args.speaker,
               "turns": turns, "ended_by_token": ended_by_token,
               "coverage": round(ledger.coverage(), 4), "complete": ledger.complete(),
               "stop_kind": stop_kind, "n_req": len(req_items),
               "ledger": ledger.as_dict(), "trace": trace}
        pipeline.write_jsonl([row], args.out)
        n_done += 1
        log("  [%d] %s seed=%d turns=%d end=%s cov=%.3f complete=%s (%.0fs, r0=%d judge=%d/%d)"
            % (n_done, cid[:8], seed, turns, stop_kind,
               ledger.coverage(), ledger.complete(), time.time() - t0,
               r0.n_calls, judge.n_calls, judge.n_cached))

    log("done: %d episodes in %.0fs" % (n_done, time.time() - t0))
    log("guards: %d candidates scored, %d blocked (empty=%d template=%d reuse=%d), "
        "%d extra draws, %d turns with no survivor, over %d turns"
        % (G["cands"], G["blocked"], G.get("empty", 0), G.get("template", 0),
           G.get("reuse", 0), G["extra"], G["nosurv"], G["turns"]))
    log("planner unparsed: %d" % G["unparsed"])
    log("speaker rejects: template %d, reuse %d, regen %d"
        % (speaker.n_reject_template, speaker.n_reject_reuse, speaker.n_regen))
    log("API: r0=%s | judge_calls=%d judge_cached=%d"
        % (json.dumps(r0.stats()), judge.n_calls, judge.n_cached))
    json.dump({"guards": G, "r0": r0.stats(), "judge_calls": judge.n_calls,
               "judge_cached": judge.n_cached,
               "speaker_rejects": {"template": speaker.n_reject_template,
                                   "reuse": speaker.n_reject_reuse,
                                   "regen": speaker.n_regen}},
              open(os.path.join(WORK, "runstats_%s.json" % args.tag), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
