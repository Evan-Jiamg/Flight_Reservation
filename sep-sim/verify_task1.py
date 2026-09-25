#!/usr/bin/env python3
"""Integrity checks for Task 1 generations written by task1_v4.py (gen.jsonl + gen.jsonl.k1.jsonl + meta).

FAIL (exit 1) on:
  structure   every scored conversation has exactly its real turns 1..n, once; one K+1 row at n+1
  trunc       Planner prompt above its budget; Speaker prompt above its budget; the SELECTED candidate cut
              by the Speaker's token cap
  leakage     a few-shot example from the same conversation, the same goal, the same persona, or (fold
              runs) from a validation/test conversation
  profile     a profile note on turn 1 (there is no earlier message to compare with)
  selection   the selected candidate failed a guard while another passed
  compaction  any Planner or Speaker (any candidate) prompt compacted -- no history is ever dropped
  M2          greedy_ended != (Speaker blank at t OR Planner end at t-1); K+1 ended != (end at n OR blank)
  capped      a capped (cut) message emitted
WARN on: Planner max_new hits > 2%, unparsed plans, any candidate cut by the Speaker cap, duplicate
first-turn candidates left after redraws.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class Rep:
    def __init__(self):
        self.fail, self.warn, self.note = [], [], []

    def ok(self, name, cond, msg):
        if not cond:
            self.fail.append("%s: %s" % (name, msg))
        return cond


def check_fits(rep, w, r, pb, sb):
    pf = r.get("planner_fit") or {}
    rep.ok("compaction.planner", not pf.get("compacted"), "%s Planner prompt compacted" % w)
    fits = r.get("speaker_fits")
    if rep.ok("trunc.speaker_fits_recorded", isinstance(fits, list) and fits, "%s speaker_fits missing" % w):
        for j, f in enumerate(fits):
            f = f or {}
            tk = f.get("final_tokens") if f.get("compacted") else f.get("original_tokens")
            rep.ok("trunc.speaker", tk is not None and tk <= sb, "%s candidate %d speaker prompt %s > %s" % (w, j, tk, sb))
            rep.ok("compaction.speaker", not f.get("compacted"), "%s candidate %d Speaker prompt compacted" % (w, j))
    rep.ok("capped", not r.get("emitted_capped"), "%s a capped message was emitted" % w)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--corpus", default="/home/mzjiang/v5-latency/data.jsonl")
    ap.add_argument("--folds", default="/tmp2/hchsu/trec2026-usersim-benchmark/domains/main_dataset_search/folds3_goal_persona_v1.json")
    ap.add_argument("--splits", default="/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json")
    ap.add_argument("--fold", type=int, default=-1, help="fold run: few-shot must avoid this fold's forbidden ids")
    ap.add_argument("--max-new-warn", type=float, default=0.02)
    a = ap.parse_args(argv)
    import fit_prompts as F
    rep = Rep()
    rows = [json.loads(l) for l in open(a.gen, encoding="utf-8") if l.strip()]
    k1 = [json.loads(l) for l in open(a.gen + ".k1.jsonl", encoding="utf-8") if l.strip()]
    meta = json.load(open(a.gen + ".meta.json", encoding="utf-8"))
    desc = meta.get("describe", {})
    recs = {}
    for l in open(a.corpus, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            recs[r["conversation_id"]] = r
    F2 = json.load(open(a.folds, encoding="utf-8"))
    goal_of, persona_of = F2["goal_of"], F2["persona_of"]
    forbidden = set()
    if a.fold < 0 and meta.get("fold", -1) >= 0:
        a.fold = int(meta["fold"])                     # a fold run is always checked against its fold
    if meta.get("sessions") in ("fold-validation", "fold-test") and a.fold < 0:
        rep.ok("leakage.fold", False, "fold run without a fold: the forbidden-pool check cannot run")
    if a.fold >= 0 and meta.get("sessions") in ("fold-validation", "fold-test"):
        spx = [x for x in json.load(open(a.splits, encoding="utf-8"))["folds"] if x["fold"] == a.fold][0]
        want = sorted(spx["validation"] if meta["sessions"] == "fold-validation" else spx["test_all"])
        if meta.get("limit"):
            want = want[: int(meta["limit"])]           # a --limit run scores the first N sorted ids (smoke only)
            rep.warn.append("limited run: %d of the split's sessions (not a result)" % len(want))
        rep.ok("structure", sorted(meta.get("session_ids") or []) == want,
               "scored ids are not splits[%d].%s" % (a.fold, "validation" if meta["sessions"] == "fold-validation" else "test_all"))
    if a.fold >= 0:
        sp = [x for x in json.load(open(a.splits, encoding="utf-8"))["folds"] if x["fold"] == a.fold][0]
        forbidden = set(sp["forbidden_for_training"])
    sys.path.insert(0, desc.get("tree", "/tmp2/mzjiang_usersim/grpo_planner/trees/e1r_cf19400"))
    from sepsim import pipeline
    by = defaultdict(list)
    for r in rows:
        by[r["conversation_id"]].append(r)
    k1_by = defaultdict(list)
    for r in k1:
        k1_by[r["conversation_id"]].append(r)
    pb = desc.get("planner_budget", F.PLANNER_BUDGET)
    sb = desc.get("speaker_budget", F.SPEAKER_BUDGET)
    n_steps = n_hit = n_unp = n_cand = n_cand_hit = n_first_dup = 0
    n_end_no_complete = n_complete_kept = 0
    for cid, rs in by.items():
        users, _ = pipeline.split_messages(recs[cid])
        n = len(users)
        ts = sorted(r["turn_index"] for r in rs)
        rep.ok("structure", ts == list(range(1, n + 1)), "%s turns %s != 1..%d" % (cid[:10], ts[:12], n))
        kk = k1_by.get(cid, [])
        rep.ok("structure", len(kk) == 1 and kk[0]["turn_index"] == n + 1, "%s K+1 rows %d" % (cid[:10], len(kk)))
        for r in kk:                                   # the K+1 turn is generated like any other: same checks
            w = "%s K+1" % cid[:10]
            reasons = r.get("guard_reasons") or []
            elig = [i for i, x in enumerate(reasons) if not x]
            if elig and r.get("selected_index") is not None:
                rep.ok("selection", r["selected_index"] in elig, "%s selected %s not in eligible %s" % (w, r["selected_index"], elig))
            for slot in (r.get("fewshot") or []):
                for ex_cid, _ in slot:
                    rep.ok("leakage.fewshot", ex_cid != cid and ex_cid not in forbidden
                           and goal_of.get(ex_cid) is not None and goal_of.get(ex_cid) != goal_of.get(cid)
                           and persona_of.get(ex_cid) is not None and persona_of.get(ex_cid) != persona_of.get(cid),
                           "%s example %s: same conversation/goal/persona, no ids, or validation/test" % (w, ex_cid[:10]))
            pt = r.get("planner_prompt_tokens")
            rep.ok("trunc.planner", pt is not None and pt <= pb, "%s prompt %s > %s" % (w, pt, pb))
            sf = r.get("speaker_fit") or {}
            stoks = sf.get("final_tokens") if sf.get("compacted") else sf.get("original_tokens")
            rep.ok("trunc.speaker", stoks is not None and stoks <= sb, "%s speaker prompt %s > %s" % (w, stoks, sb))
            hits = r.get("speaker_hit_max_new")
            rep.ok("trunc.speaker_recorded", isinstance(hits, list) and all(h is not None for h in hits), "%s cap hits unknown" % w)
            if isinstance(hits, list) and r.get("selected_index") is not None:
                rep.ok("trunc.speaker_selected", not hits[r["selected_index"]], "%s selected candidate cut by the cap" % w)
            n_hit += bool(r.get("planner_hit_max_new"))
            n_steps += 1
            last = max(rs, key=lambda x: x["turn_index"]) if rs else None
            if last is not None:
                exp = bool(last.get("planner_ends_session")) or bool(r.get("ended_speaker"))
                rep.ok("M2", r.get("end_mapping") == "M2" and bool(r.get("ended")) == exp and bool(r.get("greedy_ended")) == exp,
                       "%s K+1 ended %r != end at n %r OR blank %r" % (w, r.get("ended"), last.get("planner_ends_session"),
                                                                       r.get("ended_speaker")))
            check_fits(rep, w, r, pb, sb)
        prev_dec = False
        for r in sorted(rs, key=lambda x: x["turn_index"]):
            w = "%s t%d" % (cid[:10], r["turn_index"])
            n_steps += 1
            n_hit += bool(r.get("planner_hit_max_new"))
            n_unp += bool(r.get("planner_unparsed"))
            pt = r.get("planner_prompt_tokens")
            rep.ok("trunc.planner", pt is not None and pt <= pb, "%s prompt %s > %s" % (w, pt, pb))
            sf = r.get("speaker_fit") or {}
            stoks = sf.get("final_tokens") if sf.get("compacted") else sf.get("original_tokens")
            rep.ok("trunc.speaker", stoks is not None and stoks <= sb, "%s speaker prompt %s > %s" % (w, stoks, sb))
            rep.ok("planner_cap", not (r.get("planner_hit_max_new") and r.get("planner_ends_session")),
                   "%s a Planner output cut by the cap ended the session" % w)
            d = r.get("planner_diag") or {}
            n_end_no_complete += bool(d.get("end_without_complete_entry"))
            n_complete_kept += bool(d.get("complete_kept_no_alternative"))
            hits = r.get("speaker_hit_max_new")
            rep.ok("trunc.speaker_recorded", isinstance(hits, list) and all(h is not None for h in hits),
                   "%s speaker cap hits not recorded" % w)
            if isinstance(hits, list):
                n_cand += len(hits)
                n_cand_hit += sum(bool(h) for h in hits)
                si = r.get("selected_index")
                rep.ok("trunc.speaker_selected", si is None or not hits[si], "%s selected candidate cut by the cap" % w)
            exp = bool(r.get("speaker_ended")) or prev_dec
            rep.ok("M2", r.get("end_mapping") == "M2" and bool(r.get("greedy_ended")) == exp
                   and bool(r.get("ended_by_prev_decision")) == prev_dec, "%s greedy_ended %r != blank %r OR end at t-1 %r" % (
                       w, r.get("greedy_ended"), r.get("speaker_ended"), prev_dec))
            se = r.get("samples_speaker_ended")
            if isinstance(se, list):
                rep.ok("M2", [bool(x) for x in se] == [bool(x) for x in (r.get("samples_ended") or [])],
                       "%s samples_ended must be the Speaker flags (E1.6 convention)" % w)
            prev_dec = bool(r.get("planner_ends_session"))
            rep.ok("planner_diag", isinstance(r.get("planner_diag"), dict), "%s planner_diag missing" % w)
            check_fits(rep, w, r, pb, sb)
            reasons = r.get("guard_reasons") or []
            elig = [i for i, x in enumerate(reasons) if not x]
            if elig and r.get("selected_index") is not None:
                rep.ok("selection", r["selected_index"] in elig, "%s selected %s not in eligible %s" % (w, r["selected_index"], elig))
            if r["turn_index"] == 1:
                rep.ok("profile", not r.get("profile_note"), "%s has a profile note on turn 1" % w)
                allc = [r.get("greedy") or ""] + list(r.get("samples") or [])
                norm = [" ".join(x.lower().split()) for x in allc if x.strip()]
                n_first_dup += len(norm) - len(set(norm))
            for slot in (r.get("fewshot") or []):
                for ex_cid, _ in slot:
                    rep.ok("leakage.fewshot", ex_cid != cid, "%s example from the same conversation" % w)
                    rep.ok("leakage.fewshot", goal_of.get(ex_cid) is not None and goal_of.get(ex_cid) != goal_of.get(cid),
                           "%s example shares the goal (or has no goal id)" % w)
                    rep.ok("leakage.fewshot", persona_of.get(ex_cid) is not None and persona_of.get(ex_cid) != persona_of.get(cid),
                           "%s example shares the persona (or has no persona id)" % w)
                    rep.ok("leakage.fewshot", ex_cid not in forbidden, "%s example from validation/test %s" % (w, ex_cid[:10]))
    if "session_ids" in meta:
        missing = sorted(set(meta["session_ids"]) - set(by))
        extra = sorted(set(by) - set(meta["session_ids"]))
        rep.ok("structure", not missing, "%d scored session(s) have no rows: %s" % (len(missing), [c[:10] for c in missing[:5]]))
        rep.ok("structure", not extra, "%d row conversation(s) not in the scored list" % len(extra))
    else:
        rep.ok("structure", False, "meta has no session_ids: cannot check that every session was generated")
    if n_steps and n_hit / n_steps > a.max_new_warn:
        rep.warn.append("planner max_new hits %d/%d" % (n_hit, n_steps))
    if n_unp:
        rep.warn.append("unparsed plans %d/%d" % (n_unp, n_steps))
    if n_cand_hit:
        rep.warn.append("Speaker cap hits on %d of %d candidates" % (n_cand_hit, n_cand))
    if n_first_dup:
        rep.warn.append("duplicate first-turn candidates left: %d" % n_first_dup)
    if n_end_no_complete:
        rep.warn.append("end_session without a Complete entry (drawn act kept): %d" % n_end_no_complete)
    if n_complete_kept:
        rep.warn.append("Complete act kept while not ending (no alternative entry): %d" % n_complete_kept)
    print("conversations %d, rows %d, K+1 rows %d; implicit_profile %s, fewshot %s, selector %s" % (
        len(by), len(rows), len(k1), desc.get("implicit_profile"), desc.get("fewshot"), desc.get("selector")))
    for x in rep.warn:
        print("WARN", x)
    for x in rep.fail[:40]:
        print("FAIL", x)
    print("TASK1 VERIFICATION %s (%d failures)" % ("PASSED" if not rep.fail else "FAILED", len(rep.fail)))
    return 0 if not rep.fail else 1


if __name__ == "__main__":
    sys.exit(main())
