#!/usr/bin/env python3
"""Pipeline-integrity verifier: proves from the LOGS (not the code) that a run used the declared data
split, never truncated a prompt, and actually exercised every v3 design element. Supersedes
check_smoke_v3.py. Exit code 0 only when every check passes (WARN does not fail unless --strict).

Usage
  verify_pipeline.py --episodes a2.jsonl [more.jsonl ...] --meta run_meta_a2_rep0.json \
      --splits splits_v1.json --fold 0 --split validation [--arm a2] \
      [--expected-system-sha HEX | --sepsim-path /home/mzjiang/Sep-Simulator] [--json-out report.json]
  RL run (rollouts are training data, so --split train is required):
  verify_pipeline.py --rl-dir RUN --splits splits_v1.json --fold 0 --split train    (meta = RUN/run_meta.jsonl)

Inputs
  episodes   Task2Env / rollout_ditto_v3 rows (one episode per line, "trace" = list of steps). Rows may be
             wrapped as {"episode": {...}, "update": u, "split": ...} (train_planner_rl.py rollouts.jsonl /
             validation.jsonl); rows with kind == "summary" are skipped.
  meta       run_meta_*.json (rollout_ditto_v3), a Task2Env.describe() JSON, or train_planner_rl's
             run_meta.jsonl (config.env = describe(); row fold = gate fold; all rows must agree).
  splits     splits_v1.json: {"folds": [{"fold", "train", "validation", "test", *_all,
             "forbidden_for_training"}]}.
  --training marks the episode files as training data (SFT/RL input): ids must be in splits[f]["train"]
             and never in forbidden_for_training.

RL directory contract (train_planner_rl.py layout; checked with --rl-dir):
  rollouts*.jsonl or rollouts/*.jsonl  (wrapped) episode rows; every step has planner_gen {prompt_ids,
                                       gen_ids, temperature > 0}; wrapper "split", if present, is "train".
  updates.jsonl                        one row per update: "update" (int, strictly increasing); reward
                                       components as train_aggregate.components_mean or reward_components
                                       {name: number} (or every rollout row has reward_components);
                                       optional "checkpoint" (path, relative to the run dir or absolute).
  checkpoints                          row["checkpoint"] if given, else --ckpt-pattern
                                       (default ckpt/u{update:05d}) must exist.

Checks (name: what a FAIL means)
  io.*            unreadable file / missing meta fields
  leak.*          an id outside the declared split/fold; training data outside train or in forbidden
  episode.*       trace structure, emitted-turn count, silent exits carrying text
  trunc.*         any prompt above its budget (Planner, Speaker, judge 16384); Planner max_new hits (WARN > 2%)
  a2.*            v3 system prompt sha, GOAL STATUS in every Planner prompt with that step's status, no
                  removed ledger/agenda/band lines, stop = parsed end_session, no override, no clamp,
                  Speaker pending line = judge unmet text, judge step 1 NOT ASSESSED + raw outputs later
  a0.*            original prompt present (ledger, WHAT THEY STILL WANT, HOW LONG THEY WRITE), no GOAL STATUS
  rl.*            planner_gen, update index, checkpoints, reward components
  RL runs branch on run_meta config.spec_version (SPEC v17 B8, user 2026-09-30): "v17" -> check_v17 (SFT data and
  choice, the ref adapter, Brier Task 1 groups and where their advantages act, the stop supervision, 8 optimizer steps,
  the length drift / stop reason, the validation schedule, the tested updates, the Ditto architecture); anything else
  (v16 runs with task1_G, older runs) -> the v16 checks (refill, D2, aux floor, selection, re-selection, best.json).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_prompts as F   # noqa: E402  (pure python: budgets only)
import goal_judge as GJ   # noqa: E402  (pure python at import time)

CONV = "THE CONVERSATION SO FAR"
# planner_prompt_v3 constants (replicated so the verifier runs without sepsim; the test suite
# cross-checks them against planner_prompt_v3 when sepsim is importable)
GOAL_HEAD = "GOAL STATUS (assessed from the conversation by a separate model)"
FIRST_TURN_UNMET = "the whole request (nothing has been answered yet)"
UNKNOWN_UNMET = "(not available this turn: the assessment could not be read)"
NOTHING_UNMET = "(nothing identified)"
A2_FORBIDDEN = ("- useful replies", "- best offered so far", "- last reply repeated the previous offer",
                "stopping condition", "WHAT THEY STILL WANT", "HOW LONG THEY WRITE", "- band:",
                "unhelpful replies before frustration governs", "- condition met first")
A0_REQUIRED = ("- useful replies", "WHAT THEY STILL WANT", "HOW LONG THEY WRITE")
# E1.6-based arms (task2_env.ARM_ENV): e16 = E1.6 Planner side as generated (a0-like prompt, plus the
# ACT_FULL per-move band rows); final = v3 checks + the E1.6 switches Final keeps.
ARMS = ("a0", "a2", "e16", "final", "pend")
V3_ARMS = ("a2", "final")
E16_ARMS = ("e16", "final", "pend")
E16_ROWS = "- Disclose: "          # first ACT_FULL band row, rendered only with the band


def check_family(arm):
    """Which prompt/stop checks an arm gets: v3 (a2, final), pend, or original (a0, e16)."""
    return "a2" if arm in V3_ARMS else ("pend" if arm == "pend" else "a0")
JUDGE_STATUSES = ("SATISFIED", "PARTIAL", "NOT", "UNKNOWN")
SPEAKER_DECISIONS = ("continue", "empty", "speaker_end", "planner_end")


def unmet_text(gs):
    if gs is None or gs.get("status") == "NOT ASSESSED":
        return FIRST_TURN_UNMET
    if gs.get("status") == "UNKNOWN":
        return UNKNOWN_UNMET
    items = [u for u in (gs.get("unmet") or []) if u]
    return "; ".join(items) if items else NOTHING_UNMET


def static_part(prompt):
    return (prompt or "").split(CONV, 1)[0]


def parsed_end(raw):
    return raw is True or (isinstance(raw, str) and raw.strip().lower() == "true")


# ------------------------------------------------------------------ report
class Report:
    def __init__(self):
        self.checks = {}
        self.order = []

    def _c(self, name):
        if name not in self.checks:
            self.checks[name] = {"n": 0, "fail": 0, "warn": False, "examples": [], "notes": []}
            self.order.append(name)
        return self.checks[name]

    def ok(self, name, cond, where="", why=""):
        c = self._c(name)
        c["n"] += 1
        if not cond:
            c["fail"] += 1
            if len(c["examples"]) < 5:
                c["examples"].append(("%s: %s" % (where, why)).strip(": "))
        return cond

    def note(self, name, text):
        self._c(name)["notes"].append(text)

    def warn(self, name, text):
        c = self._c(name)
        c["warn"] = True
        c["notes"].append(text)

    def status(self, name):
        c = self.checks[name]
        if c["fail"]:
            return "FAIL"
        if c["warn"]:
            return "WARN"
        return "PASS" if c["n"] else "SKIP"

    def failed(self, strict=False):
        return any(self.status(n) == "FAIL" or (strict and self.status(n) == "WARN") for n in self.order)

    def table(self):
        lines = ["%-30s %-5s %8s %7s  %s" % ("check", "res", "checked", "failed", "notes")]
        for n in self.order:
            c = self.checks[n]
            lines.append("%-30s %-5s %8d %7d  %s" % (n, self.status(n), c["n"], c["fail"], "; ".join(c["notes"])))
            for e in c["examples"]:
                lines.append("    - " + e[:220])
        return "\n".join(lines)

    def as_dict(self):
        return {n: {**self.checks[n], "status": self.status(n)} for n in self.order}


# ------------------------------------------------------------------ helpers
def load_jsonl(path, rep):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    rep.ok("io.parse", True)
                    if r.get("kind") in ("summary", "task1"):
                        continue
                    if isinstance(r.get("episode"), dict) and "trace" in r["episode"]:
                        ep = dict(r["episode"])
                        rep.ok("io.wrapped_row_id", ep.get("conversation_id") == r.get(
                            "conversation_id", ep.get("conversation_id")),
                            "%s:%d" % (os.path.basename(path), i), "wrapper/episode id differ")
                        if "split" in r:
                            ep["_wrap_split"] = r["split"]
                        if "update" in r:
                            ep["update"] = r["update"]
                        r = ep
                    rows.append(r)
                except ValueError as e:
                    rep.ok("io.parse", False, "%s:%d" % (os.path.basename(path), i), str(e)[:80])
    except OSError as e:
        rep.ok("io.parse", False, path, str(e))
    return rows


def load_meta(path, rep):
    """run_meta JSON / describe() JSON, or train_planner_rl run_meta.jsonl (rows with config.env)."""
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        rep.ok("io.meta", False, path, str(e)[:100])
        return {}
    rows = None
    try:
        obj = json.loads(text)
    except ValueError:
        try:
            rows = [json.loads(l) for l in text.splitlines() if l.strip()]
        except ValueError as e:
            rep.ok("io.meta", False, path, str(e)[:100])
            return {}
        obj = rows[-1] if rows else {}
    if isinstance(obj, dict) and isinstance(obj.get("env"), dict) and rows is None:
        # rollout_v4 run_meta_<arm>_rep<k>.json: {"env": describe(), "gate": {...}, ...}
        meta = dict(obj["env"])
        meta["gate"] = obj.get("gate") or {}
        meta["smoke"] = obj.get("smoke", False)
        rep.ok("io.meta", bool(meta.get("arm")), path, "run_meta env lacks arm")
        return meta
    if not (isinstance(obj, dict) and isinstance(obj.get("config"), dict) and "env" in obj["config"]):
        rep.ok("io.meta", isinstance(obj, dict) and bool(obj), path, "empty meta")
        return obj if isinstance(obj, dict) else {}
    rows = rows or [obj]
    envs = [((r.get("config") or {}).get("env") or {}) for r in rows]
    rep.ok("io.meta", all(envs), path, "a run_meta row lacks config.env")
    for k in ("system_prompt_sha256", "arm", "act_prior", "planner_budget", "speaker_budget"):
        vals = {json.dumps(e.get(k), sort_keys=True) for e in envs}
        rep.ok("io.meta_consistent", len(vals) == 1, path, "%s changed across run_meta rows: %s" % (k, sorted(vals)))
    folds = {str(r.get("fold")) for r in rows}
    rep.ok("io.meta_consistent", len(folds) == 1, path, "fold changed across run_meta rows: %s" % sorted(folds))
    meta = dict(envs[-1])
    meta["gate"] = {"fold": rows[-1].get("fold")}
    return meta


def recompute_system_sha(arm, sepsim_path=None, implicit_profile=False):
    """sha256 of the system prompt the arm must use, computed in a clean subprocess (the env var
    SEPSIM_ACT_PRIOR has to be set before sepsim is imported). None when sepsim is not importable."""
    import task2_env as T2   # constants only; no torch at import
    if arm in E16_ARMS:
        defaults = (os.path.join(HERE, "..", "audit_e1r"), T2.TREE_E16)
    else:
        defaults = (os.path.join(HERE, "..", "audit_src"), T2.TREE_V2FIX)
    cands = [p for p in (sepsim_path,) + defaults if p]
    root = next((p for p in cands if os.path.isdir(os.path.join(p, "sepsim"))), None)
    if root is None:
        return None
    code = ("import hashlib,sys; sys.path[:0]=[%r,%r]\n" % (os.path.abspath(root), HERE) +
            ("import planner_prompt_v3 as V; s=V.system_prompt_v3()\n" if arm in V3_ARMS else
             ("import planner_prompt_v3 as V; s=V.system_prompt_pend(implicit_profile=%r)\n" % bool(implicit_profile))
             if arm == "pend" else
             "from sepsim import planner_prompt as PP; s=PP.system_prompt()\n") +
            "print(hashlib.sha256(s.encode()).hexdigest())")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SEPSIM_")}
    env.update(T2.ARM_ENV[arm])
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    except Exception:  # noqa: BLE001
        return None
    s = out.stdout.strip().splitlines()
    return s[-1] if out.returncode == 0 and s and len(s[-1]) == 64 else None


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ------------------------------------------------------------------ checks
def check_meta(meta, arm, fold, rep):
    for k in ("arm", "system_prompt_sha256"):
        rep.ok("io.meta_fields", k in meta, "meta", "missing %s" % k)
    rep.ok("io.meta_arm", meta.get("arm") == arm, "meta", "meta arm %r != %r" % (meta.get("arm"), arm))
    gf = (meta.get("gate") or {}).get("fold")
    if gf is not None and gf >= 0:
        rep.ok("leak.meta_fold", gf == fold, "meta.gate", "run was gated for fold %s, declared %s" % (gf, fold))
    ap = str(meta.get("act_prior", ""))
    if arm in E16_ARMS:
        rep.ok("e16.meta_tree", "e1r" in str(meta.get("tree", "")), "meta", "tree %r is not the E1.6 tree" % meta.get("tree"))
    if arm in V3_ARMS + ("pend",):
        rep.ok("a2.meta_act_prior", ap.startswith("nostopclobber"), "meta", "act_prior %r" % ap)
    elif ap:
        rep.ok("a0.meta_act_prior", ap.startswith("off"), "meta", "act_prior %r" % ap)


def check_system_sha(meta, arm, expected, rep):
    name = "%s.system_prompt_sha" % arm
    if not expected:
        rep.ok(name, False, "meta", "no expected sha (pass --expected-system-sha or make sepsim importable)")
        return
    rep.ok(name, meta.get("system_prompt_sha256") == expected, "meta",
           "sha %s != expected %s" % (str(meta.get("system_prompt_sha256"))[:12], expected[:12]))


def check_leakage(rows, splits, fold, split, training, rep):
    folds = {int(f["fold"]): f for f in splits["folds"]}
    if not rep.ok("leak.fold_declared", fold in folds, "splits", "fold %s not in split file" % fold):
        return
    f = folds[fold]
    if not rep.ok("leak.split_declared", split in f, "splits", "split %r not in fold %d" % (split, fold)):
        return
    members = set(f[split])
    forb = set(f.get("forbidden_for_training", []))
    train = set(f["train"])
    if training:
        rep.ok("leak.training_split", split == "train", "args", "training data declared as split %r" % split)
    for r in rows:
        cid = r.get("conversation_id")
        where = "%s s%s" % (str(cid)[:12], r.get("seed"))
        rep.ok("leak.split_membership", cid in members, where, "not in fold %d %s" % (fold, split))
        if "_wrap_split" in r:
            rep.ok("leak.row_split", r["_wrap_split"] in (split, split.replace("_all", "")), where,
                   "row split %r != declared %r" % (r["_wrap_split"], split))
        rf = r.get("fold")
        if rf is not None and rf >= 0:
            rep.ok("leak.row_fold", rf == fold, where, "row fold %s != declared %s" % (rf, fold))
        if training:
            rep.ok("leak.training_train_only", cid in train, where, "training row not in splits[%d].train" % fold)
            rep.ok("leak.training_forbidden", cid not in forb, where, "training row in forbidden_for_training")


def check_structure(rows, arm, rep):
    for r in rows:
        where = "%s s%s" % (str(r.get("conversation_id"))[:12], r.get("seed"))
        tr = r.get("trace") or []
        if not rep.ok("episode.structure", bool(tr), where, "empty trace"):
            continue
        rep.ok("episode.structure", [s.get("t") for s in tr] == list(range(1, len(tr) + 1)), where, "t not 1..n")
        rep.ok("episode.structure", r.get("decision_steps") == len(tr), where,
               "decision_steps %s != %d" % (r.get("decision_steps"), len(tr)))
        rep.ok("episode.arm", r.get("arm") == arm, where, "row arm %r != %r" % (r.get("arm"), arm))
        n_text = sum(1 for s in tr if (s.get("user") or "").strip())
        rep.ok("episode.emitted_count", r.get("emitted_user_turns") == n_text, where,
               "emitted_user_turns %s != %d non-empty user steps" % (r.get("emitted_user_turns"), n_text))
        rep.ok("episode.emitted_flags", all(bool(s.get("emitted")) == bool((s.get("user") or "").strip())
                                            for s in tr), where, "emitted flag disagrees with user text")
        for s in tr:
            if s.get("planner_stop") or s.get("decision") == "planner_stop":
                w = "%s t%s" % (where, s.get("t"))
                rep.ok("episode.silent_exit", s.get("user") == "" and s.get("emitted") is False
                       and s.get("agent") is None, w, "planner_stop step carries text/emission/agent reply")
                rep.ok("episode.silent_exit", s is tr[-1] and r.get("end_kind") == "planner_stop", w,
                       "planner_stop not the last step / end_kind %r" % r.get("end_kind"))
                rep.ok("episode.silent_exit", "block" not in s and "speaker_fit" not in s, w,
                       "Speaker was called on a silent exit")
        if arm in ("a0", "e16", "pend"):
            rep.ok("a0.no_planner_exit", r.get("end_kind") != "planner_stop", where, "%s executed a silent Planner exit" % arm)
        if arm == "pend":
            rep.ok("pend.human_turns", isinstance(r.get("human_turns"), int) and r["human_turns"] >= 1, where,
                   "human_turns missing")
        if arm in ("e16", "pend"):
            # E1.6 PLANNER_END: the first step whose plan ends the session is emitted and is the last step
            first = next((s for s in tr if s.get("ended_planner")), None)
            if first is not None:
                ok_emit = first.get("emitted") and r.get("end_kind") in ("planner_end", "speaker_end")
                ok_blank = (not first.get("emitted")) and r.get("end_kind") == "empty"      # blank close = END
                rep.ok("e16.planner_end", first is tr[-1] and (ok_emit or ok_blank), where,
                       "Planner ended at t%s but episode went on / end_kind %r" % (first.get("t"), r.get("end_kind")))
        if arm in E16_ARMS:
            for s in tr:
                if "block" in s:
                    rep.ok("e16.t1_sample", s.get("t1_sampled") == (s.get("t") == 1), "%s t%s" % (where, s.get("t")),
                           "t1_sampled %r on t%s" % (s.get("t1_sampled"), s.get("t")))
                if arm in ("final", "pend"):
                    rep.ok("final.no_self_judge", s.get("self_judge") is None, "%s t%s" % (where, s.get("t")),
                           "SELF_JUDGE fed the ledger in final")
                elif s.get("t", 0) >= 2 and not s.get("planner_unparsed"):
                    rep.ok("e16.self_judge", isinstance(s.get("self_judge"), dict), "%s t%s" % (where, s.get("t")),
                           "SELF_JUDGE not recorded")


def check_truncation(rows, arm, meta, speaker_budget, judge_budget, max_new_warn, rep):
    n_steps = n_hit = n_hit_known = n_pc = n_sc = n_jd = 0
    for r in rows:
        for s in r.get("trace") or []:
            w = "%s s%s t%s" % (str(r.get("conversation_id"))[:12], r.get("seed"), s.get("t"))
            if s.get("decision") == "stop_gate":
                continue
            n_steps += 1
            pf = s.get("planner_fit")
            if not isinstance(pf, dict):
                rep.ok("trunc.planner", False, w, "planner_fit missing")
                continue
            n = pf.get("prompt_tokens")
            if n is None and "original_tokens" in pf:
                n = pf["original_tokens"] - (pf.get("removed_tokens") or 0) if pf.get("compacted") else pf["original_tokens"]
            b = pf.get("budget", meta.get("planner_budget"))
            rep.ok("trunc.planner", n is not None and b is not None and n <= b, w,
                   "prompt_tokens %s budget %s" % (n, b))
            n_pc += bool(pf.get("compacted"))
            if arm == "pend":
                rep.ok("trunc.planner_compacted", not pf.get("compacted"), w, "Planner prompt compacted (history dropped)")
            if "planner_hit_max_new" in s:
                n_hit_known += 1
                n_hit += bool(s["planner_hit_max_new"])
            if "block" in s or s.get("decision") in SPEAKER_DECISIONS:
                sf = s.get("speaker_fit")
                if not isinstance(sf, dict):
                    rep.ok("trunc.speaker", False, w, "speaker_fit missing on a Speaker step")
                else:
                    tok = sf.get("final_tokens") if sf.get("compacted") else sf.get("original_tokens")
                    rep.ok("trunc.speaker", tok is not None and tok <= speaker_budget, w,
                           "speaker tokens %s budget %s" % (tok, speaker_budget))
                    n_sc += bool(sf.get("compacted"))
                    if arm == "pend":
                        fits = s.get("speaker_fits")
                        rep.ok("trunc.speaker_fits_recorded", isinstance(fits, list) and len(fits) == len(s.get("candidates") or []),
                               w, "per-candidate speaker_fits missing")
                        for j, f in enumerate(fits or []):
                            f = f or {}
                            tk = f.get("final_tokens") if f.get("compacted") else f.get("original_tokens")
                            rep.ok("trunc.speaker", tk is not None and tk <= speaker_budget, w,
                                   "candidate %d speaker tokens %s budget %s" % (j, tk, speaker_budget))
                            rep.ok("trunc.speaker_compacted", not f.get("compacted"), w,
                                   "candidate %d Speaker prompt compacted (history dropped)" % j)
            gs = s.get("goal_status")
            if arm in V3_ARMS and isinstance(gs, dict) and gs.get("status") != "NOT ASSESSED":
                pt = gs.get("prompt_tokens")
                rep.ok("trunc.judge", pt is not None and pt <= judge_budget, w,
                       "judge prompt_tokens %s budget %s" % (pt, judge_budget))
                n_jd += bool(gs.get("dropped_exchanges"))
    rep.note("trunc.planner", "compacted %d/%d" % (n_pc, n_steps))
    rep.note("trunc.speaker", "compacted %d" % n_sc)
    if arm in V3_ARMS:
        rep.note("trunc.judge", "dropped-exchange prompts %d" % n_jd)
    name = "trunc.planner_max_new"
    rep._c(name)
    if n_hit_known == 0:
        rep.warn(name, "planner_hit_max_new not recorded (runner predates Task2Env)")
    else:
        rate = n_hit / n_hit_known
        rep.checks[name]["n"] = n_hit_known
        (rep.warn if rate > max_new_warn else rep.note)(name, "hit max_new %d/%d = %.1f%%" % (
            n_hit, n_hit_known, 100 * rate))
        if n_hit_known < n_steps:
            rep.warn(name, "recorded on only %d/%d steps" % (n_hit_known, n_steps))


def check_a2(rows, rep, arm="a2"):
    n_unparsed = n_end = n_unknown = n_probs = 0
    for r in rows:
        tr = r.get("trace") or []
        for s in tr:
            w = "%s s%s t%s" % (str(r.get("conversation_id"))[:12], r.get("seed"), s.get("t"))
            gs = s.get("goal_status")
            if not rep.ok("a2.judge_outputs", isinstance(gs, dict) and "status" in gs, w, "goal_status missing"):
                continue
            st = gs["status"]
            # ---- judge outputs
            if s.get("t") == 1:
                rep.ok("a2.judge_outputs", st == "NOT ASSESSED", w, "step 1 status %r (must be NOT ASSESSED)" % st)
            else:
                rep.ok("a2.judge_outputs", st in JUDGE_STATUSES and isinstance(gs.get("raw"), str), w,
                       "status %r / raw output missing" % st)
                if "parse_ok" in gs:
                    rep.ok("a2.judge_outputs", (st == "UNKNOWN") == (gs["parse_ok"] is False), w,
                           "parse_ok %r with status %r" % (gs["parse_ok"], st))
                n_unknown += st == "UNKNOWN"
                if "status_probs" in gs:
                    n_probs += 1
                    p = gs["status_probs"]
                    good = (isinstance(p, dict) and p and set(p) <= {"SATISFIED", "PARTIAL", "NOT"}
                            and all(is_num(v) and 0 <= v <= 1 for v in p.values())
                            and abs(sum(p.values()) - 1) < 1e-3)
                    rep.ok("a2.judge_outputs", good, w, "status_probs invalid %r" % (p,))
            # ---- Planner prompt carries the goal status, and none of the removed parts
            stat = static_part(s.get("planner_prompt"))
            lines = set(stat.split("\n"))
            rep.ok("a2.goal_status_in_prompt", GOAL_HEAD in stat and ("- status: %s" % st) in lines, w,
                   "GOAL STATUS header or '- status: %s' missing" % st)
            bad = [b for b in A2_FORBIDDEN + ((E16_ROWS,) if arm == "final" else ()) if b in stat]
            rep.ok("a2.removed_lines_absent", not bad, w, "found %s" % bad)
            # ---- stop decision, override, clamp
            d = s.get("planner_diag") or {}
            if s.get("planner_unparsed"):
                n_unparsed += 1
                rep.ok("a2.end_session", s.get("ended_planner") is False, w, "unparsed plan but ended_planner true")
            else:
                if "end_session_raw" not in d:
                    rep.ok("a2.end_session", False, w, "planner_diag.end_session_raw missing")
                else:
                    t1_ign = bool(d.get("end_session_t1_ignored"))
                    rep.ok("a2.end_session", not t1_ign or s.get("t") == 1, w, "end_session ignored outside turn 1")
                    exp_end = parsed_end(d["end_session_raw"]) and not t1_ign
                    rep.ok("a2.end_session", bool(s.get("ended_planner")) == exp_end, w,
                           "ended_planner %r vs end_session_raw %r (t1 ignored %r)" % (s.get("ended_planner"), d["end_session_raw"], t1_ign))
                rep.ok("a2.no_length_clamp", d.get("length_clamped") is False, w,
                       "length_clamped %r" % d.get("length_clamped"))
            rep.ok("a2.no_stop_override", d.get("stop_override") is not True, w, "stop override applied")
            ended = bool(s.get("ended_planner"))
            n_end += ended
            rep.ok("a2.end_session", ended == (s.get("decision") == "planner_stop"), w,
                   "ended_planner %r but decision %r (silent exit not executed / spurious)" % (ended, s.get("decision")))
            # ---- Speaker block pending line = judge unmet text
            if "block" in s and not s.get("planner_unparsed"):
                pend = [l for l in (s["block"] or "").split("\n") if l.startswith("- pending:")]
                exp = "- pending: " + unmet_text(gs)
                rep.ok("a2.pending_line", pend == [exp], w, "pending %r != %r" % (pend, exp))
    rep.note("a2.end_session", "Planner exits %d; unparsed plans %d" % (n_end, n_unparsed))
    rep.note("a2.judge_outputs", "UNKNOWN %d; with status_probs %d" % (n_unknown, n_probs))


PEND_FORBIDDEN = A2_FORBIDDEN + ("GOAL STATUS", E16_ROWS)


def check_r0_attribution(rows, rep):
    """Every R0 reply cut by the token budget must be charged to the episode that received it (that
    episode is then unclean, pend.clean_flag, and never enters a reward group / evaluation). The
    *_total fields are running totals of one process, so rows are grouped by process_token (all rows
    of the run: train rollouts AND validation, which share the process) and the largest total of a
    process must equal the sum of its episodes' own counts. Rows without a token (written before the
    token existed) are pooled: their per-episode sum must at least cover the largest total seen."""
    groups = {}
    for r in rows:
        if "r0_len_truncated_total" not in r:
            continue
        g = groups.setdefault(r.get("process_token"), [0, 0])
        g[0] = max(g[0], r.get("r0_len_truncated_total") or 0)
        g[1] += (r.get("episode_counters") or {}).get("r0_len_truncated", 0)
    for tok, (total, attributed) in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if tok is None:
            rep.ok("trunc.r0_attributed", attributed >= total, "legacy rows (no process_token)",
                   "R0 truncations counted %d, charged to episodes %d" % (total, attributed))
        else:
            rep.ok("trunc.r0_attributed", attributed == total, "process %s" % tok,
                   "R0 truncations counted %d, charged to episodes %d" % (total, attributed))
    rep._c("trunc.r0_attributed")


def check_pend(rows, rep, arm="pend"):
    """pend: v3 prompt without the goal judge; Planner end = the planned message is the last one."""
    n_end = n_unparsed = 0
    for r in rows:
        for s in r.get("trace") or []:
            w = "%s s%s t%s" % (str(r.get("conversation_id"))[:12], r.get("seed"), s.get("t"))
            stat = static_part(s.get("planner_prompt"))
            bad = [b for b in PEND_FORBIDDEN if b in stat]
            rep.ok("pend.removed_lines_absent", not bad, w, "found %s" % bad)
            rep.ok("pend.facts", "- turns so far:" in stat, w, "turn count fact missing")
            rep.ok("pend.no_goal_judge", s.get("goal_status") is None, w, "goal_status present without a judge")
            d = s.get("planner_diag") or {}
            ended = bool(s.get("ended_planner"))
            if s.get("planner_unparsed"):
                n_unparsed += 1
                rep.ok("pend.end_session", not ended, w, "unparsed plan but ended_planner true")
            else:
                if "end_session_raw" not in d:
                    rep.ok("pend.end_session", False, w, "planner_diag.end_session_raw missing")
                else:
                    t1_ign = bool(d.get("end_session_t1_ignored"))
                    rep.ok("pend.end_session", not t1_ign or s.get("t") == 1, w, "end ignored outside turn 1")
                    rep.ok("pend.end_session", ended == (parsed_end(d["end_session_raw"]) and not t1_ign), w,
                           "ended_planner %r vs raw %r" % (ended, d["end_session_raw"]))
                rep.ok("pend.no_length_clamp", d.get("length_clamped") is False, w, "length clamped")
                if ended:
                    rep.ok("pend.close_act", s.get("move") == "Complete" or d.get("end_without_complete_entry"), w,
                           "Planner ended but the act is %r" % s.get("move"))
                if "block" in s:
                    pend_line = [l for l in (s["block"] or "").split("\n") if l.startswith("- pending:")]
                    exp = "- pending: " + (s.get("still_wanted") or "(nothing named)")
                    rep.ok("pend.pending_line", pend_line == [exp], w, "pending %r != %r" % (pend_line, exp))
            rep.ok("pend.no_stop_override", d.get("stop_override") is not True, w, "stop override applied")
            n_end += ended
    rep.note("pend.end_session", "Planner ends %d; unparsed plans %d" % (n_end, n_unparsed))
    for r in rows:
        w = "%s s%s" % (str(r.get("conversation_id"))[:12], r.get("seed"))
        rep.ok("pend.zero_turns", (r.get("emitted_user_turns") or 0) > 0, w, "episode with no emitted message")
        if "clean" in r or "episode_counters" in r:
            c = r.get("episode_counters") or {}
            exp = (c.get("r0_len_truncated", 0) == 0 and c.get("r0_empty", 0) == 0 and c.get("judge_empty", 0) == 0
                   and c.get("judge_unparseable", 0) == 0 and not r.get("emitted_capped_steps")
                   and not r.get("compacted_steps") and (r.get("emitted_user_turns") or 0) > 0)
            rep.ok("pend.clean_flag", r.get("clean") is exp, w, "clean %r but counters %r" % (r.get("clean"), c))
        else:
            rep.ok("pend.clean_flag", False, w, "episode has no clean flag / per-episode counters")
        for s in r.get("trace") or []:
            ws = "%s t%s" % (w, s.get("t"))
            rep.ok("trunc.emitted_capped", not s.get("emitted_capped"), ws, "a capped (cut) message was emitted")
            g = s.get("planner_gen")
            if isinstance(g, dict):
                d = s.get("planner_diag") or {}
                decision = (not s.get("planner_unparsed") and not g.get("hit_max_new") and s.get("t", 0) >= 2
                            and d.get("end_session_valid") is True and not d.get("end_session_t1_ignored"))
                if not decision:
                    rep.ok("rl.stop_mask_gated", g.get("stop_mask") is None, ws,
                           "stop credit on a step that is not a decision (unparsed / capped / turn 1 / invalid)")
                for key in ("stop_mask", "note_mask"):
                    m = g.get(key)
                    if m is not None:
                        rep.ok("rl.mask_length", len(m) == len(g.get("gen_ids") or []) and 1 in m, ws,
                               "%s length %d != %d generated tokens (or empty)" % (key, len(m), len(g.get("gen_ids") or [])))
    # generation-stage checks (fields written by Task2Env._pend_generate)
    import implicit_profile as IP
    n_hit_sel = n_hit_any = n_cand = n_dup_left = 0
    for r in rows:
        for s in r.get("trace") or []:
            if "candidates" not in s:
                continue
            w = "%s s%s t%s" % (str(r.get("conversation_id"))[:12], r.get("seed"), s.get("t"))
            cands, reasons = s["candidates"], s.get("guard_reasons") or [""] * len(s["candidates"])
            hits = s.get("speaker_hit_max_new")
            rep.ok("trunc.speaker_max_new_recorded", isinstance(hits, list) and len(hits) == len(cands)
                   and all(h is not None for h in hits), w, "speaker_hit_max_new missing or unknown (None)")
            if isinstance(hits, list) and len(hits) == len(cands):
                n_cand += len(cands)
                n_hit_any += sum(bool(h) for h in hits)
                n_hit_sel += bool(hits[s["selected_index"]])
                rep.ok("trunc.speaker_selected", not hits[s["selected_index"]], w,
                       "the emitted message was cut by the Speaker's token cap")
            for i, c in enumerate(cands):
                if IP.duplicate_of(c, cands[:i]):
                    rep.ok("pend.duplicates_flagged", bool(reasons[i]), w, "candidate %d duplicates an earlier one unflagged" % i)
                    n_dup_left += 1
            rep.ok("pend.state_clean", not IP.leaks_scaffold(s.get("block") or "")
                   and "this is their last message" not in (s.get("block") or "")
                   and "- stopping:" not in (s.get("block") or ""), w,
                   "Speaker-only lines (notes / examples / last-message / stopping line) leaked into the Planner's state")
            elig = [i for i, x in enumerate(reasons) if not x]
            if elig:
                rep.ok("pend.selected_eligible", s["selected_index"] in elig, w,
                       "selected candidate %d failed a guard while %s passed" % (s["selected_index"], elig))
    name = "trunc.speaker_max_new"
    rep._c(name)
    (rep.warn if n_hit_any else rep.note)(name, "Speaker cap hits: %d of %d candidates, %d selected" % (n_hit_any, n_cand, n_hit_sel))
    n_kept = sum(1 for r in rows for s in (r.get("trace") or []) if (s.get("planner_diag") or {}).get("complete_kept_no_alternative"))
    name = "pend.complete_kept"
    rep._c(name)
    (rep.warn if n_kept else rep.note)(name, "Complete act kept while not ending (no alternative entry): %d" % n_kept)
    name = "pend.duplicates_left"
    rep._c(name)
    (rep.warn if n_dup_left else rep.note)(name, "duplicate candidates left after redraws: %d" % n_dup_left)
    unp = max([r.get("ledger_judge_unparseable_total") or 0 for r in rows] or [0])
    name = "pend.ledger_judge_unparseable"
    rep._c(name)
    (rep.warn if unp else rep.note)(name, "ledger-judge answers that are not a verdict object (running total) %d" % unp)
    r0t = max([r.get("r0_len_truncated_total") or 0 for r in rows] or [0])
    rep.note("trunc.r0_reply", "R0 replies cut by the token budget (running total, per process) %d; "
             "attribution: trunc.r0_attributed, exclusion: pend.clean_flag" % r0t)
    rep.note("trunc.r0_reply", "R0 length retries (running total) %d" % max([r.get("r0_len_retries_total") or 0 for r in rows] or [0]))
    # a ledger that never credits anything means its judge is answering empty (seen with a reasoning
    # model under max_tokens=200): coverage would be 0 everywhere, silently
    led = [r.get("ledger") or {} for r in rows if r.get("emitted_user_turns", 0) >= 2]
    if led:
        rep.ok("pend.ledger_alive", any(l.get("revealed_at") for l in led), "episodes",
               "no requirement was ever revealed in %d episodes: ledger judge broken?" % len(led))
    empt = max([r.get("ledger_judge_empty_total") or 0 for r in rows] or [0])
    name = "pend.ledger_judge_empty"
    rep._c(name)
    (rep.warn if empt else rep.note)(name, "empty ledger-judge answers (running total) %d" % empt)


def check_a0(rows, rep, arm="a0"):
    for r in rows:
        for s in r.get("trace") or []:
            w = "%s s%s t%s" % (str(r.get("conversation_id"))[:12], r.get("seed"), s.get("t"))
            stat = static_part(s.get("planner_prompt"))
            miss = [x for x in A0_REQUIRED if x not in stat]
            rep.ok("a0.original_prompt", not miss, w, "missing %s" % miss)
            rep.ok("a0.no_goal_status", "GOAL STATUS" not in stat and s.get("goal_status") is None, w,
                   "GOAL STATUS / goal_status present in a0")
            if "block" in s:
                rep.ok("a0.original_prompt", "- pending:" in (s["block"] or ""), w, "Speaker block lacks agenda pending")


_ADAPTER_NAMES = {}


def adapter_name(rl_dir, pv):
    """The name the trainer serves an adapter under: checkpoint pv (int) -> p<pv>-<sha12 of ckpt/u<pv>/adapter>;
    v17 B3: an SFT candidate ("sft_e<k>") -> sft_e<k>-<sha12 of ckpt/sft_e<k>/adapter>."""
    key = (rl_dir, pv)
    if key not in _ADAPTER_NAMES:
        import vllm_planner
        if isinstance(pv, str):
            if not (pv.startswith("sft_e") and pv[5:].isdigit()):
                raise ValueError("adapter tag %r" % pv)
            d, tag = os.path.join(rl_dir, "ckpt", pv, "adapter"), pv
        else:
            d, tag = os.path.join(rl_dir, "ckpt", "u%05d" % pv, "adapter"), "p%d" % pv
        _ADAPTER_NAMES[key] = ("%s-%s" % (tag, vllm_planner.sha_dir(d)[:12])) if os.path.isdir(d) else None
    return _ADAPTER_NAMES[key]


def spec_version(rl_dir):
    """v17 B8: the branch of an RL run = run_meta config.spec_version; a run without it is v16 (with task1_G) or older,
    and both take the v16 branch. None without run_meta."""
    meta = _jl(os.path.join(rl_dir, "run_meta.jsonl")) if rl_dir else []
    if not meta:
        return None
    cfg = meta[0].get("config") or {}
    return cfg.get("spec_version") or ("v16" if "task1_G" in (cfg.get("args") or {}) else "pre-v16")


def check_vllm_generation(rl_dir, rollouts, meta, rep, abort=0.1):
    """Planner generation on the vLLM server: every step carries the behaviour log-probs of its tokens and the name
    of the adapter that produced them, which must be the policy of that update (p<update-1>-<sha>); every update
    reports TIS statistics and a learner/vLLM mismatch below the abort threshold."""
    if meta.get("planner_backend") != "vllm":
        return
    for r in rollouts:
        u = r.get("update")
        for s in r.get("trace") or []:
            g = s.get("planner_gen") or {}
            w = "%s u%s t%s" % (str(r.get("conversation_id"))[:12], u, s.get("t"))
            lp, ids = g.get("gen_logprobs"), g.get("gen_ids") or []
            rep.ok("rl.vllm_logprobs", isinstance(lp, list) and len(lp) == len(ids), w,
                   "behaviour log-probs missing or not one per generated token")
            if isinstance(u, int):
                rep.ok("rl.vllm_adapter", g.get("gen_adapter") == adapter_name(rl_dir, u - 1), w,
                       "generated by %r, the policy of update %d is %r" % (g.get("gen_adapter"), u, adapter_name(rl_dir, u - 1)))
    t1p = os.path.join(rl_dir, "rollouts_task1.jsonl")
    if os.path.exists(t1p):
        for line in open(t1p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            u = r.get("update")
            for x in r.get("samples") or []:
                g = x.get("planner_gen") or {}
                w = "task1 %s t%s u%s" % (str(r.get("conversation_id"))[:12], r.get("t"), u)
                rep.ok("rl.vllm_logprobs", isinstance(g.get("gen_logprobs"), list)
                       and len(g["gen_logprobs"]) == len(g.get("gen_ids") or []), w, "behaviour log-probs missing")
                rep.ok("rl.vllm_adapter", g.get("gen_adapter") == adapter_name(rl_dir, u - 1), w,
                       "generated by %r, expected %r" % (g.get("gen_adapter"), adapter_name(rl_dir, u - 1)))
    up = os.path.join(rl_dir, "updates.jsonl")
    if os.path.exists(up):
        for line in open(up, encoding="utf-8"):
            if not line.strip():
                continue
            u = json.loads(line)
            st = u.get("learner_stats") or {}
            w = "update %s" % u.get("update")
            if st.get("skipped_update") or not st.get("n_tokens"):
                continue                     # no RL token in this update: nothing to weight
            mm = st.get("behav_mismatch_mean")
            rep.ok("rl.vllm_mismatch", mm is not None and mm <= abort, w,
                   "mean |log pi_learner - log pi_vllm| = %r (abort at %g)" % (mm, abort))
            rep.ok("rl.tis", st.get("tis_w_mean") is not None and st.get("tis_capped_frac") is not None, w,
                   "TIS statistics missing")
            if mm is not None:
                rep.note("rl.vllm_mismatch", "%s: mismatch %.4f, TIS mean weight %.3f, capped %.4f" % (
                    w, mm, st.get("tis_w_mean") or 0.0, st.get("tis_capped_frac") or 0.0))


def check_intervention(rl_dir, rep):
    """A user-approved intervention (run_meta rows "intervention"): one record for the whole run, the first
    update after it used the set values, and every later update's weights lie inside the new bounds."""
    mp, up = os.path.join(rl_dir, "run_meta.jsonl"), os.path.join(rl_dir, "updates.jsonl")
    if not os.path.exists(mp):
        return
    ivs = [r["intervention"] for r in (json.loads(l) for l in open(mp, encoding="utf-8") if l.strip())
           if r.get("intervention")]
    if not ivs:
        return
    rep.ok("rl.intervention", len({(i["at_update"], i["file_sha256"]) for i in ivs}) == 1, "run_meta",
           "more than one intervention record: %r" % sorted({(i["at_update"], i["file_sha256"]) for i in ivs}))
    iv = ivs[0]
    rows = [json.loads(l) for l in open(up, encoding="utf-8") if l.strip()] if os.path.exists(up) else []
    for u in rows:
        if u.get("update", 0) < iv["at_update"]:
            continue
        cfg, w = u.get("cfg_used") or {}, "update %s" % u.get("update")
        if u["update"] == iv["at_update"]:
            for k, v in iv["set_cfg"].items():
                rep.ok("rl.intervention", abs(float(cfg.get(k, float("nan"))) - float(v)) < 1e-9, w,
                       "%s used %r, intervention set %r" % (k, cfg.get(k), v))
        for k, (lo, hi) in (iv.get("controller_bounds") or {}).items():
            rep.ok("rl.intervention", cfg.get(k) == 0 or lo - 1e-9 <= float(cfg.get(k, float("nan"))) <= hi + 1e-9, w,
                   "%s=%r outside the intervention bounds [%g, %g]" % (k, cfg.get(k), lo, hi))
    rep.note("rl.intervention", "applied before update %s: %s bounds %s (%s)" % (
        iv["at_update"], iv["set_cfg"], iv.get("controller_bounds"), iv.get("approved")))


def sha256_file(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


def check_v16(rl_dir, rep, splits_path=None):
    """SPEC v16 (user 2026-09-28): Task 1 refill / group size, Dr. GRPO setting, aux floor and supervision, w_dist
    floor, the continuous Task 1 metric (recomputed from the logged end probabilities) and the two-point D2 trigger
    (recomputed from the validation bal_p values)."""
    import task1_stop as T1
    meta = _jl(os.path.join(rl_dir, "run_meta.jsonl"))
    if not meta:
        return
    args = (meta[0].get("config") or {}).get("args") or {}
    if "task1_G" not in args:
        rep.note("rl.v16", "run written before v16 (no task1_G): v16 checks skipped")
        return
    acfg = (meta[0].get("config") or {}).get("algo_cfg") or {}
    rep.ok("rl.adv_norm", acfg.get("grpo_std_norm") is False, "run_meta",
           "algo_cfg.grpo_std_norm %r (spec v16: false)" % acfg.get("grpo_std_norm"))
    train_all = None
    sp_path = splits_path or args.get("splits")
    try:
        sp = json.load(open(sp_path, encoding="utf-8"))
        f = {int(x["fold"]): x for x in sp["folds"]}[int(args["fold"])]
        train_all = set(f.get("train_all", f.get("train", [])))
        want = meta[0].get("splits_sha256")
        if want:
            rep.ok("rl.task1_refill", sha256_file(sp_path) == want, "splits",
                   "split file %r is not the one the run used (sha differs)" % sp_path)
    except Exception as e:                       # never skipped silently
        rep.ok("rl.task1_refill", False, "splits", "cannot read the split file %r: %r" % (sp_path, e))
    upd = {u["update"]: u for u in _jl(os.path.join(rl_dir, "updates.jsonl"))}
    # ---- Task 1 groups: the update's own record of its groups decides which rows count (an aborted attempt may
    # have left rows of other groups; a conversation with no decision position leaves no row at all)
    latest = {}
    for r in _jl(os.path.join(rl_dir, "rollouts_task1.jsonl")):
        latest[(r["update"], r["conversation_id"], r["t"])] = r
    for u, row in sorted(upd.items()):
        th = (row.get("train_aggregate") or {}).get("task1_train") or {}
        st = row.get("learner_stats") or {}
        w = "task1 u%s" % u
        if int(args.get("task1_convs", 0)) <= 0:
            continue
        ok_rec = all(k in th for k in ("groups", "base_convs", "refill_convs"))
        rep.ok("rl.task1_refill", ok_rec, w, "task1_train has no group record (groups / base_convs / refill_convs)")
        if not ok_rec:
            continue
        base, refill = set(th["base_convs"]), set(th["refill_convs"])
        if train_all is not None:
            rep.ok("rl.task1_refill", (base | refill) <= train_all, w, "Task 1 conversation outside train_all")
            rep.ok("rl.task1_G", len(base) == min(int(args["task1_convs"]), len(train_all)), w,
                   "%d base conversations, task1_convs %s" % (len(base), args["task1_convs"]))
        rep.ok("rl.task1_refill", not (base & refill), w, "a refill conversation is also a base conversation")
        rep.ok("rl.task1_refill", len(refill) <= int(args["task1_convs"]), w, "%d refill conversations > cap" % len(refill))
        rows = []
        for cid, t, is_ref in th["groups"]:
            r = latest.get((u, cid, t))
            rep.ok("rl.task1_refill", r is not None, w, "group %s t%s has no row" % (str(cid)[:10], t))
            if r is None:
                continue
            rows.append(r)
            rep.ok("rl.task1_refill", bool(r.get("refill")) == bool(is_ref) and (cid in (refill if is_ref else base)), w,
                   "group %s t%s: refill flag / conversation list disagree" % (str(cid)[:10], t))
            rep.ok("rl.task1_G", len(r.get("samples") or []) == int(args["task1_G"]), w,
                   "%d samples, task1_G %s" % (len(r.get("samples") or []), args["task1_G"]))
        n_base = sum(1 for r in rows if not r.get("refill"))
        mstd = float(acfg.get("min_group_std", 1e-8))

        def _inf(r):
            rs_ = [x["reward"] for x in r.get("samples") or []]
            if len(rs_) < 2:
                return False
            m_ = sum(rs_) / len(rs_)
            return (sum((x - m_) ** 2 for x in rs_) / len(rs_)) ** 0.5 > mstd
        inf_base = sum(1 for r in rows if not r.get("refill") and _inf(r))
        inf_all = sum(1 for r in rows if _inf(r))
        stop_ = th.get("refill_stop")
        rep.ok("rl.task1_refill", th.get("n_informative_groups") == inf_all, w,
               "n_informative_groups %r != recounted %d" % (th.get("n_informative_groups"), inf_all))
        if inf_base >= n_base:
            rep.ok("rl.task1_refill", not refill and stop_ == "none_needed", w,
                   "refill drawn although the base groups had no deficit (stop %r)" % stop_)
        else:
            # a drawn refill conversation may leave no row (one message, or both positions after a capped
            # message): the trainer's own count n_refill_convs includes it, the refill_convs list does not
            n_ref = int(th.get("n_refill_convs") or 0)
            rep.ok("rl.task1_refill", n_ref >= len(refill), w,
                   "n_refill_convs %d < %d refill conversations with rows" % (n_ref, len(refill)))
            want_ok = {"filled": inf_all >= n_base,
                       "cap": n_ref == int(args["task1_convs"]) and inf_all < n_base,
                       "pool_empty": (train_all is None or len(base) + n_ref == len(train_all)) and inf_all < n_base}
            rep.ok("rl.task1_refill", bool(want_ok.get(stop_)), w,
                   "refill_stop %r inconsistent (informative %d / %d base groups, %d refill convs)"
                   % (stop_, inf_all, n_base, len(refill)))
        rep.ok("rl.task1_refill", th.get("n_base_groups") == n_base and th.get("n_refill_groups") == len(rows) - n_base, w,
               "task1_train counts %r / %r != groups %d / %d" % (th.get("n_base_groups"), th.get("n_refill_groups"),
                                                                 n_base, len(rows) - n_base))
        # the stop supervision: exactly one example per base group that has one (none from the refill groups)
        ag = row.get("train_aggregate") or {}
        want_aux = sum(1 for r in rows if not r.get("refill") and (r.get("samples") or [{}])[0].get("aux") is not None)
        if (ag.get("aux_weight") or 0) > 0:
            rep.ok("rl.aux_floor", (st.get("aux_n") or 0) == want_aux, w,
                   "stop supervision used %r examples, the base groups provide %d" % (st.get("aux_n"), want_aux))
    # ---- per update: advantage magnitude logged, aux floor, anneal only after the trigger, w_dist floor
    ivs = [m_["intervention"] for m_ in meta if m_.get("intervention")]
    iv = ivs[0] if ivs else None
    ctl = ((meta[0].get("config") or {}).get("controller") or {}).get("llm4") or {}
    wlo0 = ((ctl.get("bounds") or {}).get("w_dist") or [None])[0]
    if wlo0 is not None:
        rep.ok("rl.w_dist_floor", float(wlo0) == 1.0 or bool(args.get("ablation")), "run_meta",
               "controller w_dist lower bound %r (spec v16: 1.0)" % wlo0)
    vsum = sorted((v for v in _jl(os.path.join(rl_dir, "validation.jsonl")) if v.get("kind") == "summary"),
                  key=lambda v: v["update"])
    # D2 recomputed from the summaries' own bal_p (never trusting the logged streak)
    margin = float(args.get("t1_trigger_margin", 0.10))
    base_v = next((v for v in vsum if v.get("task1") and v["task1"].get("bal_p") is not None), None)
    t_at, streak = None, 0
    for v in vsum:
        if base_v is None or v["update"] <= base_v["update"]:
            continue
        if not v.get("task1") or v["task1"].get("bal_p") is None:
            streak = 0                           # the trainer reads a validation without Task 1 as "not met"
            continue
        met = v["task1"]["bal_p"] >= base_v["task1"]["bal_p"] + margin
        streak = streak + 1 if met else 0
        d2 = v.get("d2") or {}
        w = "validation u%s" % v["update"]
        rep.ok("rl.d2_trigger", d2.get("met") == met and d2.get("streak") == streak, w,
               "logged d2 %r, recomputed met %r streak %d" % ({k: d2.get(k) for k in ("met", "streak")}, met, streak))
        if t_at is None and streak >= 2:
            t_at = v["update"]
            rep.ok("rl.d2_trigger", d2.get("triggered_at") == t_at, w, "D2 should trigger here (streak 2), logged %r"
                   % d2.get("triggered_at"))
    for v in vsum:
        ta = (v.get("d2") or {}).get("triggered_at")
        if ta is not None:
            rep.ok("rl.d2_trigger", ta == t_at, "validation u%s" % v["update"],
                   "logged trigger at %r, recomputed %r" % (ta, t_at))
    for u, row in sorted(upd.items()):
        ag, cfg, st = row.get("train_aggregate") or {}, row.get("cfg_used") or {}, row.get("learner_stats") or {}
        w = "update %s" % u
        if row.get("n_samples"):
            rep.ok("rl.adv_norm", st.get("adv_abs_mean") is not None, w, "learner_stats.adv_abs_mean missing")
        if ag.get("aux_floor") is not None and "stop_sup_floor" in args:
            rep.ok("rl.aux_floor", abs(float(ag["aux_floor"]) - float(args["stop_sup_floor"])) < 1e-12, w,
                   "logged floor %r != the run's --stop-sup-floor %r" % (ag["aux_floor"], args["stop_sup_floor"]))
        if ag.get("aux_floor") is not None and ag.get("aux_weight") is not None:
            rep.ok("rl.aux_floor", ag["aux_weight"] >= ag["aux_floor"] - 1e-12, w,
                   "aux weight %r below the floor %r" % (ag["aux_weight"], ag["aux_floor"]))
            want = max(float(ag["aux_floor"]), float(ag.get("aux_annealed", ag["aux_weight"])))
            rep.ok("rl.aux_floor", abs(ag["aux_weight"] - want) < 1e-12, w,
                   "aux weight %r != max(floor %r, annealed %r)" % (ag["aux_weight"], ag["aux_floor"], ag.get("aux_annealed")))
            ann = float(ag.get("aux_annealed", ag["aux_weight"]))
            if ann < float(cfg.get("w_aux", 0.0)) - 1e-12:
                rep.ok("rl.d2_trigger", t_at is not None and u > t_at, w,
                       "stop supervision annealed (%r < w_aux %r) before the D2 trigger" % (ann, cfg.get("w_aux")))
        lo = wlo0
        if iv and u >= iv.get("at_update", 10 ** 9) and "w_dist" in (iv.get("controller_bounds") or {}):
            lo = iv["controller_bounds"]["w_dist"][0]            # a user-approved intervention changed the bound
        if lo is not None and "w_dist" in cfg:
            rep.ok("rl.w_dist_floor", float(cfg["w_dist"]) >= float(lo) - 1e-12, w,
                   "w_dist %r below the controller bound %r" % (cfg["w_dist"], lo))
    # ---- continuous Task 1 metric recomputed from the logged end probabilities (validation and re-selection)
    real_vllm = args.get("planner_backend") == "vllm" and not args.get("dry_run")
    for fname in ("validation.jsonl", "reselect.jsonl"):
        _check_t1prob_file(rl_dir, fname, rep, real_vllm)


# ------------------------------------------------------------------ SPEC v17 (user 2026-09-30)
V17_SPEC = {"kl": 0.01, "lr": 1e-5, "controller": "fixed", "aux_weight": 0.5, "updates": 5, "task1_G": 4,
            "task1_convs": 8, "sft_lr": 5e-5, "sft_epochs_max": 3, "sft_samples_per_point": 2,
            "length_drift_margin": 1.0, "task1_reward": "brier", "task1_positions": "all", "val_every": 1,
            "stop_credit": 1}
V17_VAL_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def _v17_nll(points):
    xs = [p for p in points if p.get("valid", True)]
    if not xs:
        return None
    return sum(-math.log(min(1 - 1e-6, max(1e-6, float(p["p_end"]) if p["real_final"] else 1 - float(p["p_end"]))))
               for p in xs) / len(xs)


def _pstd(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _unscored_ok(p):
    """The unscored-point rule (v16 item 7): an invalid plan reads as 0, a valid decision without a located value as its
    greedy decision."""
    if p.get("valid", True):
        return 0.0 <= float(p["p_end"]) <= 1.0
    return float(p["p_end"]) == (1.0 if (p.get("decision_valid") and p.get("greedy_end")) else 0.0)


def check_v17(rl_dir, rep, splits_path=None):
    """SPEC v17 §9 (user 2026-09-30), recomputed from the run's logs: settings, SFT data and choice, the ref adapter,
    the Task 1 groups (Brier rewards, drops, advantages and where they act), the stop supervision, the optimizer steps,
    the length drift and the stop reason, the validation schedule, the test update set, the Ditto architecture."""
    import task1_stop as T1
    meta = _jl(os.path.join(rl_dir, "run_meta.jsonl"))
    if not rep.ok("rl.v17_settings", bool(meta), rl_dir, "run_meta.jsonl missing"):
        return
    cfg0 = meta[0].get("config") or {}
    args, acfg = cfg0.get("args") or {}, cfg0.get("algo_cfg") or {}
    abl = bool(args.get("ablation"))
    dry = bool(args.get("dry_run"))
    real_vllm = args.get("planner_backend") == "vllm" and not dry
    # ---- settings (§8)
    rep.ok("rl.v17_settings", all((m.get("config") or {}).get("spec_version") == "v17" for m in meta), "run_meta",
           "a run_meta row of another spec version")
    for k, want in V17_SPEC.items():
        rep.ok("rl.v17_settings", args.get(k) == want or abl, "run_meta", "%s = %r (spec v17: %r)" % (k, args.get(k), want))
    rep.ok("rl.v17_settings", sorted(args.get("val_seeds") or []) == V17_VAL_SEEDS or abl, "run_meta",
           "val_seeds %r (spec v17: 0..7)" % args.get("val_seeds"))
    rep.ok("rl.v17_settings", (acfg.get("epochs"), acfg.get("minibatches")) == (2, 4) or abl, "run_meta",
           "algo epochs x minibatches %r x %r (spec v17: 2 x 4)" % (acfg.get("epochs"), acfg.get("minibatches")))
    rep.ok("rl.adv_norm", acfg.get("grpo_std_norm") is False, "run_meta", "grpo_std_norm %r" % acfg.get("grpo_std_norm"))
    rep.ok("rl.v17_settings", args.get("init_adapter") is None and all(m.get("init_adapter_sha256") is None for m in meta),
           "run_meta", "an init adapter was used")
    # ---- the split file
    sp_path = splits_path or args.get("splits")
    try:
        f = {int(x["fold"]): x for x in json.load(open(sp_path, encoding="utf-8"))["folds"]}[int(args["fold"])]
        rep.ok("rl.v17_splits", sha256_file(sp_path) == meta[0].get("splits_sha256"), "splits",
               "split file %r is not the one the run used (sha differs)" % sp_path)
    except Exception as e:                       # never skipped silently
        rep.ok("rl.v17_splits", False, "splits", "cannot read the split file %r: %r" % (sp_path, e))
        return
    train_all, forb = set(f.get("train_all", f["train"])), set(f.get("forbidden_for_training", []))
    val, val_all = set(f["validation"]), set(f.get("validation_all", f["validation"]))
    rep.ok("rl.v17_splits", val <= val_all <= forb and not val_all & (train_all | set(f["train"])), "splits",
           "validation_all must contain validation, lie in forbidden and be disjoint from train / train_all")
    # ---- Ditto architecture (S12): the env description of a real run
    env = cfg0.get("env")
    rep._c("rl.v17_ditto")
    if env is None:
        rep.note("rl.v17_ditto", "no env description (dry run)")
        rep.ok("rl.v17_ditto", dry, "run_meta", "a real run without config.env")
    else:
        ae, v2 = env.get("arm_env") or {}, env.get("v2fix") or {}
        rep.ok("rl.v17_ditto", env.get("arm") == "pend", "env", "arm %r" % env.get("arm"))
        rep.ok("rl.v17_ditto", "Ditto-8B" in str(env.get("speaker_path")), "env", "speaker %r" % env.get("speaker_path"))
        rep.ok("rl.v17_ditto", env.get("speaker_endconv_is_none") is True, "env", "the Speaker has an end token")
        rep.ok("rl.v17_ditto", any(c.endswith("DittoSpeaker") for c in env.get("speaker_class") or []), "env",
               "speaker class %r" % env.get("speaker_class"))
        for k in ("SEPSIM_ENDMASK_RETRY", "SEPSIM_ENDGATE", "SEPSIM_KEEPEND", "SEPSIM_ENDSCORE"):
            rep.ok("rl.v17_ditto", ae.get(k) == "0", "env", "%s = %r (a UserLM end-token mechanism)" % (k, ae.get(k)))
        rep.ok("rl.v17_ditto", v2.get("SEPSIM_END_PROBE") == "0", "env", "SEPSIM_END_PROBE %r" % v2.get("SEPSIM_END_PROBE"))
        rep.ok("rl.v17_ditto", "e1r_cf19400" in str(env.get("tree")), "env", "tree %r" % env.get("tree"))
    # ---- SFT data (§1.1)
    ex = _jl(os.path.join(rl_dir, "sft_examples.jsonl"))
    mp = os.path.join(rl_dir, "sft_examples_meta.json")
    em = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else None
    spp = int(args.get("sft_samples_per_point", 2))
    if rep.ok("rl.sft_data", em is not None and bool(ex), rl_dir, "sft_examples.jsonl / sft_examples_meta.json missing"):
        rep.ok("rl.sft_data", sha256_file(os.path.join(rl_dir, "sft_examples.jsonl")) == em["examples_sha256"], mp,
               "sft_examples.jsonl changed after its meta was written")
        rep.ok("rl.sft_data", sorted(em.get("conversations") or []) == sorted(train_all), mp,
               "the SFT conversations are not exactly train_all")
        n_of, pts = {}, {}
        for cid, t, n, status, n_ex in em["points"]:
            w = "sft %s t%s" % (str(cid)[:10], t)
            rep.ok("rl.sft_data", cid in train_all and cid not in forb, w, "SFT point outside train_all / in forbidden")
            rep.ok("rl.sft_data", n_of.setdefault(cid, n) == n, w, "two lengths for one conversation")
            pts[(cid, t)] = (status, n_ex)
        for cid, n in n_of.items():
            rep.ok("rl.sft_data", {t for (c, t) in pts if c == cid} == set(range(2, n + 1)), "sft %s" % str(cid)[:10],
                   "decision points are not t = 2..%d" % n)
        per = {}
        for e in ex:
            w = "sft %s t%s k%s" % (str(e.get("conversation_id"))[:10], e.get("t"), e.get("k"))
            key = (e["conversation_id"], e["t"])
            per[key] = per.get(key, 0) + 1
            st_ = pts.get(key, ("missing", 0))[0]
            rep.ok("rl.sft_data", e["conversation_id"] in train_all and e["conversation_id"] not in forb and st_ == "ok"
                   and 2 <= e["t"] <= e["n"] == n_of.get(e["conversation_id"]) and e["real_final"] == (e["t"] == e["n"]),
                   w, "example outside train_all, off a decision point or with a wrong label")
            rep.ok("rl.sft_data", (e["kind"] == "greedy") == (e["k"] == 0) and 0 <= e["k"] <= spp
                   and e["target_ids"] == (e["target_true"] if e["real_final"] else e["target_false"]), w,
                   "kind / plan index / target inconsistent with the human's decision")
        for key, (st_, n_ex) in pts.items():
            rep.ok("rl.sft_data", per.get(key, 0) == n_ex and n_ex <= 1 + spp, "sft %s t%s" % (str(key[0])[:10], key[1]),
                   "%d examples logged, the point record says %d" % (per.get(key, 0), n_ex))
        c = em.get("counts") or {}
        rep.ok("rl.sft_data", c.get("n_examples") == len(ex) and c.get("n_points") == sum(1 for v in pts.values() if v[0] == "ok"),
               mp, "counts %r disagree with the rows" % {k: c.get(k) for k in ("n_examples", "n_points")})
        base = _jl(os.path.join(rl_dir, "base_pend_train.jsonl"))
        rep.ok("rl.sft_data", sha256_file(os.path.join(rl_dir, "base_pend_train.jsonl")) == em.get("base_pend_sha256")
               if os.path.exists(os.path.join(rl_dir, "base_pend_train.jsonl")) else False, rl_dir,
               "base_pend_train.jsonl missing or changed")
        rep.ok("rl.sft_data", sorted((b["conversation_id"], b["t"]) for b in base)
               == sorted(k for k, v in pts.items() if v[0] == "ok") and all(_unscored_ok(b) for b in base), rl_dir,
               "base_pend_train.jsonl: not one scored point per SFT decision point, or a p_end breaks the rules")
        if real_vllm:
            want = adapter_name(rl_dir, "sft_e0")
            rep.ok("rl.sft_data", all(e.get("gen_adapter") == want for e in ex) and all(b.get("gen_adapter") == want for b in base),
                   rl_dir, "SFT plans not generated by the start policy's adapter %r" % want)
    # ---- SFT choice (§1.3): recomputed from sft.jsonl
    srows = _jl(os.path.join(rl_dir, "sft.jsonl"))
    sel = [r for r in srows if r.get("kind") == "selection"]
    st0p = os.path.join(rl_dir, "ckpt", "u00000", "state.json")
    st0 = json.load(open(st0p, encoding="utf-8")) if os.path.exists(st0p) else None
    if rep.ok("rl.sft_choice", bool(sel) and st0 is not None, rl_dir, "no SFT selection row / no ckpt u00000"):
        sel = sel[-1]
        eps_ = sorted((r for r in srows if r.get("kind") == "epoch" and r.get("attempt") == sel["attempt"]),
                      key=lambda r: r["epoch"])
        rep.ok("rl.sft_choice", [r["epoch"] for r in eps_] == list(range(0, int(args.get("sft_epochs_max", 3)) + 1)),
               "sft.jsonl", "candidate epochs %r" % [r["epoch"] for r in eps_])
        for r in eps_:
            w = "sft epoch %d" % r["epoch"]
            pts_ = [p for c in sorted(r.get("val_end_probs") or {}) for p in r["val_end_probs"][c]]
            rep.ok("rl.sft_choice", sorted(r.get("val_ids") or []) == sorted(val_all)
                   and sorted(r.get("val_end_probs") or {}) == sorted(val_all), w, "the probes are not on validation_all")
            n_exp = sum(n - 1 for n in (r.get("val_n_turns") or {}).values())
            rep.ok("rl.sft_choice", len(pts_) == n_exp and all(_unscored_ok(p) for p in pts_), w,
                   "%d probe points, expected sum(n - 1) = %d, or a p_end breaks the rules" % (len(pts_), n_exp))
            if pts_:
                nll = _v17_nll(pts_)
                ninv = sum(1 for p in pts_ if not p.get("valid", True))
                rep.ok("rl.sft_choice", (nll is None and r["val"]["nll"] is None) or (nll is not None and r["val"]["nll"] is not None
                       and abs(nll - r["val"]["nll"]) < 1e-9) and ninv == r["val"]["n_invalid"], w,
                       "logged nll %r / n_invalid %r, recomputed %r / %d" % (r["val"]["nll"], r["val"]["n_invalid"], nll, ninv))
            cp = os.path.join(rl_dir, "ckpt", "sft_e%d" % r["epoch"], "state.json")
            rep.ok("rl.sft_choice", os.path.exists(cp) and json.load(open(cp, encoding="utf-8"))["policy_sha"] == r["policy_sha"],
                   w, "candidate checkpoint missing or of another policy")
            if real_vllm:
                want = adapter_name(rl_dir, "sft_e%d" % r["epoch"])
                rep.ok("rl.sft_choice", all(p.get("gen_adapter") == want for p in pts_), w,
                       "probes not generated by the candidate's adapter %r" % want)
        if eps_:
            key = lambda r: (float("inf") if r["val"]["nll"] is None else r["val"]["nll"], r["val"]["n_invalid"], r["epoch"])
            k = min(eps_, key=key)["epoch"]
            chosen = next(r for r in eps_ if r["epoch"] == k)
            rep.ok("rl.sft_choice", sel["chosen_epoch"] == k, "sft.jsonl",
                   "chosen epoch %r, recomputed %r (lowest nll, then n_invalid, then earlier)" % (sel["chosen_epoch"], k))
            rep.ok("rl.sft_choice", st0["policy_sha"] == chosen["policy_sha"] == sel["policy_sha"]
                   and (st0.get("sft") or {}).get("chosen_epoch") == k, st0p, "u0 is not the chosen SFT candidate")
            a0 = os.path.join(rl_dir, "ckpt", "u00000", "adapter", "adapter_model.safetensors")
            ak = os.path.join(rl_dir, "ckpt", "sft_e%d" % k, "adapter", "adapter_model.safetensors")
            if os.path.exists(a0) or os.path.exists(ak):
                rep.ok("rl.sft_choice", os.path.exists(a0) and os.path.exists(ak) and sha256_file(a0) == sha256_file(ak),
                       st0p, "u0's adapter file differs from the chosen candidate's")
    # ---- the ref adapter (§2, B4)
    for d in sorted(glob.glob(os.path.join(rl_dir, "ckpt", "*", "adapter"))):
        rep.ok("rl.ref_adapter", not os.path.exists(os.path.join(d, "ref")), d, "a checkpoint holds the ref adapter")
    rep._c("rl.ref_adapter")
    if st0 is not None:
        sft_rows = [m for m in meta if m.get("kind") == "sft"]
        rep.ok("rl.ref_adapter", len(sft_rows) >= 1 and all(m.get("ref_policy_sha") == st0["policy_sha"] for m in sft_rows),
               "run_meta", "no sft row naming the ref = u0 sha")
        for m in meta:
            if m.get("ref_policy_sha") is not None:
                rep.ok("rl.ref_adapter", m["ref_policy_sha"] == st0["policy_sha"], "run_meta %s" % m.get("kind"),
                       "ref %s is not u0" % str(m["ref_policy_sha"])[:12])
    # ---- per update: Task 1 groups, advantages, aux, steps, settings, drift (§3)
    upd = {u["update"]: u for u in _jl(os.path.join(rl_dir, "updates.jsonl"))}
    latest = {}
    for r in _jl(os.path.join(rl_dir, "rollouts_task1.jsonl")):
        latest[(r["update"], r["conversation_id"], r["t"])] = r
    t2 = {}
    for r in _jl(os.path.join(rl_dir, "rollouts.jsonl")):
        t2[(r["update"], r["slot"], r["replicate"])] = r
    G1, n_convs = int(args.get("task1_G", 4)), int(args.get("task1_convs", 8))
    mstd, epochs, mbs = float(acfg.get("min_group_std", 1e-8)), int(acfg.get("epochs", 2)), int(acfg.get("minibatches", 4))
    t_max = int((cfg0.get("selection_cfg") or {}).get("t_max", 10))
    drift = {}
    for u, row in sorted(upd.items()):
        w = "update %s" % u
        ts, st, ag, cfg = row.get("task1_stats"), row.get("learner_stats") or {}, row.get("train_aggregate") or {}, row.get("cfg_used") or {}
        rep.ok("rl.v17_update", abs(float(cfg.get("kl_coef", -1)) - float(args.get("kl", 0.01))) < 1e-15
               and abs(float(cfg.get("lr", -1)) - float(args.get("lr", 1e-5))) < 1e-15, w,
               "kl_coef %r / lr %r differ from the run's" % (cfg.get("kl_coef"), cfg.get("lr")))
        if args.get("controller") == "fixed":
            rep.ok("rl.v17_update", row.get("controller") == "fixed" and cfg == cfg0.get("cfg0"), w,
                   "the fixed controller's weights moved")
        rep.ok("rl.v17_update", float(ag.get("aux_weight", -1)) == float(args.get("aux_weight", 0.5)), w,
               "aux weight %r, the run's --aux-weight %r" % (ag.get("aux_weight"), args.get("aux_weight")))
        # optimizer steps (§3.4, S3)
        n_s = int(row.get("n_samples") or 0)
        want = epochs * min(mbs, n_s) if n_s else (1 if st.get("aux_n") else 0)
        rep.ok("rl.v17_steps", st.get("optimizer_steps") == want, w,
               "%r optimizer steps, expected %d (epochs %d x min(%d, %d samples))" % (st.get("optimizer_steps"), want, epochs, mbs, n_s))
        if n_s == 0 and st.get("aux_n"):
            rep.ok("rl.v17_steps", st.get("aux_only") is True, w, "a supervision-only update is not flagged")
        if n_s:
            rep.ok("rl.adv_norm", st.get("adv_abs_mean") is not None and isinstance(st.get("adv_abs_mean_by_source"), dict),
                   w, "adv_abs_mean (by source) missing")
        # drift (§3.5): recomputed from the Task 2 rollouts (clean episodes of groups with >= 2 clean)
        slots = {}
        for (uu, slot, g), r in t2.items():
            if uu == u and r["episode"].get("clean"):
                slots.setdefault(slot, []).append(r["episode"])
        eps_u = [e for es in slots.values() if len(es) >= 2 for e in es]
        if rep.ok("rl.v17_drift", bool(eps_u), w, "no clean group episode to recompute the drift"):
            dv = sum(int(e["emitted_user_turns"]) - min(int(e["human_turns"]), t_max) for e in eps_u) / len(eps_u)
            drift[u] = dv
            rep.ok("rl.v17_drift", ag.get("drift_stat") is not None and abs(ag["drift_stat"] - dv) < 1e-9, w,
                   "logged drift %r, recomputed %r" % (ag.get("drift_stat"), dv))
        if n_convs <= 0:
            continue
        if not rep.ok("rl.v17_task1", isinstance(ts, dict) and all(k in ts for k in ("convs", "groups", "skipped_capped",
                                                                                      "advantages", "aux_points")),
                      w, "task1_stats record missing"):
            continue
        convs = set(ts["convs"])
        rep.ok("rl.v17_task1", convs <= train_all and not convs & forb and len(convs) == min(n_convs, len(train_all)), w,
               "Task 1 conversations %d, outside train_all or not min(task1_convs, |train_all|)" % len(convs))
        rows = {}
        for cid, t in ts["groups"]:
            r = latest.get((u, cid, t))
            if rep.ok("rl.v17_task1", r is not None and cid in convs, w, "group %s t%s has no row" % (str(cid)[:10], t)):
                rows[(cid, t)] = r
        skipped = {(c, t) for c, t in ts["skipped_capped"]}
        for cid in convs:
            ns = {r["n_real"] for (c, _), r in rows.items() if c == cid}
            if len(ns) > 1:
                rep.ok("rl.v17_task1", False, w, "two lengths for %s" % cid)
                continue
            if ns:
                n = ns.pop()
                have = {t for (c, t) in rows if c == cid} | {t for (c, t) in skipped if c == cid}
                rep.ok("rl.v17_task1", have == set(range(2, n + 1)) and not ({t for (c, t) in rows if c == cid}
                                                                             & {t for (c, t) in skipped if c == cid}),
                       w, "%s: decision points %s are not t = 2..%d" % (str(cid)[:10], sorted(have), n))
        adv_log = {(c, t): (kind, used) for c, t, kind, used in ts["advantages"]}
        cnt = {"lt2": 0, "all_invalid": 0, "zero_std": 0}
        want_aux = []
        for (cid, t), r in sorted(rows.items()):
            wg = "%s %s t%s" % (w, str(cid)[:10], t)
            smp = r.get("samples") or []
            rep.ok("rl.v17_task1", len(smp) == G1 and r["real_final"] == (t == r["n_real"]) and t >= 2, wg,
                   "%d samples (task1_G %d) or a wrong label" % (len(smp), G1))
            y = 1.0 if r["real_final"] else 0.0
            for x in smp:
                s_ = x.get("status")
                g = x.get("planner_gen") or {}
                if s_ == "valid":
                    ok = (x.get("decision_valid") and x.get("mask_ok") and not x.get("dropped")
                          and x.get("p_end") is not None and 0.0 <= x["p_end"] <= 1.0
                          and abs(x["reward"] - (1.0 - (x["p_end"] - y) ** 2)) < 1e-12
                          and g.get("stop_mask") is not None and 1 in g["stop_mask"] and g["stop_mask"].index(1) > 0)
                elif s_ == "invalid":
                    ok = not x.get("decision_valid") and x.get("reward") == 0.0 and x.get("p_end") is None and not x.get("dropped")
                elif s_ == "dropped":
                    ok = x.get("decision_valid") and not x.get("mask_ok") and x.get("dropped") and x.get("reward") is None
                else:
                    ok = False
                rep.ok("rl.v17_task1_reward", ok, wg, "sample r%s status %r: reward / P_end / flags break the Brier rule"
                       % (x.get("replicate"), s_))
            kept = [x for x in smp if not x.get("dropped")]
            if any(x.get("status") == "valid" for x in smp):
                want_aux.append([cid, t])
            if len(kept) < 2:
                kind, advs = "lt2", None
            elif all(x.get("status") == "invalid" for x in kept):
                kind, advs = "all_invalid", None
            else:
                rs = [x["reward"] for x in kept]
                m_ = sum(rs) / len(rs)
                kind, advs = ("zero_std", None) if _pstd(rs) <= mstd else ("used", [x_ - m_ for x_ in rs])
            if kind != "used":
                cnt[kind] += 1
            lk, used = adv_log.get((cid, t), (None, None))
            if rep.ok("rl.v17_advantage", lk == kind, wg, "logged group %r, recomputed %r" % (lk, kind)) and kind == "used":
                want_u = [[x["replicate"], x["status"], a_, "prefix" if x["status"] == "valid" else "all"]
                          for x, a_ in zip(kept, advs)]
                rep.ok("rl.v17_advantage", len(used) == len(want_u) and all(
                    a[0] == b[0] and a[1] == b[1] and abs(a[2] - b[2]) < 1e-12 and a[3] == b[3] for a, b in zip(used, want_u)),
                    wg, "advantages / where they act differ from R - mean(R) over the kept samples (prefix for a "
                        "valid sample, every token for an invalid one; dropped samples never)")
        rep.ok("rl.v17_advantage", set(adv_log) == set(rows), w, "advantage records != groups")
        rep.ok("rl.v17_task1", ts.get("n_groups_lt2") == cnt["lt2"] and ts.get("n_groups_all_invalid") == cnt["all_invalid"]
               and ts.get("groups_skipped_zero_std") == cnt["zero_std"], w,
               "skip counts %r != recomputed %r" % ({k: ts.get(k) for k in ("n_groups_lt2", "n_groups_all_invalid",
                                                                           "groups_skipped_zero_std")}, cnt))
        # the stop supervision (§3.3): one example per decision point with a valid sample, weight --aux-weight
        if float(args.get("aux_weight", 0.5)) > 0:
            rep.ok("rl.v17_aux", sorted(ts["aux_points"]) == sorted(want_aux) and st.get("aux_n", 0) == len(want_aux)
                   and len({tuple(p) for p in ts["aux_points"]}) == len(ts["aux_points"]), w,
                   "aux examples %r at %d points, expected one per point with a valid sample (%d)"
                   % (st.get("aux_n"), len(ts["aux_points"]), len(want_aux)))
    # ---- the stop rule (§3.5, S5, S13), recomputed from the drift values
    margin, n_up = float(args.get("length_drift_margin", 1.0)), int(args.get("updates", 5))
    fu, reason = None, None
    for u in sorted(drift):
        if u >= 2 and drift.get(u - 1) is not None and drift[u] < -margin and drift[u - 1] < -margin:
            fu, reason = u, "length_drift"
            break
        if u >= n_up:
            fu, reason = u, "max_updates"
            break
    fp = os.path.join(rl_dir, "final.json")
    fin = json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else None
    if fin is not None or fu is not None:
        rep.ok("rl.v17_stop", fin is not None and fin.get("final_update") == fu and fin.get("stop_reason") == reason, fp,
               "final.json %r, recomputed final u%s (%s)" % ({k: (fin or {}).get(k) for k in ("final_update", "stop_reason")},
                                                            fu, reason))
        rep.ok("rl.v17_stop", fu is None or max(upd) == fu, fp, "updates after the stop (u%s)" % max(upd or [0]))
        stops = [m for m in meta if m.get("kind") == "stop"]
        rep.ok("rl.v17_stop", len(stops) == 1 and stops[0].get("final_update") == fu and stops[0].get("stop_reason") == reason,
               "run_meta", "run_meta stop rows %r" % [(m.get("final_update"), m.get("stop_reason")) for m in stops])
        if fin is not None and fu is not None:
            sp_ = os.path.join(rl_dir, "ckpt", "u%05d" % fu, "state.json")
            rep.ok("rl.v17_stop", os.path.exists(sp_) and json.load(open(sp_, encoding="utf-8"))["policy_sha"] == fin.get("policy_sha"),
                   fp, "final.json policy sha is not the checkpoint's")
    rep._c("rl.v17_stop")
    # ---- validation schedule (§4, S6)
    vrows = _jl(os.path.join(rl_dir, "validation.jsonl"))
    summ = [v for v in vrows if v.get("kind") == "summary"]
    seeds_want = sorted(args.get("val_seeds") or V17_VAL_SEEDS)
    final_u = (fin or {}).get("final_update")
    for v in summ:
        u, w = v["update"], "validation u%s%s" % (v["update"], " task2" if v.get("task2") else "")
        rep.ok("rl.v17_validation", sorted(v.get("task1_ids") or []) == sorted(val_all), w, "Task 1 ids != validation_all")
        rows_ = {}
        for r in vrows:
            if r.get("kind") == "task1" and r.get("update") == u and r.get("policy_sha") == v.get("policy_sha"):
                rows_[r["conversation_id"]] = r
        rep.ok("rl.v17_validation", sorted(rows_) == sorted(val_all), w, "Task 1 rows %r != validation_all" % sorted(rows_))
        pts = [p for r in rows_.values() for p in (r.get("end_probs") or [])]
        n_exp = sum(len(r["task1"]["turns"]) - 1 for r in rows_.values())
        if rep.ok("rl.v17_validation", pts and len(pts) == n_exp and all(_unscored_ok(p) for p in pts), w,
                  "%d probe points (expected %d) or a p_end breaking the rules" % (len(pts), n_exp)):
            m = T1.task1_prob_metrics(pts)
            for k in ("bal_p", "nll"):
                a_, b_ = (v.get("task1") or {}).get(k), m[k]
                rep.ok("rl.v17_validation", (a_ is None and b_ is None) or (a_ is not None and b_ is not None and abs(a_ - b_) < 1e-9),
                       w, "summary %s %r != recomputed %r" % (k, a_, b_))
        if real_vllm:
            want_ad = adapter_name(rl_dir, u)
            rep.ok("rl.v17_validation", all(p.get("gen_adapter") == want_ad for p in pts), w,
                   "probes not generated by the policy of update %s" % u)
        if v.get("task2"):
            rep.ok("rl.v17_validation", u == 0 or u == final_u, w, "Task 2 validation at an update that is neither u0 nor final")
            rep.ok("rl.v17_validation", sorted(v.get("val_seeds") or []) == seeds_want and v.get("val_temperature") == 0.7, w,
                   "Task 2 seeds %r / temperature %r" % (v.get("val_seeds"), v.get("val_temperature")))
            ep = {}
            for r in vrows:
                if r.get("kind") == "episode" and r.get("update") == u and r.get("policy_sha") == v.get("policy_sha"):
                    ep[(r["conversation_id"], r["seed"])] = r
            rep.ok("rl.v17_validation", sorted(ep) == sorted((c, s) for c in val for s in seeds_want), w,
                   "Task 2 episodes %d != validation x seeds" % len(ep))
            rep.ok("rl.v17_validation", v.get("n_episodes", 0) + v.get("n_unclean_episodes", 0) == len(val) * len(seeds_want), w,
                   "episode counts")
        else:
            rep.ok("rl.v17_validation", not v.get("val_seeds") and not v.get("turn_stats"), w, "a Task-1-only summary with Task 2")
    t2u = {r.get("update") for r in vrows if r.get("kind") == "episode"}
    rep.ok("rl.v17_validation", t2u <= {0, final_u}, "validation.jsonl",
           "Task 2 episodes at updates %s (only u0 and the final update)" % sorted(x for x in t2u if x not in (0, final_u)))
    have = {(v["update"], bool(v.get("task2"))) for v in summ}
    if st0 is not None:
        rep.ok("rl.v17_validation", (0, True) in have, "validation.jsonl", "u0 (SFT) has no Task 2 validation")
    if fin is not None and fin.get("validated"):
        rep.ok("rl.v17_validation", (final_u, True) in have, "validation.jsonl",
               "the final update u%s has no Task 2 validation (a Task-1-only summary does not count)" % final_u)
    ve = int(args.get("val_every", 1))
    for u in upd:
        if final_u is not None and u == final_u:
            continue
        if ve > 0 and u % ve == 0 and (final_u is None or u < final_u):
            rep.ok("rl.v17_validation", (u, False) in have, "validation.jsonl", "u%d has no Task 1 validation" % u)
    # ---- test (§5): the tested updates are exactly {0, final}
    tm = _jl(os.path.join(rl_dir, "test_meta.jsonl"))
    if tm:
        rep.ok("rl.v17_test", fin is not None and sorted(tm[-1].get("updates") or []) == sorted({0, final_u}), "test_meta",
               "tested updates %r, final.json names u%s" % (tm[-1].get("updates"), final_u))


def _check_t1prob_file(rl_dir, fname, rep, real_vllm):
    import task1_stop as T1
    vrows = _jl(os.path.join(rl_dir, fname))
    vsum = sorted((v for v in vrows if v.get("kind") == "summary"), key=lambda v: v["update"])
    for v in vsum:
        w = "%s u%s" % (fname.split(".")[0], v["update"])
        if fname == "reselect.jsonl" and v.get("selection_score") is not None and v.get("task1"):
            ts = v.get("turn_stats") or {}
            m1 = v.get("selection_task1_metric")
            rep.ok("rl.selection", m1 == "bal_p", w, "a v16 re-selection must select on bal_p, summary says %r" % m1)
            parts = (ts.get("coverage_mean"), ts.get("turn_w1"), v["task1"].get(m1 or "term_f1"))
            if rep.ok("rl.selection", None not in parts, w, "re-selection summary lacks a selection part %r" % (parts,)):
                want = v["w_sel_cov"] * parts[0] - v["w_sel_w1"] * parts[1] + v["w_sel_task1"] * parts[2]
                rep.ok("rl.selection", abs(v["selection_score"] - want) < 1e-9, w,
                       "re-selection score %r != recomputed %r" % (v["selection_score"], want))
        rows = {}
        for r in vrows:
            if r.get("kind") == "task1" and r["update"] == v["update"] and r.get("policy_sha") == v.get("policy_sha"):
                rows[r["conversation_id"]] = r                      # the latest row per conversation
        if not rows or not v.get("task1"):
            continue
        pts = [p_ for r in rows.values() for p_ in (r.get("end_probs") or [])]
        if real_vllm:
            want_ad = adapter_name(rl_dir, v["update"])
            bad = [p_.get("t") for p_ in pts if p_.get("gen_adapter") != want_ad]
            rep.ok("rl.task1_prob", not bad, w, "probes at t %r not generated by the policy of update %s (%r)"
                   % (bad[:5], v["update"], want_ad))
        for p_ in pts:
            if not p_.get("valid", True) and "decision_valid" in p_:
                want = 1.0 if (p_["decision_valid"] and p_.get("greedy_end")) else 0.0
                rep.ok("rl.task1_prob", float(p_["p_end"]) == want, w,
                       "t%s: unscored point p_end %r, expected %r" % (p_.get("t"), p_["p_end"], want))
        ok_rng = all(0.0 <= float(p_["p_end"]) <= 1.0 for p_ in pts)
        rep.ok("rl.task1_prob", ok_rng, w, "an end probability outside [0, 1]")
        n_exp = sum(len(r["task1"]["turns"]) - 1 for r in rows.values())
        rep.ok("rl.task1_prob", len(pts) == n_exp, w, "%d decision points, expected sum(n - 1) = %d" % (len(pts), n_exp))
        if pts and ok_rng:
            m = T1.task1_prob_metrics(pts)
            rep.ok("rl.task1_prob", abs(m["bal_p"] - v["task1"].get("bal_p", float("nan"))) < 1e-9, w,
                   "summary bal_p %r != recomputed %r" % (v["task1"].get("bal_p"), m["bal_p"]))


def check_rl(rl_dir, rollouts, ckpt_pattern, rep, splits_path=None):
    check_intervention(rl_dir, rep)
    if spec_version(rl_dir) == "v17":
        check_v17(rl_dir, rep, splits_path)
    else:
        check_v16(rl_dir, rep, splits_path)
    for r in rollouts:
        for s in r.get("trace") or []:
            w = "%s s%s t%s" % (str(r.get("conversation_id"))[:12], r.get("seed"), s.get("t"))
            g = s.get("planner_gen")
            good = (isinstance(g, dict) and isinstance(g.get("prompt_ids"), list) and g["prompt_ids"]
                    and isinstance(g.get("gen_ids"), list) and g["gen_ids"]
                    and is_num(g.get("temperature")) and g["temperature"] > 0)
            if rep.ok("rl.planner_gen", good, w, "planner_gen missing/empty or temperature <= 0"):
                pt = (s.get("planner_fit") or {}).get("prompt_tokens")
                if pt is not None:
                    rep.ok("rl.planner_gen", len(g["prompt_ids"]) == pt, w,
                           "len(prompt_ids) %d != planner_fit.prompt_tokens %d" % (len(g["prompt_ids"]), pt))
    up_path = os.path.join(rl_dir, "updates.jsonl")
    if not rep.ok("rl.updates_monotone", os.path.exists(up_path), up_path, "missing"):
        return
    ups = load_jsonl(up_path, rep)
    rep.ok("rl.updates_monotone", bool(ups), up_path, "no update rows")
    prev = None
    for i, u in enumerate(ups):
        k = u.get("update")
        ok = isinstance(k, int) and not isinstance(k, bool) and (prev is None or k > prev)
        rep.ok("rl.updates_monotone", ok, "updates.jsonl row %d" % (i + 1), "update %r after %r" % (k, prev))
        if isinstance(k, int):
            prev = k if prev is None else max(prev, k)
        ck = u.get("checkpoint")
        path = (ck if os.path.isabs(ck) else os.path.join(rl_dir, ck)) if ck else \
            os.path.join(rl_dir, ckpt_pattern.format(update=k if isinstance(k, int) else -1))
        rep.ok("rl.checkpoints", os.path.exists(path), "update %r" % k, "checkpoint %s missing" % path)
    def valid_rc(x):
        return isinstance(x, dict) and x and all(is_num(v) for v in x.values())
    def comps(u):
        if "reward_components" in u:
            return u["reward_components"]
        agg = u.get("train_aggregate") or {}
        return agg["components_mean"] if "components_mean" in agg else None
    up_has = [comps(u) is not None for u in ups]
    ro_has = [("reward_components" in r) for r in rollouts]
    for u in ups:
        if comps(u) is not None:
            rep.ok("rl.reward_components", valid_rc(comps(u)), "update %r" % u.get("update"),
                   "invalid reward components %r" % (comps(u),))
    for r in rollouts:
        if "reward_components" in r:
            rep.ok("rl.reward_components", valid_rc(r["reward_components"]), str(r.get("conversation_id"))[:12],
                   "invalid reward_components")
    complete = (ups and all(up_has)) or (rollouts and all(ro_has))
    rep.ok("rl.reward_components", bool(complete), rl_dir,
           "reward_components on %d/%d updates and %d/%d rollouts" % (sum(up_has), len(ups), sum(ro_has), len(rollouts)))
    ro_updates = {r.get("update") for r in rollouts if "update" in r}
    if ro_updates:
        missing = [u.get("update") for u in ups if u.get("update") not in ro_updates]
        if missing:
            rep.warn("rl.updates_monotone", "updates without rollouts: %s" % missing[:5])


# ------------------------------------------------------------------ driver
def verify(episodes, meta_path, splits_path, fold, split, arm=None, training=False, expected_sha=None,
           sepsim_path=None, rl_dir=None, ckpt_pattern="ckpt/u{update:05d}",
           speaker_budget=None, judge_budget=GJ.JUDGE_BUDGET, max_new_warn=0.02):
    rep = Report()
    meta = load_meta(meta_path, rep)
    splits = json.load(open(splits_path, encoding="utf-8"))
    arm = arm or meta.get("arm")
    if not rep.ok("io.arm", arm in ARMS, "args/meta", "arm %r" % arm):
        return rep
    rows = []
    for p in episodes or []:
        rows += load_jsonl(p, rep)
    rollouts = []
    if rl_dir:
        files = sorted(p for p in glob.glob(os.path.join(rl_dir, "rollouts*.jsonl")) + glob.glob(os.path.join(rl_dir, "rollouts", "*.jsonl"))
                       if not os.path.basename(p).startswith("rollouts_task1"))
        t1p = os.path.join(rl_dir, "rollouts_task1.jsonl")
        v17 = spec_version(rl_dir) == "v17"
        if os.path.exists(t1p):
            folds = {int(f["fold"]): f for f in splits["folds"]}
            f = folds.get(fold, {})
            train, forb = set(f.get("train_all", f.get("train", []))), set(f.get("forbidden_for_training", []))
            t1rows = load_jsonl(t1p, rep)
            for r in t1rows:
                w = "task1 %s t%s u%s" % (str(r.get("conversation_id"))[:12], r.get("t"), r.get("update"))
                rep.ok("leak.task1_train_only", r.get("conversation_id") in train and r.get("conversation_id") not in forb,
                       w, "Task 1 training group on a non-train conversation")
                rep.ok("rl.task1_groups", r.get("real_final") == (r.get("t") == r.get("n_real")) and r.get("samples"), w,
                       "real_final must be exactly t == n_real, with samples")
                rep.ok("rl.task1_groups", (r.get("t") or 0) >= 2, w, "Task 1 stop group at turn 1 (cannot end)")
                for x in r.get("samples") or []:
                    g = x.get("planner_gen") or {}
                    valid = x.get("decision_valid", True)
                    if v17:
                        # v17: the Brier reward is recomputed in check_v17 (rl.v17_task1_reward); here only the logged
                        # 0/1 agreement ("correct") must match the label
                        exp, got = float(valid and bool(x.get("ended_planner")) == bool(r.get("real_final"))), x.get("correct")
                    else:
                        exp, got = float(valid and bool(x.get("ended_planner")) == bool(r.get("real_final"))), x.get("reward")
                    rep.ok("rl.task1_groups", bool(g.get("prompt_ids")) and bool(g.get("gen_ids")) and got == exp, w,
                           "sample without generation ids or with a 0/1 agreement that disagrees with the label")
                    if not valid:
                        rep.ok("rl.stop_mask_gated", g.get("stop_mask") is None, w, "stop mask on a non-decision sample")
            rep.note("rl.task1_groups", "%d Task 1 groups" % len(t1rows))
        rep.ok("rl.rollouts_present", bool(files), rl_dir, "no rollouts*.jsonl")
        for p in files:
            rollouts += load_jsonl(p, rep)
        rep.ok("rl.rollouts_present", bool(rollouts), rl_dir, "no rollout rows")
        training = True
    all_rows = rows + rollouts
    rep.ok("io.rows", bool(all_rows), "inputs", "no episode rows")
    rep.note("io.rows", "%d episodes, %d rl rollouts" % (len(rows), len(rollouts)))
    check_meta(meta, arm, fold, rep)
    if expected_sha is None:
        expected_sha = recompute_system_sha(arm, sepsim_path, implicit_profile=bool(meta.get("implicit_profile")))
        if expected_sha:
            rep.note("%s.system_prompt_sha" % arm, "expected sha recomputed from sepsim")
    check_system_sha(meta, arm, expected_sha, rep)
    check_leakage(all_rows, splits, fold, split, training, rep)
    if arm == "pend":
        check_fewshot_leak(all_rows, splits, fold, rep)
        if not training:
            # evaluation: an unclean episode (cut R0 reply, lost ledger verdict, capped emission, compaction)
            # has a wrong coverage/turn count and cannot be silently averaged in -- rerun it
            for r in rows:
                rep.ok("eval.clean", r.get("clean") is True, "%s s%s" % (str(r.get("conversation_id"))[:12], r.get("seed")),
                       "evaluation episode is not clean: %r" % (r.get("episode_counters"),))
    check_structure(all_rows, arm, rep)
    sb = speaker_budget or meta.get("speaker_budget") or F.SPEAKER_BUDGET
    check_truncation(all_rows, arm, meta, sb, judge_budget, max_new_warn, rep)
    {"a2": check_a2, "pend": check_pend, "a0": check_a0}[check_family(arm)](all_rows, rep, arm)
    if rl_dir:
        check_rl(rl_dir, rollouts, ckpt_pattern, rep, splits_path)
        check_vllm_generation(rl_dir, rollouts, meta, rep)
        check_rl_selection(rl_dir, splits, fold, rep)
        vp = os.path.join(rl_dir, "validation.jsonl")
        if os.path.exists(vp):
            # the episodes that drive best.json get the same structure / truncation / pend / few-shot checks
            vrows = load_jsonl(vp, rep)
            if vrows:
                check_structure(vrows, arm, rep)
                check_truncation(vrows, arm, meta, sb, judge_budget, max_new_warn, rep)
                {"a2": check_a2, "pend": check_pend, "a0": check_a0}[check_family(arm)](vrows, rep, arm)
                if arm == "pend":
                    check_fewshot_leak(vrows, splits, fold, rep)
            check_selection(vp, rl_dir, rep)
        rp = os.path.join(rl_dir, "reselect.jsonl")
        if os.path.exists(rp):
            rrows = load_jsonl(rp, rep)
            if rrows:
                check_structure(rrows, arm, rep)
                check_truncation(rrows, arm, meta, sb, judge_budget, max_new_warn, rep)
                {"a2": check_a2, "pend": check_pend, "a0": check_a0}[check_family(arm)](rrows, rep, arm)
                if arm == "pend":
                    check_fewshot_leak(rrows, splits, fold, rep)
    if arm == "pend":
        check_endpoints(meta, rep)
        vp = os.path.join(rl_dir, "validation.jsonl") if rl_dir else None
        vr = [r for r in (load_jsonl(vp, rep) if vp and os.path.exists(vp) else [])]
        rp = os.path.join(rl_dir, "reselect.jsonl") if rl_dir else None
        rr = [r for r in (load_jsonl(rp, rep) if rp and os.path.exists(rp) else [])]
        check_r0_attribution(all_rows + vr + rr, rep)
        n_reuse = sum(1 for r in all_rows for s in (r.get("trace") or []) if s.get("ended_planner")
                      and s.get("guard_reasons") and "reuse" in [x for x in s["guard_reasons"] if x])
        name = "pend.close_rejected_as_reuse"
        rep._c(name)
        (rep.warn if n_reuse else rep.note)(name, "closing turns with a candidate rejected as verbatim reuse: %d" % n_reuse)
    return rep


def check_endpoints(meta, rep):
    """pend: R0 and the ledger judge are our gpt-oss-120b (option A); a bypass is recorded, never silent."""
    if not meta.get("arm"):
        return
    for k in ("r0_model", "ledger_judge_model"):
        rep.ok("pend.endpoints", meta.get(k) == "gpt-oss-120b" or meta.get("task1_only"), "meta",
               "%s = %r (spec: gpt-oss-120b)" % (k, meta.get(k)))
    for k in ("r0_base_url", "judge_base_url"):
        u = str(meta.get(k) or "")
        rep.ok("pend.endpoints", ("127.0.0.1:8029" in u or "localhost:8029" in u) or meta.get("task1_only"), "meta",
               "%s = %r (spec: the local gpt-oss server on port 8029)" % (k, u))
    rep.ok("pend.endpoints", not meta.get("endpoint_bypass"), "meta", "PEND_ALLOW_OTHER_ENDPOINTS was set")


def check_selection(vp, rl_dir, rep):
    """validation summaries: D5 settings, the selection score recomputed from its logged parts, best = argmax.
    v16 branch only (v17 B8: a v17 run selects nothing; its validation schedule is checked in check_v17)."""
    if spec_version(rl_dir) == "v17":
        return
    summ = [json.loads(l) for l in open(vp, encoding="utf-8") if l.strip() and json.loads(l).get("kind") == "summary"]
    meta0 = _jl(os.path.join(rl_dir, "run_meta.jsonl"))[:1]
    v16_run = bool(meta0) and "task1_G" in ((meta0[0].get("config") or {}).get("args") or {})
    scored = {}
    for v in summ:
        w = "validation u%s" % v.get("update")
        rep.ok("rl.validation_d5", v.get("val_temperature") == 0.7 and sorted(v.get("val_seeds") or []) == [0, 1], w,
               "validation temperature %r / seeds %r (spec D5: 0.7, seeds 0 and 1)" % (v.get("val_temperature"), v.get("val_seeds")))
        if v.get("selection_withheld"):
            rep.warn("rl.validation_withheld", "%s: %s" % (w, v["selection_withheld"]))
            continue
        ts, t1 = v.get("turn_stats") or {}, v.get("task1")
        if v.get("selection_score") is None:
            continue
        m1 = v.get("selection_task1_metric", "term_f1")        # v16 runs: bal_p; older runs: term_f1
        if v16_run:
            rep.ok("rl.selection", m1 == "bal_p", w, "a v16 run must select on bal_p, summary says %r" % m1)
        want = (v["w_sel_cov"] * ts["coverage_mean"] - v["w_sel_w1"] * ts["turn_w1"]
                + v["w_sel_task1"] * (t1[m1] if t1 else 0.0))
        rep.ok("rl.selection", abs(v["selection_score"] - want) < 1e-9, w,
               "selection_score %r != recomputed %r" % (v["selection_score"], want))
        scored[v["update"]] = v["selection_score"]
    bp = os.path.join(rl_dir, "best.json")
    if scored and os.path.exists(bp):
        best = json.load(open(bp, encoding="utf-8"))
        rep.ok("rl.selection", best.get("selection_score") == max(scored.values()) and best.get("update") in scored,
               bp, "best.json is not the argmax of the validation selection scores")


FOLDS_GP = "/tmp2/hchsu/trec2026-usersim-benchmark/domains/main_dataset_search/folds3_goal_persona_v1.json"


def check_fewshot_leak(rows, splits, fold, rep, folds_gp=FOLDS_GP):
    """Task 2 few-shot examples: never the same conversation, goal or persona; never validation/test."""
    used = [(r, s) for r in rows for s in (r.get("trace") or []) if s.get("fewshot")]
    if not used:
        return
    if not os.path.exists(folds_gp):
        rep.ok("leak.fewshot", False, "fewshot", "goal/persona manifest %s missing: cannot check" % folds_gp)
        return
    G = json.load(open(folds_gp, encoding="utf-8"))
    goal_of, persona_of = G["goal_of"], G["persona_of"]
    f = {int(x["fold"]): x for x in splits["folds"]}.get(fold, {})
    forb = set(f.get("forbidden_for_training", []))
    for r, s in used:
        cid = r.get("conversation_id")
        w = "%s t%s" % (str(cid)[:12], s.get("t"))
        for slot in s["fewshot"]:
            for ex_cid, _ in slot:
                ok = (ex_cid != cid and ex_cid not in forb and goal_of.get(ex_cid) is not None
                      and goal_of.get(ex_cid) != goal_of.get(cid) and persona_of.get(ex_cid) is not None
                      and persona_of.get(ex_cid) != persona_of.get(cid))
                rep.ok("leak.fewshot", ok, w, "example %s: same conversation/goal/persona, no ids, or validation/test" % str(ex_cid)[:12])


def check_reselect_ids(rl_dir, val, train_all, rep):
    """Checkpoint re-selection (reselect.jsonl): only validation ids, only checkpoints that were validated in
    training, the seeds of reselect_meta, and reselect_best.json names the best non-withheld summary."""
    rp = os.path.join(rl_dir, "reselect.jsonl")
    if not os.path.exists(rp):
        return
    rows = [json.loads(l) for l in open(rp, encoding="utf-8") if l.strip()]
    vp = os.path.join(rl_dir, "validation.jsonl")
    validated = {json.loads(l)["update"] for l in open(vp, encoding="utf-8")
                 if l.strip() and json.loads(l).get("kind") == "summary"} if os.path.exists(vp) else set()
    mp = os.path.join(rl_dir, "reselect_meta.jsonl")
    metas = [json.loads(l) for l in open(mp, encoding="utf-8") if l.strip()] if os.path.exists(mp) else []
    rep.ok("rl.reselect_meta", bool(metas), mp, "reselect.jsonl without reselect_meta.jsonl")
    seeds = set(metas[-1]["seeds"]) if metas else set()
    for r in rows:
        w = "reselect u%s %s" % (r.get("update"), str(r.get("conversation_id"))[:12])
        rep.ok("rl.reselect_candidate", r.get("update") in validated, w, "re-selected a checkpoint never validated")
        if r.get("kind") in ("episode", "task1"):
            cid = r.get("conversation_id")
            rep.ok("leak.reselect_ids", cid in val and cid not in train_all, w, "re-selection row on a non-validation id")
        if r.get("kind") == "episode":
            rep.ok("rl.reselect_seeds", r.get("seed") in seeds, w, "seed %r not in %r" % (r.get("seed"), sorted(seeds)))
        if r.get("kind") == "summary":
            rep.ok("rl.reselect_seeds", set(r.get("val_seeds") or []) == seeds and r.get("reselect") is True, w,
                   "summary seeds %r / reselect flag %r" % (r.get("val_seeds"), r.get("reselect")))
    summ = {r["update"]: r for r in rows if r.get("kind") == "summary"}
    bp = os.path.join(rl_dir, "reselect_best.json")
    if summ and rep.ok("rl.reselect_best", os.path.exists(bp), bp, "reselect_best.json missing"):
        b = json.load(open(bp, encoding="utf-8"))
        ok = {u: v["selection_score"] for u, v in summ.items() if v.get("selection_score") is not None}
        exp = max(ok, key=lambda u: (ok[u], -u)) if ok else None
        rep.ok("rl.reselect_best", (b.get("best") or {}).get("update") == exp, bp,
               "reselect_best names u%r, best summary is u%r" % ((b.get("best") or {}).get("update"), exp))
        rep.note("rl.reselect_best", "re-selection scores %s -> u%s" % ({u: round(v, 4) for u, v in sorted(ok.items())}, exp))


def check_rl_selection(rl_dir, splits, fold, rep):
    """validation.jsonl only on validation ids (never train, never test); best.json (v16 branch) and manifests present.
    v17 (B8): Task 2 episodes on validation, Task 1 rows on validation_all; no best.json / re-selection."""
    folds = {int(f["fold"]): f for f in splits["folds"]}
    f = folds.get(fold, {})
    v17 = spec_version(rl_dir) == "v17"
    val, train_all = set(f.get("validation", [])), set(f.get("train_all", f.get("train", [])))
    val_all = set(f.get("validation_all", f.get("validation", []))) if v17 else val
    vp = os.path.join(rl_dir, "validation.jsonl")
    if rep.ok("rl.validation_present", os.path.exists(vp), vp, "missing"):
        n = 0
        for line in open(vp, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") in ("episode", "task1"):
                n += 1
                cid = r.get("conversation_id")
                ids = val_all if r.get("kind") == "task1" else val
                rep.ok("leak.validation_ids", cid in ids and cid not in train_all, "validation %s" % str(cid)[:12],
                       "validation row on a non-validation id")
        rep.note("rl.validation_present", "%d validation rows" % n)
    if v17:
        rep.ok("rl.best", not os.path.exists(os.path.join(rl_dir, "best.json"))
               and not os.path.exists(os.path.join(rl_dir, "reselect.jsonl")), rl_dir,
               "a v17 run has best.json / reselect.jsonl (no checkpoint selection in v17)")
    else:
        rep.ok("rl.best", os.path.exists(os.path.join(rl_dir, "best.json")), rl_dir, "best.json missing")
        check_reselect_ids(rl_dir, val, train_all, rep)
    for d in sorted(glob.glob(os.path.join(rl_dir, "ckpt", "u*"))):
        if d.endswith(".tmp"):
            continue
        mp = os.path.join(d, "rl_manifest.json")
        if rep.ok("rl.manifest", os.path.exists(mp), d, "rl_manifest.json missing"):
            m = json.load(open(mp, encoding="utf-8"))
            forb = set(f.get("forbidden_for_training", []))
            rep.ok("leak.manifest", not (set(m.get("train_scenarios", [])) | set(m.get("train_conversations", []))
                                         | set(m.get("fewshot_pool", [])) | set(m.get("sft_conversations", []))) & forb,
                   d, "manifest lists a forbidden id")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", nargs="*", default=[])
    ap.add_argument("--meta", help="default with --rl-dir: RL_DIR/run_meta.jsonl")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--split", required=True, help="train | validation | test | train_all | ...")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--training", action="store_true", help="episode files are training data")
    ap.add_argument("--expected-system-sha")
    ap.add_argument("--sepsim-path")
    ap.add_argument("--rl-dir")
    ap.add_argument("--ckpt-pattern", default="ckpt/u{update:05d}")
    ap.add_argument("--speaker-budget", type=int)
    ap.add_argument("--judge-budget", type=int, default=GJ.JUDGE_BUDGET)
    ap.add_argument("--max-new-warn", type=float, default=0.02)
    ap.add_argument("--strict", action="store_true", help="WARN also fails")
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    if a.rl_dir and a.split != "train":
        ap.error("--rl-dir rollouts are training data: use --split train")
    if not a.episodes and not a.rl_dir:
        ap.error("give --episodes and/or --rl-dir")
    if not a.meta:
        if not a.rl_dir:
            ap.error("--meta is required without --rl-dir")
        a.meta = os.path.join(a.rl_dir, "run_meta.jsonl")
    rep = verify(a.episodes, a.meta, a.splits, a.fold, a.split, a.arm, a.training, a.expected_system_sha,
                 a.sepsim_path, a.rl_dir, a.ckpt_pattern, a.speaker_budget, a.judge_budget, a.max_new_warn)
    print(rep.table())
    bad = rep.failed(a.strict)
    print("\nPIPELINE VERIFICATION %s" % ("FAILED" if bad else "PASSED"))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump({"passed": not bad, "checks": rep.as_dict(), "args": vars(a)}, f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
