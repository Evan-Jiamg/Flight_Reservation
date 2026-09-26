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
    rep.ok("trunc.r0_reply", r0t == 0, "episodes", "%d R0 replies still cut by the token budget" % r0t)
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
    """The name the trainer serves checkpoint pv's adapter under: p<pv>-<sha12 of the adapter directory>."""
    key = (rl_dir, pv)
    if key not in _ADAPTER_NAMES:
        import vllm_planner
        d = os.path.join(rl_dir, "ckpt", "u%05d" % pv, "adapter")
        _ADAPTER_NAMES[key] = ("p%d-%s" % (pv, vllm_planner.sha_dir(d)[:12])) if os.path.isdir(d) else None
    return _ADAPTER_NAMES[key]


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


def check_rl(rl_dir, rollouts, ckpt_pattern, rep):
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
                    exp = float(valid and bool(x.get("ended_planner")) == bool(r.get("real_final")))
                    rep.ok("rl.task1_groups", bool(g.get("prompt_ids")) and bool(g.get("gen_ids")) and
                           x.get("reward") == exp, w,
                           "sample without generation ids or with a reward that disagrees with the label")
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
        check_rl(rl_dir, rollouts, ckpt_pattern, rep)
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
    if arm == "pend":
        check_endpoints(meta, rep)
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
    """validation summaries: D5 settings, the selection score recomputed from its logged parts, best = argmax."""
    summ = [json.loads(l) for l in open(vp, encoding="utf-8") if l.strip() and json.loads(l).get("kind") == "summary"]
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
        want = (v["w_sel_cov"] * ts["coverage_mean"] - v["w_sel_w1"] * ts["turn_w1"]
                + v["w_sel_task1"] * (t1["term_f1"] if t1 else 0.0))
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


def check_rl_selection(rl_dir, splits, fold, rep):
    """validation.jsonl only on validation ids (never train, never test); best.json and manifests present."""
    folds = {int(f["fold"]): f for f in splits["folds"]}
    f = folds.get(fold, {})
    val, train_all = set(f.get("validation", [])), set(f.get("train_all", f.get("train", [])))
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
                rep.ok("leak.validation_ids", cid in val and cid not in train_all, "validation %s" % str(cid)[:12],
                       "validation row on a non-validation id")
        rep.note("rl.validation_present", "%d validation rows" % n)
    rep.ok("rl.best", os.path.exists(os.path.join(rl_dir, "best.json")), rl_dir, "best.json missing")
    for d in sorted(glob.glob(os.path.join(rl_dir, "ckpt", "u*"))):
        if d.endswith(".tmp"):
            continue
        mp = os.path.join(d, "rl_manifest.json")
        if rep.ok("rl.manifest", os.path.exists(mp), d, "rl_manifest.json missing"):
            m = json.load(open(mp, encoding="utf-8"))
            forb = set(f.get("forbidden_for_training", []))
            rep.ok("leak.manifest", not (set(m.get("train_scenarios", [])) | set(m.get("train_conversations", []))
                                         | set(m.get("fewshot_pool", []))) & forb, d, "manifest lists a forbidden id")


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
