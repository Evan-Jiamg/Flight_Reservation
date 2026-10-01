"""Reward v2 for Planner RL (pure python, no torch). PLAN.md "Reward v2".

reward_v2(episode, cfg) -> {"total": float, "components": {...}}

  goal            P(SATISFIED) + w_partial * P(PARTIAL), from the goal_status of the LAST step whose
                  status is assessed (SATISFIED / PARTIAL / NOT). status_probs is used when present,
                  else a one-hot of status. UNKNOWN and NOT ASSESSED steps are skipped. An episode
                  with no assessed step has goal 0 and goal_assessed False (e.g. a stop at t=1).
  over_continue   (# decision steps whose status was SATISFIED and the Planner did not stop)
                  / decision_steps.
  early_stop      1 if the episode ended by planner_stop at a step whose status is in
                  cfg["early_stop_statuses"] (default ["NOT"]) and whose stop_rule is not in
                  cfg["abandon_reasons"], else 0.
  rate_<k>        per-step rates of constraint events: unparsed (planner_unparsed),
                  hit_max_new (planner_hit_max_new), no_survivor, judge_unknown (status UNKNOWN);
                  denominator decision_steps.
  total = w_goal*goal - w_over*over_continue - w_early*early_stop - sum_k lambda_k * rate_k

Every weight is a declared config value with bounds (REWARD_BOUNDS); validate() refuses anything
outside them. Evaluation-only fields (coverage, complete, ledger) are never read here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

ASSESSED = ("SATISFIED", "PARTIAL", "NOT")
CONSTRAINTS = ("unparsed", "hit_max_new", "no_survivor", "judge_unknown")
# fields an episode row may carry that must NEVER influence the reward (evaluation only)
EVAL_ONLY = ("coverage", "complete", "ledger", "coverage_after", "complete_after", "n_req")

REWARD_DEFAULTS = {
    "version": "v2",
    "w_cov": 1.0,
    "w_len": 1.0,
    "w_dist": 1.0,
    "alpha_smooth": 1.0,
    "t_max": 10.0,
    "w_goal": 1.0,
    "w_partial": 0.5,
    "w_over": 0.5,
    "w_early": 0.5,
    "lambda_unparsed": 0.0,
    "lambda_hit_max_new": 0.0,
    "lambda_no_survivor": 0.0,
    "lambda_judge_unknown": 0.0,
    "abandon_reasons": ["disgust", "cost_exceeds_value"],
    "early_stop_statuses": ["NOT"],
}
REWARD_BOUNDS = {
    "w_cov": (0.0, 10.0),
    "w_len": (0.0, 10.0),
    "w_dist": (0.0, 10.0),
    "alpha_smooth": (0.01, 10.0),
    "t_max": (1.0, 100.0),
    "w_goal": (0.0, 10.0),
    "w_partial": (0.0, 1.0),
    "w_over": (0.0, 10.0),
    "w_early": (0.0, 10.0),
    "lambda_unparsed": (0.0, 10.0),
    "lambda_hit_max_new": (0.0, 10.0),
    "lambda_no_survivor": (0.0, 10.0),
    "lambda_judge_unknown": (0.0, 10.0),
}
PROB_TOL = 1e-3


def default_cfg(**over):
    cfg = copy.deepcopy(REWARD_DEFAULTS)
    cfg.update(over)
    return validate(cfg)


def validate(cfg):
    """Return a full, checked copy of the reward config (missing keys -> declared defaults)."""
    out = copy.deepcopy(REWARD_DEFAULTS)
    for k, v in cfg.items():
        if k in REWARD_DEFAULTS:
            out[k] = copy.deepcopy(v)
    for k, (lo, hi) in REWARD_BOUNDS.items():
        v = float(out[k])
        if not (lo <= v <= hi):
            raise ValueError("reward cfg %s=%r outside declared bounds [%g, %g]" % (k, v, lo, hi))
        out[k] = v
    if out["version"] not in ("v2", "v3", "v4"):
        raise ValueError("reward cfg version must be v2, v3 or v4, got %r" % (out["version"],))
    for k in ("abandon_reasons", "early_stop_statuses"):
        if not isinstance(out[k], (list, tuple)) or not all(isinstance(x, str) for x in out[k]):
            raise ValueError("reward cfg %s must be a list of strings" % k)
        out[k] = list(out[k])
    return out


def cfg_sha(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def status_probs(goal_status):
    """-> dict over ASSESSED statuses, or None when the step is not assessed."""
    if not goal_status:
        return None
    st = goal_status.get("status")
    if st not in ASSESSED:
        return None
    sp = goal_status.get("status_probs")
    if sp:
        p = {k: float(sp.get(k, 0.0)) for k in ASSESSED}
        if any(v < 0 for v in p.values()):
            raise ValueError("negative status probability %r" % sp)
        s = sum(p.values())
        if abs(s - 1.0) > PROB_TOL:
            raise ValueError("status_probs must sum to 1 (got %.6f)" % s)
        return {k: v / s for k, v in p.items()}
    return {k: float(k == st) for k in ASSESSED}


def _steps(episode):
    trace = episode.get("trace") or []
    n = int(episode.get("decision_steps", len(trace)))
    if n != len(trace):
        raise ValueError("decision_steps %d != len(trace) %d" % (n, len(trace)))
    return trace, n


def _stopped(step):
    return step.get("decision") == "planner_stop" or bool(step.get("ended_planner"))


def reward_v2(episode, cfg):
    cfg = validate(cfg)
    trace, n = _steps(episode)
    if n == 0:
        raise ValueError("episode without decision steps")
    # goal: last assessed step
    goal, goal_t, goal_probs = 0.0, None, None
    for step in trace:
        p = status_probs(step.get("goal_status"))
        if p is not None:
            goal_probs, goal_t = p, step["t"]
    if goal_probs is not None:
        goal = goal_probs["SATISFIED"] + cfg["w_partial"] * goal_probs["PARTIAL"]
    # over-continue: satisfied yet the Planner went on
    over_n = sum(1 for s in trace
                 if (s.get("goal_status") or {}).get("status") == "SATISFIED" and not _stopped(s))
    over = over_n / n
    # early stop
    last = trace[-1]
    early = 0
    if episode.get("end_kind") == "planner_stop":
        if last.get("decision") != "planner_stop":
            raise ValueError("end_kind planner_stop but last step decision is %r" % last.get("decision"))
        st = (last.get("goal_status") or {}).get("status", "NOT ASSESSED")
        if st in cfg["early_stop_statuses"] and (last.get("stop_rule") or "none") not in cfg["abandon_reasons"]:
            early = 1
    counts = {
        "unparsed": sum(bool(s.get("planner_unparsed")) for s in trace),
        "hit_max_new": sum(bool(s.get("planner_hit_max_new")) for s in trace),
        "no_survivor": sum(bool(s.get("no_survivor")) for s in trace),
        "judge_unknown": sum((s.get("goal_status") or {}).get("status") == "UNKNOWN" for s in trace),
    }
    rates = {k: counts[k] / n for k in CONSTRAINTS}
    penalty = sum(cfg["lambda_" + k] * rates[k] for k in CONSTRAINTS)
    total = cfg["w_goal"] * goal - cfg["w_over"] * over - cfg["w_early"] * early - penalty
    comps = {"goal": goal, "goal_assessed": goal_probs is not None, "goal_t": goal_t,
             "over_continue": over, "over_continue_steps": over_n, "early_stop": early,
             "decision_steps": n, "constraint_penalty": penalty}
    comps.update({"rate_" + k: rates[k] for k in CONSTRAINTS})
    return {"total": float(total), "components": comps}


def reward_v3(episode, cfg):
    """Reward v3 (pend arm, no goal judge; user direction 2026-09-25).

      coverage   the requirement ledger's final coverage of this episode (0..1). DECLARED: unlike v2,
                 v3 uses the ledger as a reward term (it keeps the Planner from ending before the
                 assistant has delivered anything); coverage therefore cannot be read as an independent
                 evaluation of a v3-trained policy without saying so.
      len_err    |emitted user turns - the real person's number of messages in this conversation| / t_max
                 (the human count is a label of the TRAIN conversation being rolled out).
      rate_<k>   constraint rates as in v2 (unparsed, hit_max_new, no_survivor; judge_unknown is 0).
      total = w_cov*coverage - w_len*len_err - sum_k lambda_k * rate_k
    """
    cfg = validate(cfg)
    trace, n = _steps(episode)
    if n == 0:
        raise ValueError("episode without decision steps")
    if "human_turns" not in episode:
        raise ValueError("reward v3 needs episode['human_turns']")
    turns, human = int(episode["emitted_user_turns"]), int(episode["human_turns"])
    cov = float(episode["coverage"])
    if not (0.0 <= cov <= 1.0):
        raise ValueError("coverage %r outside [0, 1]" % cov)
    target = min(human, int(cfg["t_max"]))       # the protocol caps an episode at t_max: that is the reachable target
    len_err = abs(turns - target) / cfg["t_max"]
    counts = {
        "unparsed": sum(bool(s.get("planner_unparsed")) for s in trace),
        "hit_max_new": sum(bool(s.get("planner_hit_max_new")) for s in trace),
        "no_survivor": sum(bool(s.get("no_survivor")) for s in trace),
        "judge_unknown": sum((s.get("goal_status") or {}).get("status") == "UNKNOWN" for s in trace),
    }
    rates = {k: counts[k] / n for k in CONSTRAINTS}
    penalty = sum(cfg["lambda_" + k] * rates[k] for k in CONSTRAINTS)
    total = cfg["w_cov"] * cov - cfg["w_len"] * len_err - penalty
    comps = {"coverage": cov, "len_err": len_err, "turns": turns, "human_turns": human, "turn_target": target,
             "stop_part": -cfg["w_len"] * len_err,
             "turn_diff": turns - human, "decision_steps": n, "constraint_penalty": penalty}
    comps.update({"rate_" + k: rates[k] for k in CONSTRAINTS})
    return {"total": float(total), "components": comps}


def turn_distribution(turns, t_max, alpha):
    """Smoothed distribution over conversation lengths 0..t_max (lengths above t_max count as t_max,
    the protocol's cap): (count + alpha) / (n + alpha * (t_max + 1))."""
    t_max = int(t_max)
    c = [0] * (t_max + 1)
    for x in turns:
        c[min(max(int(x), 0), t_max)] += 1
    n = sum(c)
    return [(v + alpha) / (n + alpha * (t_max + 1)) for v in c]


def reward_v4(episode, cfg, ctx):
    """Reward v4 (user decision D1(b), 2026-09-25): match the human DISTRIBUTION of conversation length.

      dist      log p_h(T) - log q(T), T = emitted turns of this episode (capped at t_max);
                p_h = smoothed distribution of the real people's number of messages in the TRAIN
                conversations (ctx["p_h"]); q = smoothed distribution of T over the current update's
                rollouts (ctx["q"]). E_q[dist] = -KL(q || p_h), so the optimum is q = p_h (the whole
                distribution, not one length for every conversation).
      coverage  the ledger's final coverage (declared reward term, as in v3).
      rate_<k>  constraint rates (unparsed, hit_max_new, no_survivor; judge_unknown 0).
      total = w_cov*coverage + w_dist*dist - sum_k lambda_k * rate_k
    components["stop_part"] = w_dist*dist: the part caused by the end decisions (stop credit)."""
    import math
    cfg = validate(cfg)
    trace, n = _steps(episode)
    if n == 0:
        raise ValueError("episode without decision steps")
    t_max = int(cfg["t_max"])
    p_h, q = ctx["p_h"], ctx["q"]
    if len(p_h) != t_max + 1 or len(q) != t_max + 1:
        raise ValueError("distributions must cover 0..t_max")
    turns = int(episode["emitted_user_turns"])
    T = min(max(turns, 0), t_max)
    dist = math.log(p_h[T]) - math.log(q[T])
    cov = float(episode["coverage"])
    if not (0.0 <= cov <= 1.0):
        raise ValueError("coverage %r outside [0, 1]" % cov)
    counts = {
        # a capped plan is also unparsed: it pays lambda_hit_max_new only, not both
        "unparsed": sum(bool(s.get("planner_unparsed")) and not bool(s.get("planner_hit_max_new")) for s in trace),
        "hit_max_new": sum(bool(s.get("planner_hit_max_new")) for s in trace),
        "no_survivor": sum(bool(s.get("no_survivor")) for s in trace),
        "judge_unknown": 0,
    }
    rates = {k: counts[k] / n for k in CONSTRAINTS}
    penalty = sum(cfg["lambda_" + k] * rates[k] for k in CONSTRAINTS)
    stop_part = cfg["w_dist"] * dist
    total = cfg["w_cov"] * cov + stop_part - penalty
    comps = {"coverage": cov, "dist": dist, "turns": turns, "log_p_h": math.log(p_h[T]), "log_q": math.log(q[T]),
             "decision_steps": n, "constraint_penalty": penalty, "stop_part": stop_part}
    if "human_turns" in episode:
        comps["human_turns"] = int(episode["human_turns"])
    comps.update({"rate_" + k: rates[k] for k in CONSTRAINTS})
    return {"total": float(total), "components": comps}


def reward(episode, cfg, ctx=None):
    """Dispatch on cfg['version'] (v2: goal-judge reward; v3: coverage and per-scenario turn count;
    v4: coverage and the human length distribution -- needs ctx with p_h and q)."""
    v = validate(cfg)["version"]
    if v == "v4":
        if not ctx or "p_h" not in ctx or "q" not in ctx:
            raise ValueError("reward v4 needs ctx={'p_h': ..., 'q': ...}")
        return reward_v4(episode, cfg, ctx)
    return reward_v3(episode, cfg) if v == "v3" else reward_v2(episode, cfg)


# ================================================================== SPEC v18 (ops/SPEC_v18_multiobj_rerank.md)
# Pure python; the act rules use the E1.6 tree's own sepsim.acts (normalise, COARSE_ORDER), loaded by load_acts() -- the
# trainer, the verifier and the labeller all call these functions, so r_act / r_len are recomputed with the SAME code.
V18_MOVES = ("Disclose", "Reveal", "Inquire", "Navigate", "Note")        # A: the five non-closing moves (§3.1.2)
V18_Q7 = ("Disclose", "Reveal", "Inquire", "Navigate", "Note", "Complete", "Other")
_ACTS = {}


def load_acts():
    """The sepsim.acts module of the E1.6 tree (cf19400). Already importable in a run (task2_env puts the tree on sys.path);
    else from $E1R_TREE, the server tree, or the local audit snapshot ../audit_e1r. -> module (cached)."""
    if "m" in _ACTS:
        return _ACTS["m"]
    import importlib
    import os
    import sys
    try:
        m = importlib.import_module("sepsim.acts")
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        for root in (os.environ.get("E1R_TREE"), "/tmp2/mzjiang_usersim/grpo_planner/trees/e1r_cf19400",
                     os.path.join(here, "..", "audit_e1r")):
            if root and os.path.isfile(os.path.join(root, "sepsim", "acts.py")):
                sys.path.insert(0, os.path.abspath(root))
                break
        m = importlib.import_module("sepsim.acts")
    _ACTS["m"] = m
    return m


def acts_sha():
    import os
    m = load_acts()
    return hashlib.sha256(open(m.__file__, "rb").read()).hexdigest() if os.path.isfile(m.__file__) else None


def _float_or_none(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def act_entries(dist):
    """§3.1.2 drop rules 1-2, exactly as E1.6 sample_act: the entries of an act_distribution that are dicts, whose
    (move, act) normalise without turning a non-"other" act into "other", and whose p converts to float.
    -> [(move, act, p, raw entry)] (p may be <= 0)."""
    acts = load_acts()
    out = []
    for e in (dist if isinstance(dist, list) else []):
        if not isinstance(e, dict):
            continue
        mv, ac = acts.normalise(str(e.get("move", "")), str(e.get("act", "")))
        if ac == "other" and str(e.get("act", "")).strip().lower() != "other":
            continue
        p = _float_or_none(e.get("p"))
        if p is None:
            continue
        out.append((mv, ac, p, e))
    return out


def act_dist_parseable(dist):
    """§3.1.5 (turn 1): at least one act_distribution entry survives the drop rules."""
    return bool(act_entries(dist))


def plan_act_probs(dist):
    """§3.1.2: the plan's distribution p over A = the five non-closing moves (rules 1-4: drop, p <= 0 not counted, the
    same move's entries summed, A only, renormalised); A-mass 0 -> the uniform distribution (fallback True).
    -> (dict move -> p, fallback)."""
    mass = {m: 0.0 for m in V18_MOVES}
    for mv, _ac, p, _e in act_entries(dist):
        if p > 0 and mv in mass:
            mass[mv] += p
    tot = sum(mass.values())
    if tot <= 0:
        return {m: 1.0 / len(V18_MOVES) for m in V18_MOVES}, True
    return {m: v / tot for m, v in mass.items()}, False


def label_majority(q7):
    """The majority move of a soft label (most votes; ties broken by acts.COARSE_ORDER). None for an empty label."""
    order = load_acts().COARSE_ORDER
    best = None
    for m in order:
        v = float((q7 or {}).get(m, 0.0))
        if v > 0 and (best is None or v > best[1]):
            best = (m, v)
    return best[0] if best else None


def human_act_probs(q7):
    """§3.1.2: the human's q = q7 without Complete / Other, renormalised over A; None when nothing is left."""
    q = {m: float((q7 or {}).get(m, 0.0)) for m in V18_MOVES}
    tot = sum(q.values())
    return {m: v / tot for m, v in q.items()} if tot > 0 else None


def r_act_of(dist, label, t, n):
    """§3.1.2 r_act = 1 - 1/2 sum_{m in A} (p(m) - q(m))^2 of one plan. label: {"q7", "n_valid_votes"} (None = no label).
    -> (value or None, skip reason or None, uniform fallback used)."""
    if label is None or int(label.get("n_valid_votes", 0)) < 2:
        return None, "votes", False
    if t >= n:
        return None, "final", False
    maj = label_majority(label.get("q7"))
    if maj == "Complete":
        return None, "complete", False
    if maj == "Other" or maj is None:
        return None, "other", False
    q = human_act_probs(label["q7"])
    if q is None:
        return None, "other", False
    p, fb = plan_act_probs(dist)
    return 1.0 - 0.5 * sum((p[m] - q[m]) ** 2 for m in V18_MOVES), None, fb


def plan_length_for(dist, move):
    """§3.1.3: the FIRST act_distribution entry whose normalised move is `move` and whose length_words converts to a
    positive integer (round), matched as read_plan_v3 matches. -> int or None."""
    acts = load_acts()
    for e in (dist if isinstance(dist, list) else []):
        if not isinstance(e, dict):
            continue
        mv, _ac = acts.normalise(str(e.get("move", "")), str(e.get("act", "")))
        if mv != move:
            continue
        v = _float_or_none(e.get("length_words"))
        if v is None:
            continue
        lw = int(round(v))
        if lw > 0:
            return lw
    return None


def r_len_of(dist, label, human_words, band=2.0 / 3.0):
    """§3.1.3 r_len = min(1, rho / rho0), rho = min(l+1, L*+1) / max(l+1, L*+1), l = the plan length of the human's
    majority move m*. -> (value or None, skip reason or None): None/"votes" (< 2 valid votes), None/"other" (m* Other),
    0.0/"missing" (no entry of m* with a positive length)."""
    if label is None or int(label.get("n_valid_votes", 0)) < 2:
        return None, "votes"
    maj = label_majority(label.get("q7"))
    if maj is None or maj == "Other":
        return None, "other"
    lw = plan_length_for(dist, maj)
    if lw is None:
        return 0.0, "missing"
    a, b = lw + 1.0, float(human_words) + 1.0
    rho = min(a, b) / max(a, b)
    return min(1.0, rho / float(band)), None


def act_entropy(dist):
    """Entropy (nats) of the plan's distribution over A (the uniform fallback included)."""
    p, _ = plan_act_probs(dist)
    return -sum(v * math.log(v) for v in p.values() if v > 0)


def task1_components(sample, t, n, label, human_words, band=2.0 / 3.0):
    """SPEC v18 §3.1 / §3.1.5: the reward components of ONE Task 1 sample (a task2_env.task1_sample row, already scored:
    status in valid / dropped / invalid and, at t >= 2 and valid, p_end). -> {"r_stop", "r_act", "r_len", "r_fmt"} with
    None = "not applicable", plus "skip" reasons. A dropped sample has no component at all (no gradient)."""
    st = sample.get("status")
    out = {"r_stop": None, "r_act": None, "r_len": None, "r_fmt": None, "skip": {}}
    if st == "dropped":
        return out
    if st == "invalid":
        out["r_fmt"] = 0.0
        return out
    if st != "valid":
        raise ValueError("Task 1 sample status %r" % st)
    out["r_fmt"] = 1.0
    if t >= 2:
        pe = sample.get("p_end")
        if pe is None:
            raise ValueError("a valid t >= 2 sample without P_end")
        y = 1.0 if t == n else 0.0
        out["r_stop"] = 1.0 - (float(pe) - y) ** 2
    dist = sample.get("act_distribution_raw")
    ra, why_a, fb = r_act_of(dist, label, t, n)
    out["r_act"] = ra
    if why_a:
        out["skip"]["act"] = why_a
    out["uniform_fallback"] = bool(fb)
    rl, why_l = r_len_of(dist, label, human_words, band)
    out["r_len"] = rl
    if why_l:
        out["skip"]["len"] = why_l
    return out


def r_fmt2_of(episode):
    """§3.2 r_fmt2 = 1 - (steps that are unparsed, capped or without a guard survivor) / decision steps (a step counted
    once when several hold)."""
    trace, n = _steps(episode)
    if n == 0:
        raise ValueError("episode without decision steps")
    bad = sum(1 for s in trace if s.get("planner_unparsed") or s.get("planner_hit_max_new") or s.get("no_survivor"))
    return 1.0 - bad / n


def reward_v5(episode, t_max=10, w_cov=0.0):
    """SPEC v18 §3.2 Task 2 components: r_turn = 1 - |T - min(H, t_max)| / t_max and r_fmt2; coverage is a diagnostic
    only (w_cov 0, §17 Q8 (a)) -- a judge failure makes it None. -> {"r_turn", "r_fmt2", "coverage_diag"}."""
    if float(w_cov) != 0.0:
        raise ValueError("reward v5: w_cov must be 0 (SPEC v18 §17 Q8 (a): coverage is a diagnostic only)")
    if "human_turns" not in episode:
        raise ValueError("reward v5 needs episode['human_turns']")
    T, H = int(episode["emitted_user_turns"]), int(episode["human_turns"])
    tm = int(t_max)
    return {"r_turn": 1.0 - abs(T - min(H, tm)) / float(tm), "r_fmt2": r_fmt2_of(episode),
            "coverage_diag": coverage_diag(episode), "turns": T, "human_turns": H}


def coverage_diag(episode):
    """§3.2: the Ledger coverage as a diagnostic; None when a ledger verdict was lost (judge_empty / unparseable / error)."""
    c = episode.get("episode_counters") or {}
    if c.get("judge_empty", 0) or c.get("judge_unparseable", 0) or c.get("judge_error", 0):
        return None
    cov = episode.get("coverage")
    return None if cov is None else float(cov)


def clean_v18(episode):
    """§3.2 clean_v18 recomputed from an episode row: no cut / empty R0 reply, no emitted capped message, no compacted
    step, at least one emitted turn (the ledger-judge counters are recorded only)."""
    c = episode.get("episode_counters") or {}
    return (c.get("r0_len_truncated", 0) == 0 and c.get("r0_empty", 0) == 0 and not episode.get("emitted_capped_steps")
            and not episode.get("compacted_steps") and int(episode.get("emitted_user_turns") or 0) > 0)


AGG_KEYS = {"v2": ["goal", "over_continue", "early_stop", "decision_steps"],
            "v3": ["coverage", "len_err", "turns", "human_turns", "turn_diff", "decision_steps"],
            "v4": ["coverage", "dist", "turns", "decision_steps"]}


def aggregate(episodes_with_rewards):
    """Train aggregates for controllers: means of reward and components over a list of
    (episode, reward) pairs. Pure; the caller guarantees these are TRAIN rollouts."""
    rows = list(episodes_with_rewards)
    if not rows:
        raise ValueError("no episodes to aggregate")
    tot = [r["total"] for _, r in rows]
    m = sum(tot) / len(tot)
    sd = (sum((x - m) ** 2 for x in tot) / len(tot)) ** 0.5
    c0 = rows[0][1]["components"]
    ver = "v4" if "dist" in c0 else ("v3" if "len_err" in c0 else "v2")
    keys = AGG_KEYS[ver] + ["rate_" + k for k in CONSTRAINTS]
    comp = {k: sum(float(r["components"][k]) for _, r in rows) / len(rows) for k in keys}
    ends = {}
    for ep, _ in rows:
        ends[ep.get("end_kind")] = ends.get(ep.get("end_kind"), 0) + 1
    return {"n_episodes": len(rows), "reward_mean": m, "reward_std": sd,
            "components_mean": comp, "end_kind_frac": {k: v / len(rows) for k, v in sorted(ends.items())}}
