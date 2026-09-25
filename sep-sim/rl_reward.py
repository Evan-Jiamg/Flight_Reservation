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

ASSESSED = ("SATISFIED", "PARTIAL", "NOT")
CONSTRAINTS = ("unparsed", "hit_max_new", "no_survivor", "judge_unknown")
# fields an episode row may carry that must NEVER influence the reward (evaluation only)
EVAL_ONLY = ("coverage", "complete", "ledger", "coverage_after", "complete_after", "n_req")

REWARD_DEFAULTS = {
    "version": "v2",
    "w_cov": 1.0,
    "w_len": 1.0,
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
    if out["version"] not in ("v2", "v3"):
        raise ValueError("reward cfg version must be v2 or v3, got %r" % (out["version"],))
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
             "turn_diff": turns - human, "decision_steps": n, "constraint_penalty": penalty}
    comps.update({"rate_" + k: rates[k] for k in CONSTRAINTS})
    return {"total": float(total), "components": comps}


def reward(episode, cfg):
    """Dispatch on cfg['version'] (v2: goal-judge reward; v3: coverage and turn count)."""
    return reward_v3(episode, cfg) if validate(cfg)["version"] == "v3" else reward_v2(episode, cfg)


AGG_KEYS = {"v2": ["goal", "over_continue", "early_stop", "decision_steps"],
            "v3": ["coverage", "len_err", "turns", "human_turns", "turn_diff", "decision_steps"]}


def aggregate(episodes_with_rewards):
    """Train aggregates for controllers: means of reward and components over a list of
    (episode, reward) pairs. Pure; the caller guarantees these are TRAIN rollouts."""
    rows = list(episodes_with_rewards)
    if not rows:
        raise ValueError("no episodes to aggregate")
    tot = [r["total"] for _, r in rows]
    m = sum(tot) / len(tot)
    sd = (sum((x - m) ** 2 for x in tot) / len(tot)) ** 0.5
    ver = "v3" if "len_err" in rows[0][1]["components"] else "v2"
    keys = AGG_KEYS[ver] + ["rate_" + k for k in CONSTRAINTS]
    comp = {k: sum(float(r["components"][k]) for _, r in rows) / len(rows) for k in keys}
    ends = {}
    for ep, _ in rows:
        ends[ep.get("end_kind")] = ends.get(ep.get("end_kind"), 0) + 1
    return {"n_episodes": len(rows), "reward_mean": m, "reward_std": sd,
            "components_mean": comp, "end_kind_frac": {k: v / len(rows) for k, v in sorted(ends.items())}}
