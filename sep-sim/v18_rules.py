"""SPEC v18 rules shared by the trainer (train_planner_rl.py), the verifier (verify_pipeline.py), the test evaluation and
the smoke test -- one definition, recomputed from the logs with the SAME code (ops/SPEC_v18_multiobj_rerank.md).
Pure python (the act rules come from rl_reward, which loads the E1.6 tree's sepsim.acts).

  load_labels / check_label_cids   the act labels of label_acts.py (train_all for the reward, validation_all for §7)
  t1_status                        §3.1.5 the state of a Task 1 sample (valid / dropped / invalid) from its row fields
  t1_members                       §3.1 the reward components of a Task 1 group's kept samples
  t2_members                       §3.2 the components of a Task 2 group's clean_v18 episodes
  advantages_t1 / advantages_t2    §4.2 c, z (clipped), A_pre / A_plan / A_ep of one group, with the group's skip kind
  task1_sample_masks               §4.4 the masks of a Task 1 sample (prefix, plan, value) -- what the learner gets
  validation_metrics               §7.2 the Task 1 validation scores of one policy (stop / act / length / invalid)
  selection / stop_rule            §7.3 the candidate set, J, guards and order; the early-stop rule
  tau_alarms                       §4.3 the warnings of one update
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import rl_reward as RR

T1_KEYS = ("stop", "act", "len", "fmt")
T2_KEYS = ("turn", "fmt2")
SCALED_T1 = ("stop", "act", "len")          # measured on the u0 batch (§4.1); fmt fixed at --fmt-scale
SCALED_T2 = ("turn",)
MUST_NOT_FREEZE = ("stop", "act", "turn")   # §4.1 item 4: training refused when one of these is frozen
TAU_V17_MAX = {"task1": 0.0248, "task2": 0.1194}        # v17 per-update range maxima (SPEC v18 §4.3 table)
TAU_ALARM = {"task1": 0.050, "task2": 0.239}            # 2x the v17 range (§4.3)
GRAD_RATIO_ALARM = 8.0
RL_GRAD_ALARM = 0.025
GUARDS = {"nll": 0.05, "act": 0.02, "len": 0.05, "entropy_frac": 0.5, "invalid": 1}


def sha_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ------------------------------------------------------------------ labels (§1.1)
def load_labels(path):
    """label_acts.py output: one row per message {conversation_id, t, q7, n_valid_votes, n_words, ...} and its
    <path>.meta.json. -> (labels {(cid, t): row}, meta or None, sha256 of the label file)."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    labels = {}
    for r in rows:
        key = (r["conversation_id"], int(r["t"]))
        if key in labels:
            raise ValueError("%s: two labels for %s t%d" % (path, key[0], key[1]))
        q = r.get("q7") or {}
        bad = set(q) - set(RR.V18_Q7)
        if bad:
            raise ValueError("%s: %s t%d has q7 keys outside the 7 moves: %s" % (path, key[0], key[1], sorted(bad)))
        labels[key] = r
    mp = path + ".meta.json"
    meta = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else None
    return labels, meta, sha_file(path)


def check_label_cids(labels, allowed, forbidden, what="train_all"):
    """Every label conversation inside `allowed` and outside `forbidden`. -> list of offending ids."""
    return sorted({c for c, _ in labels if c not in allowed or c in forbidden})


def label_of(labels, cid, t):
    r = labels.get((cid, int(t)))
    if r is None:
        return None
    return {"q7": r.get("q7") or {}, "n_valid_votes": int(r.get("n_valid_votes", 0))}


# ------------------------------------------------------------------ Task 1 (§3.1)
def t1_status(x, t):
    """§3.1.5 the state of one Task 1 sample from the fields task2_env.task1_sample logged:
      t >= 2  valid (decision_valid and mask_ok) / dropped (decision_valid, the value not located) / invalid
      t = 1   valid (valid_t1 and the value mask located + decode-checked) / dropped (valid_t1, value mask not found or
              not matching) / invalid (not parsed, capped, act_distribution unparseable, end_session missing / invalid).
    -> (status, drop reason or None)."""
    if int(t) == 1:
        if not x.get("valid_t1"):
            return "invalid", None
        if not x.get("value_mask_ok"):
            return "dropped", "t1_value_mismatch"
        return "valid", None
    if not x.get("decision_valid"):
        return "invalid", None
    if not x.get("mask_ok"):
        return "dropped", "stop_mask_mismatch"
    return "valid", None


def t1_member(x, t, n, label, human_words, band):
    """The component dict of one kept Task 1 sample: {"stop", "act", "len", "fmt"} (None = not applicable)."""
    c = RR.task1_components(x, t, n, label, human_words, band)
    return {"stop": c["r_stop"], "act": c["r_act"], "len": c["r_len"], "fmt": c["r_fmt"],
            "skip": c["skip"], "uniform_fallback": c.get("uniform_fallback", False)}


def t1_members(row, labels, band):
    """The kept samples of a Task 1 group row (dropped ones removed) with their components. -> (kept samples, members)."""
    t, n = int(row["t"]), int(row["n_real"])
    lab = label_of(labels, row["conversation_id"], t)
    kept = [x for x in row["samples"] if x.get("status") != "dropped"]
    return kept, [t1_member(x, t, n, lab, row["human_words"], band) for x in kept]


def t2_member(episode, t_max):
    r = RR.reward_v5(episode, t_max=t_max)
    return {"turn": r["r_turn"], "fmt2": r["r_fmt2"]}


def _skip_kind(spread, frozen, keys):
    """A group whose components have no spread (among the non-frozen ones) is skipped (Dr. GRPO)."""
    return not any(spread[k] for k in keys if k not in frozen)


def advantages_t1(members, scales, frozen, z_clip, weights, kappa1):
    """§4.2 Task 1: c, z, A_pre = kappa1 w_stop z_stop, A_plan = kappa1 (w_act z_act + w_len z_len + w_fmt z_fmt).
    -> (rows [{"c", "z", "clipped", "A_pre", "A_plan"}] or None when the group is skipped (zero spread), spread)."""
    import rl_algos as RA
    rows, spread = RA.fixed_scale_z(members, T1_KEYS, scales, frozen, z_clip)
    if _skip_kind(spread, frozen, T1_KEYS):
        return None, spread
    for r in rows:
        z = r["z"]
        r["A_pre"] = kappa1 * weights["stop"] * z["stop"]
        r["A_plan"] = kappa1 * (weights["act"] * z["act"] + weights["len"] * z["len"] + weights["fmt"] * z["fmt"])
    return rows, spread


def advantages_t2(members, scales, frozen, z_clip, weights, kappa2):
    """§4.2 Task 2: A_ep = kappa2 (w_turn z_turn + w_fmt2 z_fmt2). -> (rows or None, spread)."""
    import rl_algos as RA
    rows, spread = RA.fixed_scale_z(members, T2_KEYS, scales, frozen, z_clip)
    if _skip_kind(spread, frozen, T2_KEYS):
        return None, spread
    for r in rows:
        z = r["z"]
        r["A_ep"] = kappa2 * (weights["turn"] * z["turn"] + weights["fmt2"] * z["fmt2"])
    return rows, spread


def task1_sample_masks(x, t):
    """§4.4: the masks of a Task 1 sample's generation -> (where, prefix_mask, plan_mask):
      valid t >= 2   "prefix+plan"  prefix_mask = before the value, plan_mask = 1 - stop_mask
      valid t = 1    "plan_t1"      plan_mask = 1 - value_mask (the ignored turn-1 value never gets an advantage)
      invalid        "all"          plan_mask = all ones (A_plan holds only the format term)."""
    import rl_algos as RA
    g = x["planner_gen"]
    n = len(g["gen_ids"])
    if x["status"] == "valid" and int(t) >= 2:
        pm = RA.prefix_mask_of(g.get("stop_mask"))
        if pm is None:
            raise AssertionError("a valid t >= 2 Task 1 sample without a located value")
        return "prefix+plan", pm, RA.plan_mask_of(n, g["stop_mask"])
    if x["status"] == "valid":
        if g.get("value_mask") is None:
            raise AssertionError("a valid turn-1 Task 1 sample without its value mask")
        return "plan_t1", None, RA.plan_mask_of(n, g["value_mask"])
    return "all", None, RA.plan_mask_of(n, None)


# ------------------------------------------------------------------ validation (§7.2)
def _quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def w1_int(a, b):
    """Wasserstein-1 between two samples of integers (sum of |CDF difference| over unit steps)."""
    if not a or not b:
        return None
    hi = max(max(a), max(b))
    w, ca, cb = 0.0, 0.0, 0.0
    for k in range(0, hi + 1):
        ca += sum(1 for x in a if x == k) / len(a)
        cb += sum(1 for x in b if x == k) / len(b)
        w += abs(ca - cb)
    return w


def turn_plan_valid(x):
    """§3.1.5 format validity of a greedy validation plan (one run_task1 turn): t >= 2 parsed, not capped, a valid
    end_session value (the v17 decision); t = 1 also the act_distribution parseable."""
    d = x.get("planner_diag") or {}
    ok = not x.get("planner_unparsed") and not x.get("planner_hit_max_new") and d.get("end_session_valid") is True
    if int(x["t"]) == 1:
        ok = ok and RR.act_dist_parseable(x.get("act_distribution_raw"))
    else:
        ok = ok and not d.get("end_session_t1_ignored")
    return bool(ok)


def validation_metrics(t1rows, val_labels, human_words_by_cid, band):
    """§7.2 per validated policy, from its validation Task 1 rows (Trainer.task1_eval_row: the greedy teacher-forced run
    with the v18 turn fields, and the end probabilities at t >= 2): every score is "higher is better".
      stop_score    mean 1 - (P_end - y)^2 over the probe points (t >= 2; the unscored-point rule as v17)
      act_score     mean r_act of the greedy plans (§3.1.2, validation labels) over the applicable valid turns
      act_entropy   mean entropy over A of the valid greedy plans
      len_score     mean r_len over the applicable valid turns
      len_iqr       the interquartile range of the plans' own length_from_planner
      len_w1        W1 between the selected messages' and the people's word counts
      n_invalid     greedy plans that are not valid (§3.1.5), every turn
      rerank_changed_frac   share of turns whose selection the reranker changed (None without a reranker)."""
    pts = [p for r in t1rows for p in (r.get("end_probs") or [])]
    stop = [1.0 - (float(p["p_end"]) - (1.0 if p["real_final"] else 0.0)) ** 2 for p in pts]
    acts, ents, lens, lw, sel_w, hum_w, n_inv, rc = [], [], [], [], [], [], 0, []
    for r in t1rows:
        cid = r["conversation_id"]
        turns = r["task1"]["turns"]
        n = len(turns)
        hw = human_words_by_cid[cid]
        for x in turns:
            t = int(x["t"])
            sel_w.append(int(x.get("selected_words") or 0))
            hum_w.append(int(hw[t - 1]))
            if x.get("rerank_changed") is not None:
                rc.append(bool(x["rerank_changed"]))
            if not turn_plan_valid(x):
                n_inv += 1
                continue
            dist = x.get("act_distribution_raw")
            lab = label_of(val_labels, cid, t)
            ra, _, _ = RR.r_act_of(dist, lab, t, n)
            if ra is not None:
                acts.append(ra)
            ents.append(RR.act_entropy(dist))
            rl_, _ = RR.r_len_of(dist, lab, hw[t - 1], band)
            if rl_ is not None:
                lens.append(rl_)
            l0 = (x.get("planner_diag") or {}).get("length_from_planner")
            if isinstance(l0, int) and l0 > 0:
                lw.append(l0)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    q1, q3 = _quantile(lw, 0.25), _quantile(lw, 0.75)
    return {"stop_score": mean(stop), "act_score": mean(acts), "act_entropy": mean(ents), "len_score": mean(lens),
            "len_iqr": (q3 - q1) if lw else None, "len_w1": w1_int(sel_w, hum_w), "n_invalid": n_inv,
            "rerank_changed_frac": mean([1.0 if c else 0.0 for c in rc]) if rc else None,
            "n_stop_points": len(stop), "n_act_points": len(acts), "n_len_points": len(lens), "n_turns": len(sel_w)}


def drift_of(episodes, t_max):
    """mean(T - min(H, t_max)) over episodes (None without one)."""
    if not episodes:
        return None
    return sum(int(e["emitted_user_turns"]) - min(int(e["human_turns"]), int(t_max)) for e in episodes) / len(episodes)


# ------------------------------------------------------------------ selection and stopping (§7.3)
def j_of(m, m0, w):
    """J(u) = sum_k w_k (s_k(u) - s_k(u0)), s = (stop_score, act_score, len_score), nominal weights; None if a score is
    missing."""
    parts = [("stop_score", w["stop"]), ("act_score", w["act"]), ("len_score", w["len"])]
    if any(m.get(k) is None or m0.get(k) is None for k, _ in parts):
        return None
    return sum(wk * (m[k] - m0[k]) for k, wk in parts)


def drift_stop_at(drift_by_u, u, margin):
    """§5 / §7.3 (audit B NIT 1): the length-drift STOP rule triggered by update u -- the drift of u AND of u-1 below
    -margin (two consecutive updates, as v17's drift_decision). A single update's training-rollout drift (which measures
    pi_(u-1)) does not exclude u."""
    d, p = drift_by_u.get(u), drift_by_u.get(u - 1)
    return u >= 2 and d is not None and p is not None and d < -float(margin) and p < -float(margin)


def guards_of(m, m0, drift_stop, margin=None):
    """§7.3 item 1 against u0 (a missing value fails its guard); drift_stop: the drift stop rule triggered by this update
    (drift_stop_at). -> {name: bool}."""
    def ge(a, b):
        return a is not None and b is not None and a >= b

    def le(a, b):
        return a is not None and b is not None and a <= b
    return {"nll": le(m.get("nll"), None if m0.get("nll") is None else m0["nll"] + GUARDS["nll"]),
            "act": ge(m.get("act_score"), None if m0.get("act_score") is None else m0["act_score"] - GUARDS["act"]),
            "len": ge(m.get("len_score"), None if m0.get("len_score") is None else m0["len_score"] - GUARDS["len"]),
            "entropy": ge(m.get("act_entropy"),
                          None if m0.get("act_entropy") is None else GUARDS["entropy_frac"] * m0["act_entropy"]),
            "invalid": le(m.get("n_invalid"), None if m0.get("n_invalid") is None else m0["n_invalid"] + GUARDS["invalid"]),
            "no_length_drift": not drift_stop}


def selection(metrics_by_u, drift_by_u, margin, last_u, w):
    """§7.3 items 1-2: the candidates u in 1..last_u passing every guard (u is excluded when the length-drift stop rule
    triggered AT u, drift_stop_at), J for every u, the candidates ordered by J (larger first; a tie -> the later update).
    metrics_by_u: {u: validation_metrics + "nll"} incl. u0. -> dict."""
    m0 = metrics_by_u[0]
    J, guards = {0: 0.0}, {}
    for u in range(1, int(last_u) + 1):
        m = metrics_by_u.get(u)
        if m is None:
            J[u], guards[u] = None, {"validated": False}
            continue
        J[u] = j_of(m, m0, w)
        guards[u] = guards_of(m, m0, drift_stop_at(drift_by_u, u, margin))
    cands = [u for u in range(1, int(last_u) + 1) if J.get(u) is not None and guards.get(u) and all(guards[u].values())]
    order = sorted(cands, key=lambda u: (-J[u], -u))
    return {"candidates": cands, "order": order, "J": {str(k): v for k, v in J.items()},
            "guards": {str(k): v for k, v in guards.items()}}


def stop_rule(drift_by_u, j_by_u, margin, max_updates, done_u):
    """§7.3 item 4 + §5: the first u in 1..done_u with (a) two consecutive updates of drift < -margin -> "length_drift",
    (b) J(u) < J(u-1) < J(u-2) (J(u0) = 0) -> "val_decline", (c) u == max_updates -> "max_updates".
    -> (u, reason) or (None, None)."""
    for u in range(1, int(done_u) + 1):
        d, p = drift_by_u.get(u), drift_by_u.get(u - 1)
        if u >= 2 and d is not None and p is not None and d < -margin and p < -margin:
            return u, "length_drift"
        j, j1, j2 = j_by_u.get(u), j_by_u.get(u - 1), j_by_u.get(u - 2)
        if u >= 2 and None not in (j, j1, j2) and j < j1 < j2:
            return u, "val_decline"
        if u >= int(max_updates):
            return u, "max_updates"
    return None, None


def tau_alarms(tau, rl_gn_mean, aux_gn_mean):
    """§4.3 warnings (recorded, never acted on). -> list of strings."""
    out = []
    for src in ("task1", "task2"):
        v = (tau.get(src) or {}).get("tau")
        if v is not None and v > TAU_ALARM[src]:
            out.append("tau_%s %.4f > %.3f (2x the v17 range)" % (src, v, TAU_ALARM[src]))
    if rl_gn_mean is not None and aux_gn_mean:
        r = rl_gn_mean / aux_gn_mean
        if r > GRAD_RATIO_ALARM:
            out.append("rl_grad_norm / aux_grad_norm %.2f > %g" % (r, GRAD_RATIO_ALARM))
    if rl_gn_mean is not None and rl_gn_mean > RL_GRAD_ALARM:
        out.append("rl_grad_norm %.4f > %g" % (rl_gn_mean, RL_GRAD_ALARM))
    return out
