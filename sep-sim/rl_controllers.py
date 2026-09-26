"""Hyper-parameter / Lagrange controllers for Planner RL (pure python, no torch).

All controllers share
    .propose(history) -> cfg        cfg for the NEXT update (full dict: lr, kl_coef, reward weights, lambdas)
    .state_dict() / .load_state_dict(d)
`history` is a list of per-update TRAIN aggregates (rl_reward.aggregate output plus a few training
statistics). check_history() refuses any entry that is not marked split == "train" or that carries a
key naming validation/test data, so validation can never feed a controller.

  FixedController       returns the initial cfg forever.
  DualAscentController  lambda_k <- clip(lambda_k + eta * (rate_k - budget_k), 0, upper bound),
                        rate_k = mean per-step constraint rate of the LAST train update.
  LLMController         gpt-5-mini (reasoning_effort minimal) reads ONLY the train aggregates of the
                        last `window` updates and the current cfg + bounds, and proposes lr, kl_coef and
                        reward weights. Each proposal is clipped to a per-round factor (<= max_ratio,
                        relative to max(|current|, zero_ref[k])) and then to the declared bounds.
                        Every request and response is appended to a JSONL log with sha256 digests.
                        Transport is injectable (tests use a stub); failures keep the current cfg.
Every constant below is a declared config value with bounds and is logged by the trainer.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time

import rl_reward as RR

# w_aux: weight of the auxiliary stop-token supervision on the Task 1 positions (D2 anneals it further);
# a training knob like lr / kl_coef, not part of the reward (so the shadow / selection reward ignore it)
TRAIN_DEFAULTS = {"lr": 2e-5, "kl_coef": 0.04, "w_aux": 1.0}   # lr 2e-5: phase-0 KL at 1e-5 was ~1.1e-3/token
TRAIN_BOUNDS = {"lr": (1e-7, 1e-4), "kl_coef": (0.0, 1.0), "w_aux": (0.0, 10.0)}
CFG_BOUNDS = {**TRAIN_BOUNDS, **RR.REWARD_BOUNDS}

DUAL_DEFAULTS = {"eta": 1.0, "budget_unparsed": 0.02, "budget_hit_max_new": 0.02,
                 "budget_no_survivor": 0.05, "budget_judge_unknown": 0.02}
DUAL_BOUNDS = {"eta": (0.0, 100.0), "budget_unparsed": (0.0, 1.0), "budget_hit_max_new": (0.0, 1.0),
               "budget_no_survivor": (0.0, 1.0), "budget_judge_unknown": (0.0, 1.0)}

LLM_DEFAULTS = {"model": "gpt-5-mini", "reasoning_effort": "minimal", "window": 5, "max_ratio": 3.0,
                "keys": ["lr", "kl_coef", "w_goal", "w_partial", "w_over", "w_early"],
                "zero_ref": {"lr": 1e-6, "kl_coef": 1e-3, "w_goal": 0.05, "w_partial": 0.05,
                             "w_over": 0.05, "w_early": 0.05},
                "timeout_s": 120}
LLM_BOUNDS = {"window": (1, 50), "max_ratio": (1.0, 3.0)}
LLM_URL = "https://api.openai.com/v1/chat/completions"

FORBIDDEN_KEY_PARTS = ("valid", "val_", "test", "coverage", "complete")
HISTORY_KEYS = ("update", "split", "reward_version", "n_episodes", "reward_mean", "reward_std", "components_mean",
                "end_kind_frac", "n_groups", "n_groups_skipped_zero_std", "loss", "kl", "ratio_mean",
                "clip_frac", "grad_norm", "n_tokens", "lr", "kl_coef", "value_mse", "ratio_init_maxdev",
                "shadow_reward_mean", "turn_hist", "p_h", "aux_weight", "aux_stats", "task1_train", "rl_grad_norm")


def sha256(s):
    return hashlib.sha256(s.encode("utf-8") if isinstance(s, str) else s).hexdigest()


def initial_cfg(**over):
    cfg = {**copy.deepcopy(TRAIN_DEFAULTS), **RR.default_cfg()}
    cfg.update(over)
    return validate_cfg(cfg)


def validate_cfg(cfg):
    out = RR.validate(cfg)
    for k, (lo, hi) in TRAIN_BOUNDS.items():
        v = float(cfg.get(k, TRAIN_DEFAULTS[k]))
        if not (lo <= v <= hi):
            raise ValueError("cfg %s=%r outside declared bounds [%g, %g]" % (k, v, lo, hi))
        out[k] = v
    return out


def _check_bounds(d, bounds, name):
    for k, (lo, hi) in bounds.items():
        if k in d and not (lo <= float(d[k]) <= hi):
            raise ValueError("%s %s=%r outside declared bounds [%g, %g]" % (name, k, d[k], lo, hi))


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def check_history(history):
    """Every entry must be a TRAIN aggregate; no key may name validation/test/eval-only data."""
    for h in history:
        if h.get("split") != "train":
            raise ValueError("controller history entry is not a train aggregate: split=%r" % h.get("split"))
        # reward v3 uses the TRAIN rollouts' ledger coverage as a declared reward term, so under v3 the
        # train-aggregate key "coverage" is allowed; validation/test keys stay forbidden in every version
        parts = FORBIDDEN_KEY_PARTS if h.get("reward_version") not in ("v3", "v4") else \
            tuple(p for p in FORBIDDEN_KEY_PARTS if p != "coverage")
        for k in _walk_keys(h):
            kl = k.lower()
            if any(p in kl for p in parts):
                raise ValueError("controller history carries forbidden key %r" % k)
    return history


def sanitize(h):
    """Keep only declared train-aggregate keys (what an LLM controller may see)."""
    return {k: h[k] for k in HISTORY_KEYS if k in h}


class Controller:
    kind = "base"

    def __init__(self, cfg0=None):
        self.cfg = validate_cfg(cfg0 or initial_cfg())
        self.n_proposals = 0

    def propose(self, history):
        raise NotImplementedError

    def state_dict(self):
        return {"kind": self.kind, "cfg": copy.deepcopy(self.cfg), "n_proposals": self.n_proposals}

    def load_state_dict(self, d):
        if d.get("kind") != self.kind:
            raise ValueError("controller kind mismatch: %r vs %r" % (d.get("kind"), self.kind))
        self.cfg = validate_cfg(d["cfg"])
        self.n_proposals = int(d["n_proposals"])

    def describe(self):
        return {"kind": self.kind, "cfg": self.cfg, "cfg_bounds": CFG_BOUNDS}


class FixedController(Controller):
    kind = "fixed"

    def propose(self, history):
        check_history(history)
        self.n_proposals += 1
        return copy.deepcopy(self.cfg)


class DualAscentController(Controller):
    kind = "dual"

    def __init__(self, cfg0=None, **dual):
        super().__init__(cfg0)
        self.dual = {**DUAL_DEFAULTS, **dual}
        _check_bounds(self.dual, DUAL_BOUNDS, "dual")
        self.last_rates = None

    def propose(self, history):
        check_history(history)
        self.n_proposals += 1
        if not history:
            return copy.deepcopy(self.cfg)
        comp = history[-1]["components_mean"]
        rates = {k: float(comp["rate_" + k]) for k in RR.CONSTRAINTS}
        cfg = copy.deepcopy(self.cfg)
        for k in RR.CONSTRAINTS:
            key = "lambda_" + k
            lo, hi = RR.REWARD_BOUNDS[key]
            v = cfg[key] + self.dual["eta"] * (rates[k] - self.dual["budget_" + k])
            cfg[key] = min(hi, max(lo, max(0.0, v)))
        self.cfg, self.last_rates = validate_cfg(cfg), rates
        return copy.deepcopy(self.cfg)

    def state_dict(self):
        return {**super().state_dict(), "dual": dict(self.dual), "last_rates": self.last_rates}

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.dual, self.last_rates = dict(d["dual"]), d.get("last_rates")

    def describe(self):
        return {**super().describe(), "dual": self.dual, "dual_bounds": DUAL_BOUNDS}


def http_transport(request, timeout_s=120):
    """POST the chat-completions request; returns the parsed JSON response. Key from OPENAI_API_KEY."""
    import urllib.request
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    req = urllib.request.Request(LLM_URL, data=json.dumps(request).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


LLM_SYSTEM = (
    "You tune a reinforcement-learning run that trains a dialogue Planner. You see only aggregates of "
    "TRAINING rollouts from the most recent updates, the current configuration and the allowed bounds. "
    "Propose the configuration for the next update. Reply with one JSON object mapping each key you want "
    "to set to a number, plus a key \"rationale\" with one short sentence. Keys you may set: %s. "
    "Change any value by at most a factor of %g per round and stay inside the bounds.")


class LLMController(Controller):
    kind = "llm"

    def __init__(self, cfg0=None, log_path=None, transport=None, **llm):
        super().__init__(cfg0)
        self.llm = copy.deepcopy(LLM_DEFAULTS)
        self.llm.update(llm)
        _check_bounds(self.llm, LLM_BOUNDS, "llm")
        bad = [k for k in self.llm["keys"] if k not in CFG_BOUNDS or k.startswith("lambda_")]
        if bad:
            raise ValueError("LLM controller may not set %r" % bad)
        self.log_path = log_path
        self.transport = transport or (lambda req: http_transport(req, self.llm["timeout_s"]))
        self.n_failures = 0

    def build_request(self, history):
        window = [sanitize(h) for h in check_history(history)[-int(self.llm["window"]):]]
        payload = {"train_aggregates_last_rounds": window,
                   "current": {k: self.cfg[k] for k in self.llm["keys"]},
                   "bounds": {k: list(CFG_BOUNDS[k]) for k in self.llm["keys"]},
                   "fixed_lagrange_multipliers": {k: self.cfg[k] for k in self.cfg if k.startswith("lambda_")}}
        return {"model": self.llm["model"], "reasoning_effort": self.llm["reasoning_effort"],
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system",
                              "content": LLM_SYSTEM % (", ".join(self.llm["keys"]), self.llm["max_ratio"])},
                             {"role": "user", "content": json.dumps(payload, sort_keys=True)}]}

    def clip(self, key, current, proposed):
        """Per-round factor limit, then declared bounds. All controlled values are >= 0; a value at 0
        may grow to at most zero_ref[key] * max_ratio in one round."""
        r = float(self.llm["max_ratio"])
        if current > 0:
            lo_step, hi_step = current / r, current * r
        else:
            lo_step, hi_step = 0.0, float(self.llm["zero_ref"][key]) * r
        v = min(hi_step, max(lo_step, proposed))
        lo, hi = CFG_BOUNDS[key]
        return min(hi, max(lo, v))

    def _log(self, rec):
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    def propose(self, history):
        self.n_proposals += 1
        if not history:
            check_history(history)
            return copy.deepcopy(self.cfg)
        req = self.build_request(history)
        req_s = json.dumps(req, sort_keys=True)
        rec = {"time": time.time(), "proposal": self.n_proposals, "request": req,
               "request_sha256": sha256(req_s), "cfg_before": {k: self.cfg[k] for k in self.llm["keys"]}}
        try:
            resp = self.transport(req)
            resp_s = json.dumps(resp, sort_keys=True)
            rec.update(response=resp, response_sha256=sha256(resp_s))
            content = resp["choices"][0]["message"]["content"]
            prop = json.loads(content)
            if not isinstance(prop, dict):
                raise ValueError("proposal is not a JSON object")
            cfg = copy.deepcopy(self.cfg)
            applied = {}
            for k in self.llm["keys"]:
                if k in prop:
                    v = float(prop[k])
                    if not math.isfinite(v):
                        raise ValueError("non-finite proposal for %s" % k)
                    cfg[k] = self.clip(k, self.cfg[k], v)
                    applied[k] = {"proposed": v, "applied": cfg[k]}
            ignored = sorted(k for k in prop if k not in self.llm["keys"] and k != "rationale")
            self.cfg = validate_cfg(cfg)
            rec.update(ok=True, applied=applied, ignored_keys=ignored, rationale=prop.get("rationale"))
        except Exception as e:  # keep the current cfg, count and log the failure
            self.n_failures += 1
            rec.update(ok=False, error="%s: %s" % (type(e).__name__, e))
        rec["cfg_after"] = {k: self.cfg[k] for k in self.llm["keys"]}
        self._log(rec)
        return copy.deepcopy(self.cfg)

    def state_dict(self):
        return {**super().state_dict(), "llm": copy.deepcopy(self.llm), "n_failures": self.n_failures}

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.llm, self.n_failures = copy.deepcopy(d["llm"]), int(d["n_failures"])

    def describe(self):
        return {**super().describe(), "llm": self.llm, "llm_bounds": LLM_BOUNDS}


# ------------------------------------------------------------------ LLM controller for reward v4
# User decision (2026-09-25): reuse the LLM controller, adapted to the v4 reward, no baselines for now.
# Adaptations taken from the literature review (AHRS 2505.02483; ERFSL 2409.02428; LLMZero 2606.18388):
# discrete multiplicative steps instead of free numbers, a summarised (not raw) train log, a fixed-weight
# "shadow" reward so rounds stay comparable, and rollback after two worse decision points.
LLM4_DEFAULTS = {"model": "gpt-oss-120b", "every": 5, "window": 5, "rollback_windows": 2,
                 "factors": [0.5, 0.8, 1.0, 1.25, 2.0],
                 "keys": ["w_cov", "w_dist", "lambda_unparsed", "lambda_hit_max_new", "w_aux"],
                 "bounds": {"w_cov": [0.1, 5.0], "w_dist": [0.1, 5.0],
                            "lambda_unparsed": [0.1, 5.0], "lambda_hit_max_new": [0.1, 5.0],
                            "w_aux": [0.01, 5.0]},
                 "max_tokens": 4000, "timeout_s": 300}

LLM4_SYSTEM = (
    "You adjust the reward weights of a reinforcement-learning run. The policy is the Planner of a "
    "user simulator: each turn it plans the simulated user's next message and decides whether that "
    "message is the user's last. Reward = w_cov * coverage (share of the user's requirements the "
    "assistant addressed) + w_dist * dist (log p_human(T) - log q(T): how well the distribution of "
    "conversation lengths T matches real people's) - lambda_unparsed * (share of unreadable plans) "
    "- lambda_hit_max_new * (share of plans cut by the length cap). Separately, w_aux weights an auxiliary "
    "supervised loss that pulls the Planner's end_session decision towards the real person's on training "
    "conversations (it is also annealed to 0 later); both share one optimizer step: compare aux_grad_norm with "
    "rl_grad_norm (the RL part of the same step) to "
    "judge whether it dominates or is negligible, and task1_train accuracy to judge whether it is still needed. "
    "You see ONLY statistics of TRAINING "
    "rollouts, summarised per update, and your earlier decisions with what followed. "
    "For each weight choose one factor from %s (1.0 = keep). Keep changes small unless the statistics "
    "clearly call for them. Reply with ONE JSON object: {\"factors\": {<key>: <factor>, ...}, "
    "\"rationale\": \"<one sentence>\"}. Keys: %s.")


def local_transport(request, base_url, timeout_s=300):
    """POST an OpenAI-compatible chat request to a local endpoint (no key needed)."""
    import urllib.request
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("OPENAI_API_KEY")
    if key and "api.openai.com" in url:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(request).encode("utf-8"), method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def first_json_object(text):
    import re
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


class LLMFactorController(Controller):
    kind = "llm4"

    def __init__(self, cfg0=None, log_path=None, transport=None, **opts):
        super().__init__(cfg0)
        self.opt = copy.deepcopy(LLM4_DEFAULTS)
        self.opt.update(opts)
        bad = [k for k in self.opt["keys"] if k not in CFG_BOUNDS]
        if bad:
            raise ValueError("unknown controller keys %r" % bad)
        for k in self.opt["keys"]:
            lo, hi = self.opt["bounds"][k]
            if not (CFG_BOUNDS[k][0] <= lo <= hi <= CFG_BOUNDS[k][1]):
                raise ValueError("controller bounds for %s outside the reward bounds" % k)
            v = float(self.cfg[k])
            if not (lo <= v <= hi) and not (k == "w_aux" and v == 0.0):
                raise ValueError("initial %s=%g outside the controller bounds [%g, %g]: it would be clamped silently "
                                 "at the first decision" % (k, v, lo, hi))
        self.log_path = log_path
        base_url = os.environ.get("CONTROLLER_BASE_URL") or os.environ.get("R0_BASE_URL") or "http://127.0.0.1:8029/v1"
        self.opt["base_url"] = base_url
        self.transport = transport or (lambda req: local_transport(req, base_url, self.opt["timeout_s"]))
        self.pending = None           # {"prev_cfg", "baseline", "bad"} after an applied change
        self.decisions = []           # [{"at", "factors", "shadow_before"}] shown to the LLM
        self.n_failures = self.n_rollbacks = 0

    @staticmethod
    def _mean(xs):
        xs = [float(x) for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def summarise(self, history):
        win = history[-int(self.opt["window"]):]
        rows = []
        for h in win:
            c = h.get("components_mean") or {}
            rows.append({"update": h.get("update"), "reward_mean": round(float(h.get("reward_mean", 0.0)), 4),
                         "shadow_reward_mean": h.get("shadow_reward_mean"),
                         "coverage": c.get("coverage"), "dist": c.get("dist"), "turns_mean": c.get("turns"),
                         "rate_unparsed": c.get("rate_unparsed"), "rate_hit_max_new": c.get("rate_hit_max_new"),
                         "kl": h.get("kl"), "rl_grad_norm": h.get("rl_grad_norm"),
                         "aux_weight_effective": h.get("aux_weight"), "aux": h.get("aux_stats"),
                         "task1_train": {k: (h.get("task1_train") or {}).get(k)
                                         for k in ("acc", "end_at_final", "end_at_nonfinal")}})
        hist = [0] * len((win[-1].get("turn_hist") or []))
        for h in win:
            for i, v in enumerate(h.get("turn_hist") or []):
                hist[i] += v
        return {"per_update": rows, "train_turn_histogram_0_to_tmax": hist,
                "human_train_length_distribution_0_to_tmax": win[-1].get("p_h")}

    def build_request(self, history):
        payload = {"train_statistics": self.summarise(check_history(history)),
                   "current": {k: self.cfg[k] for k in self.opt["keys"]},
                   "bounds": {k: self.opt["bounds"][k] for k in self.opt["keys"]},
                   "earlier_decisions": self.decisions[-6:]}
        return {"model": self.opt["model"], "max_tokens": int(self.opt["max_tokens"]),
                "messages": [{"role": "system", "content": LLM4_SYSTEM % (self.opt["factors"], ", ".join(self.opt["keys"]))},
                             {"role": "user", "content": json.dumps(payload, sort_keys=True)}]}

    def _log(self, rec):
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    def propose(self, history):
        check_history(history)
        self.n_proposals += 1
        every = int(self.opt["every"])
        if not history or len(history) % every:
            return copy.deepcopy(self.cfg)
        u = history[-1].get("update")
        shadow = self._mean(h.get("shadow_reward_mean") for h in history[-every:])
        rec = {"time": time.time(), "update": u, "shadow_window_mean": shadow, "cfg_before": {k: self.cfg[k] for k in self.opt["keys"]}}
        # rollback: the last change made the fixed-weight shadow reward worse at two decision points
        if self.pending is not None and shadow is not None:
            self.pending["bad"] = self.pending["bad"] + 1 if shadow < self.pending["baseline"] else 0
            if self.pending["bad"] >= int(self.opt["rollback_windows"]):
                self.cfg = validate_cfg(self.pending["prev_cfg"])
                self.n_rollbacks += 1
                rec.update(ok=True, rollback=True, cfg_after={k: self.cfg[k] for k in self.opt["keys"]})
                self.pending = None
                self._log(rec)
                return copy.deepcopy(self.cfg)
        req = self.build_request(history)
        rec.update(request=req, request_sha256=sha256(json.dumps(req, sort_keys=True)))
        try:
            resp = self.transport(req)
            rec["response_sha256"] = sha256(json.dumps(resp, sort_keys=True))
            rec["finish_reason"] = resp["choices"][0].get("finish_reason")
            content = resp["choices"][0]["message"].get("content") or ""
            rec["response_content"] = content[:4000]
            if rec["finish_reason"] == "length":
                raise ValueError("controller reply cut by max_tokens=%s" % self.opt["max_tokens"])
            prop = first_json_object(content)
            facs = prop.get("factors") or {}
            if not isinstance(facs, dict):
                raise ValueError("factors is not an object")
            allowed = [float(f) for f in self.opt["factors"]]
            cfg, applied = copy.deepcopy(self.cfg), {}
            for k in self.opt["keys"]:
                if k not in facs:
                    continue
                f = float(facs[k])
                if f not in allowed:
                    raise ValueError("factor %r for %s is not one of %s" % (f, k, allowed))
                lo, hi = self.opt["bounds"][k]
                if self.cfg[k] == 0:
                    applied[k] = {"factor": f, "value": 0.0, "note": "switched off by the run; stays 0"}
                    continue
                cfg[k] = min(hi, max(lo, self.cfg[k] * f))
                applied[k] = {"factor": f, "value": cfg[k]}
            changed = any(abs(cfg[k] - self.cfg[k]) > 1e-12 for k in self.opt["keys"])
            if changed:
                if self.pending is not None and self.pending["bad"] > 0:
                    # an earlier change is still under suspicion: keep ITS last-good cfg, baseline and count, so
                    # a second worse point rolls back to before it even though the LLM changed something between
                    pass
                else:
                    self.pending = {"prev_cfg": copy.deepcopy(self.cfg), "baseline": shadow, "bad": 0}
            self.cfg = validate_cfg(cfg)
            self.decisions.append({"at_update": u, "factors": {k: v["factor"] for k, v in applied.items()},
                                   "shadow_reward_before": shadow})
            rec.update(ok=True, applied=applied, rationale=prop.get("rationale"), changed=changed)
        except Exception as e:                      # keep the current cfg; count and log
            self.n_failures += 1
            rec.update(ok=False, error="%s: %s" % (type(e).__name__, e))
        rec["cfg_after"] = {k: self.cfg[k] for k in self.opt["keys"]}
        self._log(rec)
        return copy.deepcopy(self.cfg)

    def state_dict(self):
        return {**super().state_dict(), "opt": copy.deepcopy(self.opt), "pending": copy.deepcopy(self.pending),
                "decisions": copy.deepcopy(self.decisions), "n_failures": self.n_failures, "n_rollbacks": self.n_rollbacks}

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.pending, self.decisions = copy.deepcopy(d.get("pending")), copy.deepcopy(d.get("decisions", []))
        self.n_failures, self.n_rollbacks = int(d.get("n_failures", 0)), int(d.get("n_rollbacks", 0))

    def describe(self):
        return {**super().describe(), "llm4": {k: v for k, v in self.opt.items()}}


def make_controller(kind, cfg0=None, log_path=None, transport=None, **kw):
    if kind == "fixed":
        return FixedController(cfg0)
    if kind == "dual":
        return DualAscentController(cfg0, **kw)
    if kind == "llm":
        if cfg0 is not None and cfg0.get("version") == "v4":
            return LLMFactorController(cfg0, log_path=log_path, transport=transport, **kw)
        return LLMController(cfg0, log_path=log_path, transport=transport, **kw)
    raise ValueError(kind)
