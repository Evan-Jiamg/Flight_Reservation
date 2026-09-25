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

TRAIN_DEFAULTS = {"lr": 1e-5, "kl_coef": 0.04}
TRAIN_BOUNDS = {"lr": (1e-7, 1e-4), "kl_coef": (0.0, 1.0)}
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
HISTORY_KEYS = ("update", "split", "n_episodes", "reward_mean", "reward_std", "components_mean",
                "end_kind_frac", "n_groups", "n_groups_skipped_zero_std", "loss", "kl", "ratio_mean",
                "clip_frac", "grad_norm", "n_tokens", "lr", "kl_coef", "value_mse", "ratio_init_maxdev")


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
        for k in _walk_keys(h):
            kl = k.lower()
            if any(p in kl for p in FORBIDDEN_KEY_PARTS):
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


def make_controller(kind, cfg0=None, log_path=None, transport=None, **kw):
    if kind == "fixed":
        return FixedController(cfg0)
    if kind == "dual":
        return DualAscentController(cfg0, **kw)
    if kind == "llm":
        return LLMController(cfg0, log_path=log_path, transport=transport, **kw)
    raise ValueError(kind)
