#!/usr/bin/env python3
"""Planner RL for the pend arm, SPEC v17 (user 2026-09-30): SFT warm-up -> short GRPO. Design: ops/SPEC_v17_sft_pend.md
(where it is silent, ops/SPEC_v16_grpo_opt.md and ops/AUDIT_SPEC_pend_grpo.md). The architecture is fixed: Planner =
Qwen3-4B-Instruct-2507 + LoRA (the only trained part), Speaker = Ditto-8B (frozen; NO end token: a session ends by the
Planner's end_session or a blank selected message) + Borda selector, Implicit Profile, few-shot. This trainer implements
v17 only; verify_pipeline.py keeps the v16 checks for archived runs.

Stage 0, SFT warm-up (sft_stage, before u0; §1):
  the start policy (seeded LoRA init, B1) is saved as ckpt/sft_e0 and served by vLLM; on every train_all conversation
  (never a forbidden id) at every real decision point t = 2..n (a point after a capped emitted message is skipped and
  counted) the start policy's greedy teacher-forced prompt gets 1 greedy + --sft-samples-per-point sampled plans (T 1,
  top-p 1, seeds seed_of(seed, "sft", cid, t, k)); every valid plan with a located value gives one example: the plan's
  own prefix, the value re-encoded with the human's decision (true iff t == n). Cached in sft_examples.jsonl +
  sft_examples_meta.json (reused on resume only when start policy / splits / code match); base_pend_train.jsonl = the
  start policy's teacher-forced P_end at every point (threshold_control.py). Loss = value_nll_loss (no class weights),
  own AdamW (--sft-lr), minibatches of 8, --sft-epochs-max epochs; candidates ckpt/sft_e<k> (k = 0 is the start
  policy), each probed greedily on validation_all (teacher-forced P_end) -> sft.jsonl; the candidate with the lowest
  validation NLL (then fewer invalid points, then the earlier epoch) becomes u0 = "SFT (u0)"; u0 is then loaded as the
  frozen adapter "ref", the KL reference of every update (§2).
Loop (update u = 1 .. --updates (SPEC 5, fixed); the policy that generates update u's rollouts has policy_version u-1):
  1. Task 2: --scenarios-per-update scenarios (seeded by (seed, u)) from splits[fold]["train"] ONLY, G rollouts each
     (Task2Env(arm="pend").run_episode, T 1); unclean episodes never enter a group; reward v4 (coverage + log p_h(T) -
     log q(T) - penalties, p_h from train_all), Dr. GRPO, stop credit (the length term's advantage on the end_session
     value tokens), fixed controller;
  2. Task 1 (§3.2): --task1-convs train_all conversations (seeded by (seed, u)), EVERY decision point t = 2..n, --task1-G
     sampled plans each from the current policy's greedy teacher-forced prompt; reward = Brier 1 - (P_end - y)^2 with
     P_end the rollout policy's teacher-forced P(end) on the sample's OWN prefix; a valid plan whose value cannot be
     located is dropped (no gradient), an invalid plan has R = 0; advantage R - mean(R) on the prefix tokens (valid) or
     on every token (invalid), never on the value tokens; plus the stop supervision (--aux-weight, one example per
     decision point, split across the minibatches);
  3. learner: epochs 2 x minibatches 4 = 8 optimizer steps, lr 1e-5, KL 0.01 (k3) to the ref adapter u0, TIS, clip 0.2;
  4. checkpoint EVERY update; updates.jsonl; length-drift monitor: two consecutive updates with mean(T - min(human
     turns, t_max)) over the clean group episodes < -(--length-drift-margin) stop the run (stop_reason length_drift);
  5. validation: u0 and the final update: Task 2 on validation x --val-seeds (0..7) + Task 1 on validation_all;
     u1 .. u(final-1): Task 1 only. No checkpoint selection: the final policy is the last completed update (final.json).
Validation never enters history, reward statistics or the controller; the test ids are never read.

--resume continues from ckpt/LATEST (or re-runs an unfinished SFT from its cached examples); rollouts of an interrupted
update that were produced by the same policy (policy_version and policy_sha match) are reused, the rest regenerated.
Code SHA256s must match the first launch unless --allow-code-change. --dry-run runs the whole loop with a fake env and
a pure-python learner (no torch, no GPU) to test the loop, checkpointing and resume.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rl_algos as RA  # noqa: E402
import rl_controllers as RC  # noqa: E402
import rl_reward as RR  # noqa: E402
import task1_stop as T1  # noqa: E402

SPEC_VERSION = "v17"
CODE_FILES = ("train_planner_rl.py", "rl_reward.py", "rl_controllers.py", "rl_algos.py", "task2_env.py",
              "task2_episode.py", "goal_judge.py", "planner_prompt_v3.py", "fit_prompts.py", "ditto_e16.py",
              "task1_stop.py", "batching.py", "implicit_profile.py", "style_select.py", "vllm_planner.py")
TIME_KEYS = ("time", "wall_s", "rollout_s", "update_s", "timing", "validation_s",
             "planner_s", "speaker_s", "r0_s", "ledger_s", "sft_s", "validated_time")
VAL_RETRIES = 2          # an unclean validation episode (infrastructure incident) is re-run up to this many times
# v17 B2: "updates" is NOT here (the run is 5 updates, fixed); the v16 re-selection flags are gone
RESUME_MAY_CHANGE = ("resume", "allow_code_change", "rollout_workers", "gpu", "max_batch",
                     "keep_optimizer_last", "dry_run_crash_after_episodes", "vllm_url", "intervention")
# SPEC v17 values (user 2026-09-30; S10: kept here, rl_controllers.TRAIN_DEFAULTS is unchanged)
SPEC_V17 = {"lr": 1e-5, "kl": 0.01, "sft_lr": 5e-5, "sft_epochs_max": 3, "sft_samples_per_point": 2, "task1_G": 4,
            "task1_convs": 8, "task1_reward": "brier", "task1_positions": "all", "aux_weight": 0.5, "updates": 5,
            "val_every": 1, "val_seeds": [0, 1, 2, 3, 4, 5, 6, 7], "length_drift_margin": 1.0, "controller": "fixed",
            "stop_credit": 1, "epochs": 2, "minibatches": 4}
SFT_BATCH = 8            # §1.2: SFT minibatch of 8 examples
SFT_TEMPERATURE = 1.0    # §1.1: the sampled SFT plans (T 1, top-p 1)
SFT_TOP_P = 1.0


# ------------------------------------------------------------------ small utilities
def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    with open(p, "rb") as f:
        return sha_bytes(f.read())


def sha_path(p):
    """File sha, or a combined sha over every file of a directory (sorted relative paths)."""
    if p is None:
        return None
    if os.path.isfile(p):
        return sha_file(p)
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(p)):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            h.update(os.path.relpath(fp, p).replace("\\", "/").encode())
            h.update(sha_file(fp).encode())
    return h.hexdigest()


def code_shas():
    return {f: sha_file(os.path.join(HERE, f)) for f in CODE_FILES if os.path.exists(os.path.join(HERE, f))}


def append_jsonl(path, row):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":                       # torn last line from a crash: drop it
                f.seek(0)
                data = f.read()
                f.seek(0)
                f.truncate(data.rfind(b"\n") + 1)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError("%s: undecodable line %d (not the last one): the file is corrupt" % (path, i + 1))
            # a torn last line after a crash; the row is regenerated
    return out


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def write_jsonl_atomic(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def sft_choice(rows):
    """v17 §1.3 / B5: the SFT candidate with the lowest validation nll, then the fewer invalid points, then the earlier
    epoch (an nll of None -- no valid point -- ranks last). rows: sft.jsonl epoch rows of one attempt. -> epoch."""
    key = lambda r: (float("inf") if r["val"]["nll"] is None else r["val"]["nll"], r["val"]["n_invalid"], r["epoch"])
    return min(rows, key=key)["epoch"]


def drift_decision(drift_by_update, margin, max_updates):
    """v17 §3.5 / S5: (final update, stop reason) from the per-update drift statistics {u: mean(T - min(human, t_max))}:
    the first u >= 2 with drift < -margin at u-1 and u -> (u, "length_drift"); else (max_updates, "max_updates") once
    that update exists; else (None, None) (still running)."""
    for u in sorted(drift_by_update):
        d, p = drift_by_update.get(u), drift_by_update.get(u - 1)
        if u >= 2 and d is not None and p is not None and d < -margin and p < -margin:
            return u, "length_drift"
        if u >= max_updates:
            return u, "max_updates"
    return None, None


# ------------------------------------------------------------------ leakage control
def load_split(path, fold):
    d = json.load(open(path, encoding="utf-8"))
    folds = {int(f["fold"]): f for f in d["folds"]}
    f = folds[int(fold)]
    train, val = list(f["train"]), list(f["validation"])
    train_all = list(f.get("train_all", train))
    val_all = list(f.get("validation_all", val))
    forbidden = set(f["forbidden_for_training"])
    assert train, "empty train split"
    assert not set(train) & forbidden, "train intersects forbidden_for_training"
    assert set(val) <= forbidden, "validation must be listed as forbidden for training"
    assert not set(train) & set(val), "train/validation overlap"
    assert not set(train_all) & forbidden, "train_all intersects forbidden_for_training"
    # v17 §4: Task 1 validation on validation_all (the 4 sessions of every fold)
    assert set(val_all) <= forbidden, "validation_all must be listed as forbidden for training"
    assert not set(val_all) & (set(train) | set(train_all)), "validation_all intersects train / train_all"
    assert set(val) <= set(val_all), "validation_all must contain validation"
    for k, n in (f.get("sizes") or {}).items():          # the declared sizes of the split file
        assert len(f[k]) == n, "splits fold %d: %s has %d ids, declared %d" % (fold, k, len(f[k]), n)
    # the test ids are not kept in memory at all (only through 'forbidden')
    return {"fold": int(fold), "train": train, "train_all": train_all, "validation": val, "validation_all": val_all,
            "forbidden": forbidden, "sha256": sha_file(path)}


def check_judge_manifest(adapter, split, strict=True):
    if adapter is None:
        return None
    mp = os.path.join(adapter, "train_manifest.json")
    m = json.load(open(mp, encoding="utf-8"))
    tr = set(m["train_scenarios"])
    assert not tr & split["forbidden"], "judge trained on forbidden (validation/test) scenarios: %s" % sorted(tr & split["forbidden"])[:5]
    ref = set(split["train"]) if strict else set(split["train_all"])
    assert tr <= ref, "judge train_scenarios not inside splits[%d].%s: %s" % (
        split["fold"], "train" if strict else "train_all", sorted(tr - ref)[:5])
    return {"manifest_sha256": sha_file(mp), "n_train_scenarios": len(tr), "strict_train": strict}


def assert_train_id(cid, split):
    assert cid in split["train"] and cid not in split["forbidden"], "non-train scenario %r in a training rollout" % cid


def assert_train_all_id(cid, split):
    assert cid in split["train_all"] and cid not in split["forbidden"], "non-train conversation %r in Task 1 / SFT" % cid


# ------------------------------------------------------------------ dry-run fakes (pure python)
STATUS_IDX = {"NOT ASSESSED": 0, "NOT": 1, "PARTIAL": 2, "SATISFIED": 3, "UNKNOWN": 4}
FAKE_ACT = 1          # every fake generation is [8, action, 7]: the end_session value is token 1 (stop_mask [0, 1, 0])


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


class FakeLearner:
    """Tabular stop policy: p(stop | status) = sigmoid(theta[status]); samples carry the status in prompt_ids[0] and the
    action (1 = stop) in gen_ids[1] (token 0 = the fake prefix, token 2 = the rest). Same interface as
    rl_algos.TorchLearner: multi-step updates with the aux split (v17 S3/S4), the S2 per-token advantage (recorded in
    self.token_log for the tests), a ref stub, p_end_batch, and the SFT steps."""

    def __init__(self, algo, acfg, lr, seed=0, lr_scale=1e4):
        self.algo, self.acfg, self.lr_scale = algo, RA.algo_cfg(**acfg), lr_scale
        self.theta = [-1.0, -1.5, -0.5, 0.0, -1.0]
        self.value = [0.0] * 5
        self.m = [0.0] * 5            # momentum = "optimizer state"
        self.has_ref, self.ref_theta, self.ref_path = False, None, None
        self.sft_lr = None
        self.token_log = []           # [(source, conversation_id, t, token advantages)] of the last update's first epoch

    def policy_sha(self):
        return sha_bytes(json.dumps([round(x, 12) for x in self.theta]).encode())

    def trainable_names(self):
        return ["theta.%d" % i for i in range(5)]

    def logp(self, s):
        p = _sig(self.theta[s["prompt_ids"][0]])
        return math.log(p if s["gen_ids"][FAKE_ACT] == 1 else 1 - p)

    def load_ref(self, path):
        """Stub of the frozen ref adapter: the theta of ckpt/u00000 (path = ckpt/u00000/adapter, as the torch learner)."""
        if self.has_ref:
            raise AssertionError("the ref adapter is already loaded")
        self.ref_theta = list(json.load(open(os.path.join(os.path.dirname(path), "fake_learner.json")))["theta"])
        self.has_ref, self.ref_path = True, path

    def prepare(self, samples):
        if samples and not self.has_ref:
            raise AssertionError("no KL reference loaded (v17: the SFT policy u0)")
        for s in samples:
            s["old_logp"] = self.logp(s)
        if self.algo == "ppo":
            adv = RA.ppo_advantages([s["ret"] for s in samples], [self.value[s["prompt_ids"][0]] for s in samples])
            if self.acfg["ppo_adv_norm"]:
                adv = RA.normalize(adv, self.acfg["adv_eps"])
            for s, a in zip(samples, adv):
                s["adv"] = a

    def _aux_grad(self, ex, g):
        """Adds the (ascent) gradient of the value NLL of these examples, normalised by their gen_len, to g. -> p list."""
        n_t = sum(int(x["gen_len"]) for x in ex)
        ps = []
        for x in ex:
            k = x["prompt_ids"][0]
            p = _sig(self.theta[k])
            ps.append(p if x["target_ids"][0] == 1 else 1 - p)
            g[k] += float(x["weight"]) * ((1 - p) if x["target_ids"][0] == 1 else -p) / n_t
        return ps

    def update(self, samples, cfg, seed, aux=None, aux_orders=None):
        samples, aux = list(samples or []), list(aux or [])
        if not samples and not aux:
            return {"n_samples": 0, "n_tokens": 0, "skipped_update": True, "optimizer_steps": 0}
        n_ep = int(self.acfg["epochs"])
        k = min(int(self.acfg["minibatches"]), len(samples)) if samples else 1
        if aux and samples and (aux_orders is None or len(aux_orders) != n_ep):
            raise ValueError("the stop supervision needs one example order per epoch")
        self.prepare(samples)
        lr, eps = cfg["lr"] * self.lr_scale, self.acfg["clip_eps"]
        st = {"n_samples": len(samples), "n_tokens": 3 * len(samples) * n_ep, "loss": 0.0, "kl": 0.0,
              "clip_frac": 0.0, "grad_norm": 0.0, "ratio_init_maxdev": 0.0, "optimizer_steps": 0,
              "n_minibatches": k if samples else 0, "aux_only": bool(aux and not samples)}
        if aux:
            ps = [(_sig(self.theta[x["prompt_ids"][0]]) if x["target_ids"][0] == 1 else 1 - _sig(self.theta[x["prompt_ids"][0]]))
                  for x in aux]
            st["aux_n"], st["aux_p_correct_before"] = len(aux), sum(ps) / len(ps)
        self.token_log, ratios, steps = [], [], []
        clipped = self.algo in ("grpo", "ppo") or self.acfg["rloo_clipped"]

        def step(g, rl_gn, aux_gn, n_aux):
            for i in range(5):
                self.m[i] = 0.9 * self.m[i] + g[i]
                self.theta[i] += lr * self.m[i]
            st["optimizer_steps"] += 1
            steps.append({"rl_grad_norm": rl_gn, "aux_grad_norm": aux_gn, "n_aux": n_aux, "kl": 0.0, "clip_frac": 0.0,
                          "grad_norm": math.sqrt(sum(x * x for x in g))})

        for ep in (range(n_ep) if samples else ()):
            parts = RA.aux_split(len(aux), aux_orders[ep], k) if aux else None
            for j, mb in enumerate(RA.minibatches(len(samples), k, seed * 1000 + ep)):
                g = [0.0] * 5
                for i in mb:
                    s = samples[i]
                    r = math.exp(self.logp(s) - s["old_logp"])
                    if st["optimizer_steps"] == 0:
                        st["ratio_init_maxdev"] = max(st["ratio_init_maxdev"], abs(r - 1))
                    ratios.append(r)
                    tok = RA.token_advantages(s, len(s["gen_ids"]))
                    if ep == 0:
                        self.token_log.append((s.get("source"), s.get("conversation_id"), s.get("t"), tok))
                    adv = tok[FAKE_ACT]          # the tabular policy only has the action token
                    kk = s["prompt_ids"][0]
                    p = _sig(self.theta[kk])
                    dlogp = (1 - p) if s["gen_ids"][FAKE_ACT] == 1 else -p
                    active = (not clipped) or RA.clipped_surrogate(r, adv, eps) == r * adv
                    if active:
                        g[kk] += adv * r * dlogp / len(mb)
                    if self.algo == "ppo":
                        self.value[kk] += 0.1 * (s["ret"] - self.value[kk]) / len(mb)
                rl_gn = math.sqrt(sum(x * x for x in g))
                aux_gn = None
                if parts and parts[j]:
                    g0 = list(g)
                    self._aux_grad([aux[i] for i in parts[j]], g)
                    aux_gn = math.sqrt(sum((a - b) ** 2 for a, b in zip(g, g0)))
                step(g, rl_gn, aux_gn, len(parts[j]) if parts else 0)
        if aux and not samples:                  # v17 S3: supervision only -> one step
            g = [0.0] * 5
            self._aux_grad(aux, g)
            step(g, 0.0, math.sqrt(sum(x * x for x in g)), len(aux))
        if st["ratio_init_maxdev"] > self.acfg["ratio_init_tol"]:
            raise AssertionError("off-policy start")
        st.update(RA.step_stats(steps))
        st["ratio_mean"] = (sum(ratios) / len(ratios)) if ratios else 1.0
        st["grad_norm"] = max(s_["grad_norm"] for s_ in steps) if steps else 0.0
        return st

    # ---- SFT (v17 §1.2)
    def sft_begin(self, lr):
        self.sft_lr = float(lr)

    def sft_step(self, batch):
        g = [0.0] * 5
        n_t = sum(int(x["gen_len"]) for x in batch)
        loss = 0.0
        for x in batch:
            p = _sig(self.theta[x["prompt_ids"][0]])
            q = p if x["target_ids"][0] == 1 else 1 - p
            loss += -float(x["weight"]) * math.log(max(q, 1e-12)) / n_t
        self._aux_grad(batch, g)
        for i in range(5):
            self.theta[i] += self.sft_lr * self.lr_scale * g[i]
        return {"loss": loss, "grad_norm": math.sqrt(sum(v * v for v in g)), "n": len(batch)}

    def sft_end(self):
        self.sft_lr = None

    def save(self, d):
        write_json_atomic(os.path.join(d, "fake_learner.json"),
                          {"theta": self.theta, "value": self.value, "m": self.m})
        self._stub_adapter(d)

    def save_adapter(self, d):
        write_json_atomic(os.path.join(d, "fake_learner.json"), {"theta": self.theta})
        self._stub_adapter(d)

    def _stub_adapter(self, d):
        """A stand-in d/adapter/adapter_model.safetensors (the policy only, like the torch learner's "default"-only
        save), so the verifier's adapter-directory checks (no ref/, u0 bytes == the chosen SFT candidate's) are
        exercised by the dry run (fix round 1, C-N6)."""
        os.makedirs(os.path.join(d, "adapter"), exist_ok=True)
        with open(os.path.join(d, "adapter", "adapter_model.safetensors"), "wb") as f:
            f.write(json.dumps([round(x, 12) for x in self.theta]).encode())

    def load(self, d):
        s = json.load(open(os.path.join(d, "fake_learner.json")))
        self.theta, self.value, self.m = s["theta"], s["value"], s["m"]

    def load_policy(self, d):
        self.theta = json.load(open(os.path.join(d, "fake_learner.json")))["theta"]

    def end_prob(self, x):
        return _sig(self.theta[x["prompt_ids"][0]])

    def p_end_batch(self, items):
        return [self.end_prob(x) for x in items]


class FakeEnv:
    """Synthetic Task 2 episodes with the Task2Env row schema. Goal status drifts NOT -> PARTIAL ->
    SATISFIED at a scenario-specific rate; the Planner (FakeLearner) decides end_session."""

    def __init__(self, learner, t_max=10):
        self.learner, self.t_max = learner, t_max

    def run_episode(self, conversation_id, seed, replicate=0, planner_temperature=0.0, planner_top_p=1.0,
                    record_generation=False):
        rng = random.Random(seed_of("fake", conversation_id, seed, replicate, planner_temperature))
        speed = 0.2 + (seed_of(conversation_id) % 60) / 100.0
        level, trace = 0, []
        end_kind = "t_max"
        for t in range(1, self.t_max + 1):
            if t == 1:
                gs = {"status": "NOT ASSESSED", "unmet": []}
            else:
                if rng.random() < speed:
                    level = min(2, level + 1)
                st = ("NOT", "PARTIAL", "SATISFIED")[level]
                q = 0.7 + 0.25 * rng.random()
                probs = {k: (q if k == st else (1 - q) / 2) for k in ("SATISFIED", "PARTIAL", "NOT")}
                gs = {"status": st, "unmet": [], "status_probs": probs}
                if rng.random() < 0.03:
                    gs = {"status": "UNKNOWN", "unmet": []}
            k = STATUS_IDX[gs["status"]]
            p = _sig(self.learner.theta[k])
            stop = ((rng.random() < p) if planner_temperature > 0 else p > 0.5) and t >= 2   # turn 1 cannot end
            unparsed = rng.random() < 0.02
            step = {"t": t, "goal_status": gs, "ended_planner": stop, "planner_unparsed": unparsed,
                    "planner_hit_max_new": False, "no_survivor": False, "planner_diag": {},
                    "stop_rule": "satiation" if stop and level else ("disgust" if stop and rng.random() < 0.3 else "none")}
            if record_generation:
                step["planner_gen"] = {"prompt_ids": [k, t], "gen_ids": [8, int(stop), 7], "stop_mask": [0, 1, 0],
                                       "temperature": planner_temperature, "top_p": planner_top_p, "seed": t}
            if stop:
                step.update(decision="planner_stop", emitted=False, user="", agent=None)
                trace.append(step)
                end_kind = "planner_stop"
                break
            step.update(decision="continue", emitted=True, user="u%d" % t, agent="a%d" % t)
            trace.append(step)
        emitted = sum(s["emitted"] for s in trace)
        return {"conversation_id": conversation_id, "record_id": "r_" + conversation_id, "seed": seed,
                "arm": "a2", "replicate": replicate, "emitted_user_turns": emitted, "decision_steps": len(trace),
                "end_kind": end_kind, "coverage": level / 2.0, "complete": level == 2, "trace": trace,
                "clean": True, "episode_counters": {}, "human_turns": self.human_turns(conversation_id)}

    def human_turns(self, conversation_id):
        return 2 + seed_of("human", conversation_id) % 6

    def run_task1(self, conversation_id, seed=0, keep_prompts=False):
        """Greedy teacher-forced stop decisions of the fake policy (theta[3] at the real last message,
        theta[1] before it); turn 1 cannot end."""
        n = self.human_turns(conversation_id)
        turns = [{"t": t, "n_real": n, "real_final": t == n, "speaker_blank": False, "planner_unparsed": False,
                  "ended_planner": t >= 2 and _sig(self.learner.theta[3 if t == n else 1]) > 0.5} for t in range(1, n + 1)]
        return {"conversation_id": conversation_id, "n_real": n, "end_mapping": "M2", "k1_speaker_blank": False,
                "turns": turns}


def _fake_task1_prompts(self, conversation_id):
    n = self.human_turns(conversation_id)
    return [{"t": t, "n_real": n, "real_final": t == n, "user_prompt": "fake", "emitted_capped": False}
            for t in range(1, n + 1)]


def _fake_task1_sample(self, conversation_id, t, user_prompt, real_final, G, temperature, top_p, seed):
    """Fake Task2Env.task1_sample: ~8% of the plans are not decisions (invalid), ~6% are valid decisions whose value
    could not be located (mask_ok False, dropped from the Brier groups); the rest carry prefix [8] and the value
    re-encoded as true [1] / false [0] (v17 S1). Greedy (temperature 0) plans are always valid."""
    out = []
    k = 3 if real_final else 1
    for g in range(G):
        rng = random.Random(seed_of("fake-t1", conversation_id, t, g, seed, temperature))
        p = _sig(self.learner.theta[k])
        stop = (rng.random() < p) if temperature > 0 else p > 0.5
        r = rng.random() if temperature > 0 else 1.0
        valid, mask_ok = r >= 0.08, r >= 0.14
        act = int(stop) if valid else 9
        out.append({"t": t, "replicate": g, "real_final": bool(real_final), "ended_planner": bool(stop and valid),
                    "planner_unparsed": not valid, "planner_hit_max_new": False, "decision_valid": valid,
                    "correct": float(valid and stop == bool(real_final)), "mask_ok": mask_ok,
                    "value_roundtrip_ok": True if mask_ok else None,
                    "prefix_ids": [8] if mask_ok else None, "target_true": [1] if mask_ok else None,
                    "target_false": [0] if mask_ok else None,
                    "planner_gen": {"prompt_ids": [k, t], "gen_ids": [8, act, 7],
                                    "stop_mask": [0, 1, 0] if mask_ok else None, "note_mask": None,
                                    "temperature": temperature, "top_p": top_p, "seed": g}})
    return out


def _fake_task1_end_probe(self, conversation_id, t, user_prompt, real_final):
    k = 3 if real_final else 1
    return {"valid": True, "decision_valid": True, "greedy_end": _sig(self.learner.theta[k]) > 0.5,
            "prompt_ids": [k, t], "prefix_ids": [8], "target_true": [1], "target_false": [0]}


def _fake_run_task1_prompts(self, conversation_id, seed=0, keep_prompts=False):
    out = FakeEnv._run_task1_plain(self, conversation_id, seed=seed)
    if keep_prompts:
        for x in out["turns"]:
            x["user_prompt"] = "fake"
    return out


FakeEnv.task1_prompts = _fake_task1_prompts
FakeEnv.task1_sample = _fake_task1_sample
FakeEnv.task1_end_probe = _fake_task1_end_probe
FakeEnv._run_task1_plain = FakeEnv.run_task1
FakeEnv.run_task1 = _fake_run_task1_prompts


def stub_llm_transport(request):
    """Deterministic offline stand-in for the LLM controller (dry run / tests)."""
    cur = json.loads(request["messages"][1]["content"])["current"]
    if "factors" in request["messages"][0]["content"]:            # v4 factor controller
        prop = {"factors": {k: 1.25 for k in cur}, "rationale": "stub"}
    else:
        prop = {k: v * 1.5 if v else 0.01 for k, v in cur.items()}
        prop["rationale"] = "stub"
    return {"choices": [{"message": {"content": json.dumps(prop)}}]}


def kl_div(q, p):
    """KL(q || p) of two distributions over 0..t_max (both smoothed, so p > 0)."""
    return sum(a * math.log(a / b) for a, b in zip(q, p) if a > 0)


# ------------------------------------------------------------------ trainer
class Trainer:
    def __init__(self, a):
        self.a = a
        os.makedirs(a.out, exist_ok=True)
        self.ckpt_root = os.path.join(a.out, "ckpt")
        os.makedirs(self.ckpt_root, exist_ok=True)
        self.p_roll = os.path.join(a.out, "rollouts.jsonl")
        self.p_roll_t1 = os.path.join(a.out, "rollouts_task1.jsonl")
        self.p_upd = os.path.join(a.out, "updates.jsonl")
        self.p_val = os.path.join(a.out, "validation.jsonl")
        self.p_final = os.path.join(a.out, "final.json")
        self.p_meta = os.path.join(a.out, "run_meta.jsonl")
        self.p_sft_ex = os.path.join(a.out, "sft_examples.jsonl")
        self.p_sft_meta = os.path.join(a.out, "sft_examples_meta.json")
        self.p_sft = os.path.join(a.out, "sft.jsonl")
        self.p_base_pend = os.path.join(a.out, "base_pend_train.jsonl")
        self.split = load_split(a.splits, a.fold)
        self.judge_info = check_judge_manifest(a.judge_adapter, self.split, strict=not a.judge_manifest_train_all)
        self.io_lock = threading.Lock()
        user_cfg = json.load(open(a.config, encoding="utf-8")) if a.config else {}
        # pend: reward v4 (D1(b): human length DISTRIBUTION matching) with format constraints on
        # (declared defaults; a --config file may override)
        base_reward = {"version": "v4", "lambda_unparsed": 1.0, "lambda_hit_max_new": 1.0} if a.arm == "pend" else {}
        # v17 (S9/S10): lr 1e-5, kl 0.01 from the trainer's SPEC table (the args), w_aux = --aux-weight (a constant)
        cfg0 = RC.initial_cfg(**{**base_reward, **user_cfg.get("reward", {}), "lr": a.lr, "kl_coef": a.kl,
                                 "w_aux": a.aux_weight})
        # fixed-weight cfg (the shadow reward and the validation reward; the name is kept from v16 for the eval tools,
        # v17 selects nothing on it)
        self.selection_cfg = copy.deepcopy(cfg0)
        self.acfg = RA.algo_cfg(**user_cfg.get("algo", {}))
        ctl_opts = dict(user_cfg.get(a.controller, {}))
        self.controller = RC.make_controller(
            a.controller, cfg0, log_path=os.path.join(a.out, "llm_controller.jsonl"),
            transport=stub_llm_transport if (a.dry_run and a.controller == "llm") else None, **ctl_opts)
        self.cfg = copy.deepcopy(cfg0)
        self.history, self.update_done = [], 0
        self.task1_base = None       # Task 1 metrics of SFT (u0) at its validation (the comparison base)
        self.stop_reason = None      # v17: "max_updates" | "length_drift" once final.json is written
        self.sft_info = None         # v17: the SFT warm-up's record (chosen epoch, shas), kept in ckpt u0
        self.p_h = None              # v4: smoothed human length distribution of the TRAIN conversations
        self.env = self.learner = None
        self.n_new_episodes = 0
        self.gen_name = None
        self.config_record = {"args": vars(a), "cfg0": cfg0, "selection_cfg": self.selection_cfg,
                              "algo_cfg": self.acfg, "algo_bounds": RA.ALGO_BOUNDS, "lora": RA.LORA,
                              "controller": self.controller.describe(), "user_config": user_cfg,
                              "spec_version": SPEC_VERSION, "spec_values": SPEC_V17}

    def gpu(self):
        """The env's GPU lock (learner forwards share the GPU with the Speaker / Planner calls of Task2Env)."""
        return getattr(self.env, "gpu_lock", None) or contextlib.nullcontext()

    # -------------------------------------------------------- setup
    def build(self):
        a = self.a
        if a.dry_run:
            self.learner = FakeLearner(a.algo, self.acfg, self.cfg["lr"], seed=a.seed)
            self.env = FakeEnv(self.learner)
            self.set_p_h()
            self.load_ref_if_u0()
            return
        import torch
        # fix round 1 (B-1): every CUDA allocation of this process (the learner, the ref adapter, Ditto) goes to --gpu;
        # no context is ever created on another GPU
        torch.cuda.set_device(int(a.gpu))
        from task2_env import PlannerLM, Task2Env
        planner = PlannerLM(a.planner_path, gpu=a.gpu, nf4=a.planner_nf4, dtype=a.planner_dtype,
                            adapter=None, trainable=False)
        # v17 B1: the LoRA init is seeded -> the same start policy (and sha) at every launch
        model = RA.setup_policy(planner, seed_of(a.seed, "lora_init"))
        planner.adapter = "rl:%s" % a.out
        if a.planner_backend == "vllm":
            # Planner generation by the vLLM server; the HF model stays the learner (it re-scores every token)
            import vllm_planner
            vllm_planner.VLLMPlanner(a.vllm_url).attach(planner)
        assert a.arm == "pend", "v17 implements the pend arm only (no goal judge)"
        self.env = Task2Env(arm=a.arm, gpu=a.gpu, planner=planner, judge=None, batch=bool(a.batch), max_batch=a.max_batch,
                            implicit_profile=bool(a.implicit_profile), selector=a.selector)
        if a.fewshot == "fold":
            from task2_env import make_fewshot_pool
            pool_ids = list(self.split["train_all"])
            assert not set(pool_ids) & self.split["forbidden"], "few-shot pool intersects validation/test"
            self.env.fewshot = make_fewshot_pool(self.env.recs, pool_ids)
        import task2_env as TE
        assert int(self.cfg["t_max"]) == TE.T_MAX, "reward t_max %r != environment T_MAX %d" % (self.cfg["t_max"], TE.T_MAX)
        self.set_p_h()
        self.learner = RA.TorchLearner(model, a.algo, self.acfg, lr=self.cfg["lr"], seed=a.seed)
        self.config_record["env"] = self.env.describe()
        self.config_record["trainable_params"] = self.learner.trainable_names()[:8] + ["..."]
        self.load_ref_if_u0()

    def load_ref_if_u0(self):
        """v17 B4: once u0 (the SFT policy) exists, it is the frozen "ref" adapter of the learner (resume and the test
        evaluation build; a fresh run loads it at the end of sft_stage). After the optimizer exists. Fix round 1 (A-2):
        only once LATEST.json exists -- a crash between the u0 rename and LATEST leaves an unfinished SFT, which the
        resume completes (sft_stage re-saves u0 and then loads the ref itself)."""
        if os.path.exists(os.path.join(self.ckpt_root, "LATEST.json")) and \
                os.path.exists(os.path.join(self.ckpt_dir(0), "state.json")):
            self.learner.load_ref(os.path.join(self.ckpt_dir(0), "adapter"))

    def set_p_h(self):
        """v4: p_h from the real people's number of messages in splits[fold].train_all ONLY (Task 2 length
        target; train_all has no requirement shards but its lengths are training data like any other)."""
        ids = sorted(self.split["train_all"])
        for cid in ids:
            assert cid not in self.split["forbidden"], "p_h would read a validation/test conversation"
        turns = [self.env.human_turns(cid) for cid in ids]
        self.p_h = RR.turn_distribution(turns, self.cfg["t_max"], self.selection_cfg["alpha_smooth"])
        self.config_record["p_h"] = {"source": "train_all", "n_conversations": len(ids), "dist": self.p_h,
                                     "alpha_smooth": self.selection_cfg["alpha_smooth"]}

    def reward_ctx(self, episodes, cfg):
        """v4 context: p_h (train, fixed) and q = smoothed length distribution of THESE episodes."""
        if cfg["version"] != "v4":
            return None
        q = RR.turn_distribution([e["emitted_user_turns"] for e in episodes], cfg["t_max"], cfg["alpha_smooth"])
        return {"p_h": self.p_h, "q": q}

    def aux_weight(self):
        """v17 §3.3 / S9 (user 2026-09-30): the stop-supervision weight is the constant --aux-weight (SPEC 0.5); no D2
        trigger, no anneal, no floor, not tuned by the controller."""
        return float(self.a.aux_weight)

    def u0_state(self):
        p = os.path.join(self.ckpt_dir(0), "state.json")
        return json.load(open(p)) if os.path.exists(p) else None

    def meta(self, kind):
        a = self.a
        assert getattr(a, "init_adapter", None) is None, "v17: no init adapter"
        st0 = self.u0_state()
        sft = (st0 or {}).get("sft") or {}
        row = {"kind": kind, "time": time.time(), "code_sha256": code_shas(),
               "config_sha256": sha_bytes(json.dumps(self.config_record, sort_keys=True, default=str).encode()),
               "config": self.config_record, "splits_sha256": self.split["sha256"], "fold": a.fold,
               "judge_manifest": self.judge_info,
               "judge_adapter_sha256": None if a.dry_run else sha_path(a.judge_adapter),
               "init_adapter_sha256": None,
               "config_file_sha256": sha_file(a.config) if a.config else None,
               # v17 §8 provenance: the SFT arguments, the cached examples, u0, the chosen epoch, the KL reference
               "spec_version": SPEC_VERSION,
               "sft_args": {"sft_lr": a.sft_lr, "sft_epochs_max": a.sft_epochs_max,
                            "sft_samples_per_point": a.sft_samples_per_point},
               "sft_examples_sha256": sha_file(self.p_sft_ex) if os.path.exists(self.p_sft_ex) else None,
               "u0_policy_sha": (st0 or {}).get("policy_sha"),
               "sft_chosen_epoch": sft.get("chosen_epoch"),
               "ref_policy_sha": (st0 or {}).get("policy_sha") if getattr(self.learner, "has_ref", False) else None}
        return row

    def check_provenance(self, row):
        prev = read_jsonl(self.p_meta)
        if not prev:
            return
        first = prev[0]
        if first.get("spec_version") != SPEC_VERSION:
            raise SystemExit("run %s was written by spec %r; this trainer implements %s only"
                             % (self.a.out, first.get("spec_version"), SPEC_VERSION))
        for k in ("code_sha256", "splits_sha256", "judge_adapter_sha256", "init_adapter_sha256", "config_file_sha256"):
            if first.get(k) != row.get(k) and not self.a.allow_code_change:
                raise SystemExit("provenance mismatch on resume (%s); rerun in a new --out or pass --allow-code-change" % k)
        for k in sorted(set(first["config"]["args"]) | set(row["config"]["args"])):
            if k in RESUME_MAY_CHANGE:
                continue
            if first["config"]["args"].get(k) != row["config"]["args"].get(k):
                raise SystemExit("argument %s changed on resume (%r -> %r)" % (k, first["config"]["args"].get(k),
                                                                             row["config"]["args"].get(k)))

    # -------------------------------------------------------- checkpoint
    def ckpt_dir(self, u):
        return os.path.join(self.ckpt_root, "u%05d" % u)

    def sft_dir(self, k):
        return os.path.join(self.ckpt_root, "sft_e%d" % k)

    def save_checkpoint(self, u, update_row):
        d, tmp = self.ckpt_dir(u), self.ckpt_dir(u) + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        self.learner.save(tmp)
        rng = {"python": random.getstate()}
        if not self.a.dry_run:
            import torch
            torch.save(RA.rng_state(), os.path.join(tmp, "rng.pt"))
        state = {"update": u, "policy_version": u, "policy_sha": self.learner.policy_sha(),
                 "cfg": self.cfg, "controller": self.controller.state_dict(), "history": self.history,
                 "update_row": update_row, "task1_base": self.task1_base, "stop_reason": self.stop_reason,
                 "sft": self.sft_info if u == 0 else None, "spec_version": SPEC_VERSION,
                 "python_rng": [rng["python"][0], list(rng["python"][1]), rng["python"][2]]}
        write_json_atomic(os.path.join(tmp, "state.json"), state)
        write_json_atomic(os.path.join(tmp, "rl_manifest.json"), self.manifest())
        if os.path.exists(d):             # only possible if LATEST was not advanced after a crash
            shutil.rmtree(d)
        os.replace(tmp, d)
        write_json_atomic(os.path.join(self.ckpt_root, "LATEST.json"), {"update": u, "dir": os.path.basename(d)})
        if not self.a.dry_run and self.a.keep_optimizer_last > 0:
            old = self.ckpt_dir(u - self.a.keep_optimizer_last)
            for fn in ("optimizer.pt",):
                if u - self.a.keep_optimizer_last >= 1 and os.path.exists(os.path.join(old, fn)):
                    os.remove(os.path.join(old, fn))   # adapters are kept for every update

    def update_state(self, u, **kv):
        """Persist fields into ckpt u's state.json now (a crash must not lose them)."""
        sp = os.path.join(self.ckpt_dir(u), "state.json")
        st = json.load(open(sp))
        st.update(kv)
        write_json_atomic(sp, st)

    def manifest(self):
        """What this policy was trained on (checked by the evaluation CLIs before any validation/test run)."""
        a = self.a
        return {"kind": "planner_rl", "spec_version": SPEC_VERSION, "fold": self.split["fold"],
                "splits_sha256": self.split["sha256"],
                "train_scenarios": sorted(self.split["train"]), "train_conversations": sorted(self.split["train_all"]),
                "sft_conversations": sorted(self.split["train_all"]),
                "fewshot_pool": sorted(self.split["train_all"]) if a.fewshot == "fold" else [],
                "p_h_source": "train_all",
                "validation_used_for": "the SFT epoch choice (Task 1 NLL on validation_all) and reporting; "
                                       "no checkpoint selection",
                "planner_path": a.planner_path, "init_adapter": None, "init_adapter_sha256": None,
                "sft_examples_sha256": sha_file(self.p_sft_ex) if os.path.exists(self.p_sft_ex) else None,
                "sft_args": {"sft_lr": a.sft_lr, "sft_epochs_max": a.sft_epochs_max,
                             "sft_samples_per_point": a.sft_samples_per_point},
                "sft_chosen_epoch": (self.sft_info or {}).get("chosen_epoch"),
                "ref_policy_sha": (self.sft_info or {}).get("policy_sha"),
                "arm": a.arm, "implicit_profile": a.implicit_profile, "fewshot": a.fewshot, "selector": a.selector,
                "planner_backend": a.planner_backend, "tis_cap": self.acfg["tis_cap"]}

    def load_checkpoint(self):
        p = os.path.join(self.ckpt_root, "LATEST.json")
        if not os.path.exists(p):
            return False
        u = json.load(open(p))["update"]
        d = self.ckpt_dir(u)
        st = json.load(open(os.path.join(d, "state.json")))
        self.learner.load(d)
        if self.learner.policy_sha() != st["policy_sha"]:
            raise AssertionError("restored policy sha differs from the checkpoint record")
        self.controller.load_state_dict(st["controller"])
        self.cfg, self.history, self.update_done = st["cfg"], st["history"], u
        self.task1_base = st.get("task1_base")
        self.stop_reason = st.get("stop_reason")
        self.sft_info = (self.u0_state() or {}).get("sft")
        py = st["python_rng"]
        random.setstate((py[0], tuple(py[1]), py[2]))
        if not self.a.dry_run:
            import torch
            RA.set_rng_state(torch.load(os.path.join(d, "rng.pt"), weights_only=False))
        # reconcile updates.jsonl (crash between checkpoint and log)
        logged = {r["update"] for r in read_jsonl(self.p_upd)}
        if st["update_row"] is not None and u not in logged:
            append_jsonl(self.p_upd, st["update_row"])
        return True

    # -------------------------------------------------------- generation policy
    def sync_generation_policy(self, pv=None, path=None, tag=None):
        """v17 B3: the vLLM server generates with the adapter at `path` (default: checkpoint pv's) under the name
        "<tag>-<sha12 of the adapter directory>" (tag default "p<pv>"; the SFT candidates use "sft_e<k>"); every
        generation records that name. The learner must hold the same policy (the caller's responsibility)."""
        if path is None:
            path, tag = os.path.join(self.ckpt_dir(pv), "adapter"), "p%d" % pv
        remote = getattr(getattr(self.env, "planner", None), "remote", None)
        if remote is None:
            self.gen_name = None
            return None
        self.gen_name = remote.use_adapter(path, tag)
        return self.gen_name

    # -------------------------------------------------------- SFT warm-up (v17 §1)
    def save_sft_candidate(self, k):
        """ckpt/sft_e<k>: the learner's current policy adapter (k = 0: the start policy) + its policy sha."""
        d, tmp = self.sft_dir(k), self.sft_dir(k) + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        self.learner.save_adapter(tmp)          # the torch learner writes tmp/adapter (the "default" adapter only)
        write_json_atomic(os.path.join(tmp, "state.json"), {"epoch": k, "policy_sha": self.learner.policy_sha(),
                                                            "time": time.time()})
        if os.path.exists(d):
            shutil.rmtree(d)
        os.replace(tmp, d)
        return d

    def sft_examples(self, start_sha):
        """v17 §1.1 (user 2026-09-30): the SFT examples of every train_all decision point, generated by the START
        policy (served as ckpt/sft_e0), plus the start policy's teacher-forced P_end at every point
        (base_pend_train.jsonl). Cached: reused only when the cache key (start policy sha, splits, code, seed, plan
        settings) matches; a mismatching cache is never reused and never overwritten. -> (examples, meta)."""
        a = self.a
        key = {"start_policy_sha": start_sha, "splits_sha256": self.split["sha256"], "code_sha256": code_shas(),
               "seed": a.seed, "sft_samples_per_point": a.sft_samples_per_point,
               "sample_temperature": SFT_TEMPERATURE, "sample_top_p": SFT_TOP_P}
        if os.path.exists(self.p_sft_meta):
            m = json.load(open(self.p_sft_meta, encoding="utf-8"))
            bad = [k for k in key if m.get("key", {}).get(k) != key[k]]
            if not bad and os.path.exists(self.p_sft_ex) and sha_file(self.p_sft_ex) == m["examples_sha256"] \
                    and os.path.exists(self.p_base_pend) and sha_file(self.p_base_pend) == m["base_pend_sha256"]:
                return read_jsonl(self.p_sft_ex), m
            raise SystemExit("the SFT example cache in %s does not match this run (%s): not reused, not overwritten"
                             % (a.out, bad or "file sha"))
        t0 = time.time()
        self.sync_generation_policy(path=os.path.join(self.sft_dir(0), "adapter"), tag="sft_e0")
        cids = sorted(self.split["train_all"])
        for cid in cids:
            assert_train_all_id(cid, self.split)

        def one_conv(cid):
            n = self.env.human_turns(cid)
            ex, base, pts, cnt = [], [], [], {"unparsed": 0, "hit_max_new": 0, "invalid_value": 0, "mask_not_found": 0,
                                              "value_roundtrip_fail": 0}
            if n < 2:
                return ex, base, pts, cnt           # turn 1 never ends: no decision point
            prompts = self.env.task1_prompts(cid)
            assert len(prompts) == n
            for t in range(2, n + 1):
                if any(p.get("emitted_capped") for p in prompts[: t - 1]):
                    pts.append([cid, t, n, "skipped_capped_history", 0])
                    continue
                pr = prompts[t - 1]
                assert pr["t"] == t and pr["real_final"] == (t == n)
                plans = []
                for k in range(1 + a.sft_samples_per_point):          # N1: k = 0 greedy, then the samples
                    temp = 0.0 if k == 0 else SFT_TEMPERATURE
                    x = self.env.task1_sample(cid, t, pr["user_prompt"], pr["real_final"], 1, temp, SFT_TOP_P,
                                              seed_of(a.seed, "sft", cid, t, k))[0]
                    plans.append(x)
                n_ex = 0
                for k, x in enumerate(plans):
                    g = x["planner_gen"]
                    if not x["decision_valid"]:
                        cnt["unparsed" if x.get("planner_unparsed") else
                            "hit_max_new" if x.get("planner_hit_max_new") else "invalid_value"] += 1
                        continue
                    if not x.get("mask_ok"):
                        cnt["mask_not_found"] += 1
                        continue
                    # D-N7: the re-encoded value of the plan's OWN decision differs from the sampled value tokens
                    cnt["value_roundtrip_fail"] += int(x.get("value_roundtrip_ok") is False)
                    ex.append({"conversation_id": cid, "t": t, "n": n, "real_final": t == n,
                               "kind": "greedy" if k == 0 else "sample", "k": k,
                               "seed": seed_of(a.seed, "sft", cid, t, k),
                               "prompt_ids": list(g["prompt_ids"]), "prefix_ids": list(x["prefix_ids"]),
                               "target_ids": list(x["target_true"] if t == n else x["target_false"]),
                               "target_true": list(x["target_true"]), "target_false": list(x["target_false"]),
                               "gen_len": len(g["gen_ids"]), "gen_adapter": g.get("gen_adapter")})
                    n_ex += 1
                pts.append([cid, t, n, "ok", n_ex])
                # base P_end (§1.1 item 6): the greedy plan = task1_end_probe's, scored by the start policy
                g0, x0 = plans[0]["planner_gen"], plans[0]
                if x0["decision_valid"] and x0.get("mask_ok"):
                    with self.gpu():
                        pe = float(self.learner.p_end_batch([{"prompt_ids": g0["prompt_ids"], "prefix_ids": x0["prefix_ids"],
                                                              "target_true": x0["target_true"],
                                                              "target_false": x0["target_false"]}])[0])
                    valid = True
                else:
                    pe = 1.0 if (x0["decision_valid"] and x0["ended_planner"]) else 0.0     # the unscored-point rule
                    valid = False
                base.append({"conversation_id": cid, "t": t, "n": n, "real_final": t == n, "p_end": pe, "valid": valid,
                             "decision_valid": bool(x0["decision_valid"]), "greedy_end": bool(x0["ended_planner"]),
                             "gen_adapter": g0.get("gen_adapter"), "policy_sha": start_sha})
            return ex, base, pts, cnt

        if a.rollout_workers <= 1:
            res = [one_conv(c) for c in cids]
        else:
            with ThreadPoolExecutor(max_workers=a.rollout_workers) as ex_:
                res = list(ex_.map(one_conv, cids))
        examples = sorted((e for r in res for e in r[0]), key=lambda e: (e["conversation_id"], e["t"], e["k"]))
        base = sorted((b for r in res for b in r[1]), key=lambda b: (b["conversation_id"], b["t"]))
        points = sorted((p for r in res for p in r[2]), key=lambda p: (p[0], p[1]))
        counts = {k: sum(r[3][k] for r in res) for k in ("unparsed", "hit_max_new", "invalid_value", "mask_not_found",
                                                          "value_roundtrip_fail")}
        counts.update(n_conversations=len(cids), n_points=sum(1 for p in points if p[3] == "ok"),
                      n_points_skipped_capped=sum(1 for p in points if p[3] != "ok"),
                      n_plans=sum(1 for p in points if p[3] == "ok") * (1 + a.sft_samples_per_point),
                      n_examples=len(examples), n_examples_greedy=sum(1 for e in examples if e["kind"] == "greedy"),
                      n_examples_sample=sum(1 for e in examples if e["kind"] == "sample"))
        if not examples:
            raise SystemExit("SFT: no valid plan with a located value on train_all (counts %r)" % counts)
        write_jsonl_atomic(self.p_sft_ex, examples)
        write_jsonl_atomic(self.p_base_pend, base)
        # fix round 2 (A R2-1): the length of EVERY train_all conversation, also those with one message (no point)
        n_by_conv = {cid: int(self.env.human_turns(cid)) for cid in cids}
        m = {"key": key, "counts": counts, "points": points, "n_by_conv": n_by_conv,
             "examples_sha256": sha_file(self.p_sft_ex),
             "base_pend_sha256": sha_file(self.p_base_pend), "gen_adapter": self.gen_name,
             "conversations": cids, "sft_s": round(time.time() - t0, 1), "time": time.time()}
        write_json_atomic(self.p_sft_meta, m)          # written last: its presence marks a complete cache
        return examples, m

    def sft_probe(self, attempt, k, examples, steps):
        """One SFT candidate (the learner holds it; vLLM serves ckpt/sft_e<k>, S8): the train P_end on the examples and
        the greedy Task 1 run + teacher-forced P_end on validation_all -> an sft.jsonl epoch row."""
        a = self.a
        psha = self.learner.policy_sha()
        cst = json.load(open(os.path.join(self.sft_dir(k), "state.json")))
        assert cst["policy_sha"] == psha, "SFT candidate %d: the learner does not hold the saved adapter" % k
        name = self.sync_generation_policy(path=os.path.join(self.sft_dir(k), "adapter"), tag="sft_e%d" % k)
        with self.gpu():
            pe = self.learner.p_end_batch([{"prompt_ids": e["prompt_ids"], "prefix_ids": e["prefix_ids"],
                                            "target_true": e["target_true"], "target_false": e["target_false"]}
                                           for e in examples])
        fin = [p for p, e in zip(pe, examples) if e["real_final"]]
        mid = [p for p, e in zip(pe, examples) if not e["real_final"]]
        ids = sorted(self.split["validation_all"])
        for cid in ids:
            assert cid in self.split["forbidden"] and cid not in self.split["train_all"], "validation_all id %r" % cid
        with ThreadPoolExecutor(max_workers=max(1, a.rollout_workers)) as ex:
            rows = list(ex.map(lambda c: self.task1_eval_row(None, psha, c), ids))
        pts = [p for r in rows for p in r["end_probs"]]
        m = T1.task1_prob_metrics(pts)
        row = {"kind": "epoch", "attempt": attempt, "epoch": k, "policy_sha": psha, "gen_adapter": name,
               "train_steps": len(steps), "train_loss": (sum(s["loss"] for s in steps) / len(steps)) if steps else None,
               "train_grad_norm_max": max(s["grad_norm"] for s in steps) if steps else None,
               "train_p_end_final_mean": (sum(fin) / len(fin)) if fin else None,
               "train_p_end_nonfinal_mean": (sum(mid) / len(mid)) if mid else None,
               "val": {k_: m[k_] for k_ in ("nll", "bal_p", "auc", "n_points", "n_invalid", "n_final")},
               "val_ids": ids, "val_end_probs": {r["conversation_id"]: r["end_probs"] for r in rows},
               "val_n_turns": {r["conversation_id"]: len(r["task1"]["turns"]) for r in rows}, "time": time.time()}
        append_jsonl(self.p_sft, row)
        return row

    def sft_stage(self):
        """v17 §1 (user 2026-09-30): the SFT warm-up before u0 (see the module docstring). Ends with u0 = the chosen
        candidate saved as ckpt/u00000 (LATEST) and loaded as the frozen KL reference "ref". An interrupted SFT is re-run
        from its cached examples (a new attempt in sft.jsonl); u0 / LATEST are written only when it is complete."""
        a = self.a
        assert not os.path.exists(os.path.join(self.ckpt_root, "LATEST.json")), "u0 exists: SFT never runs again"
        t0 = time.time()
        start_sha = self.learner.policy_sha()
        self.save_sft_candidate(0)                 # B3: the start policy as an adapter vLLM can serve
        examples, emeta = self.sft_examples(start_sha)
        attempt = 1 + sum(1 for r in read_jsonl(self.p_sft) if r.get("kind") == "attempt_start")
        append_jsonl(self.p_sft, {"kind": "attempt_start", "attempt": attempt, "start_policy_sha": start_sha,
                                  "sft_examples_sha256": emeta["examples_sha256"], "n_examples": len(examples),
                                  "sft_lr": a.sft_lr, "sft_epochs_max": a.sft_epochs_max, "batch": SFT_BATCH,
                                  "time": time.time()})
        rows = [self.sft_probe(attempt, 0, examples, [])]
        self.learner.sft_begin(a.sft_lr)
        for ep in range(1, a.sft_epochs_max + 1):
            order = list(range(len(examples)))
            random.Random(seed_of(a.seed, "sft_shuffle", ep)).shuffle(order)
            steps = []
            for i in range(0, len(order), SFT_BATCH):
                batch = [dict(examples[j], weight=1.0) for j in order[i:i + SFT_BATCH]]
                with self.gpu():
                    steps.append(self.learner.sft_step(batch))
            self.save_sft_candidate(ep)
            rows.append(self.sft_probe(attempt, ep, examples, steps))
        self.learner.sft_end()
        k = sft_choice(rows)
        chosen = next(r for r in rows if r["epoch"] == k)
        self.learner.load_policy(self.sft_dir(k))
        if self.learner.policy_sha() != chosen["policy_sha"]:
            raise AssertionError("SFT: the loaded candidate %d does not have its recorded sha" % k)
        append_jsonl(self.p_sft, {"kind": "selection", "attempt": attempt, "chosen_epoch": k,
                                  "policy_sha": chosen["policy_sha"],
                                  "rule": "lowest validation_all nll, then fewer n_invalid, then the earlier epoch",
                                  "candidates": {str(r["epoch"]): [r["val"]["nll"], r["val"]["n_invalid"]] for r in rows},
                                  "u0_is_start_policy": k == 0, "time": time.time()})
        self.sft_info = {"attempt": attempt, "chosen_epoch": k, "policy_sha": chosen["policy_sha"],
                         "start_policy_sha": start_sha, "sft_examples_sha256": emeta["examples_sha256"],
                         "n_examples": len(examples), "u0_is_start_policy": k == 0,
                         "sft_s": round(time.time() - t0, 1)}
        self.save_checkpoint(0, None)              # policy_version 0 = SFT (u0); LATEST
        self.learner.load_ref(os.path.join(self.ckpt_dir(0), "adapter"))
        if self.learner.policy_sha() != chosen["policy_sha"]:
            raise AssertionError("loading the ref adapter changed the policy")
        append_jsonl(self.p_meta, {**self.meta("sft"), "sft_info": self.sft_info})
        print(json.dumps({"sft": {k_: self.sft_info[k_] for k_ in ("chosen_epoch", "n_examples", "u0_is_start_policy")},
                          "candidates": {r["epoch"]: r["val"]["nll"] for r in rows}}), flush=True)
        return self.sft_info

    # -------------------------------------------------------- rollouts
    def scenarios_for(self, u):
        rng = random.Random(seed_of(self.a.seed, "scenarios", u))
        k = self.a.scenarios_per_update
        pool = sorted(self.split["train"])
        if k <= len(pool):
            return rng.sample(pool, k)
        return [rng.choice(pool) for _ in range(k)]

    def rollouts(self, u):
        a, pv, psha = self.a, u - 1, self.learner.policy_sha()
        self.sync_generation_policy(pv)
        reuse = {}
        aborted = os.path.exists(os.path.join(a.out, "ABORTED_u%05d.json" % u))
        for r in read_jsonl(self.p_roll):
            if not aborted and r["update"] == u and r["policy_version"] == pv and r["policy_sha"] == psha:
                reuse[(r["slot"], r["replicate"])] = r
        cids = self.scenarios_for(u)
        for cid in cids:
            assert_train_id(cid, self.split)
        rows = {}
        todo = []
        for slot, cid in enumerate(cids):
            for g in range(a.G):
                key = (slot, g)
                if key in reuse:
                    assert reuse[key]["conversation_id"] == cid
                    rows[key] = reuse[key]
                else:
                    todo.append((slot, g, cid))

        def run_one(job):
            slot, g, cid = job
            t0 = time.time()
            ep = self.env.run_episode(cid, seed=seed_of(a.seed, "episode", u, slot) % 100000, replicate=g,
                                      planner_temperature=a.temperature, planner_top_p=a.top_p,
                                      record_generation=True)
            row = {"update": u, "slot": slot, "replicate": g, "conversation_id": cid, "split": "train",
                   "policy_version": pv, "policy_sha": psha, "cfg_sha256": RR.cfg_sha(self.cfg),
                   "time": time.time(), "rollout_s": time.time() - t0, "episode": ep}
            with self.io_lock:
                append_jsonl(self.p_roll, row)
                self.n_new_episodes += 1
                if a.dry_run_crash_after_episodes and self.n_new_episodes >= a.dry_run_crash_after_episodes:
                    raise SystemExit("dry-run: simulated crash after %d new episodes" % self.n_new_episodes)
            return (slot, g), row

        if a.rollout_workers <= 1:
            for job in todo:
                k, r = run_one(job)
                rows[k] = r
        else:
            with ThreadPoolExecutor(max_workers=a.rollout_workers) as ex:
                for k, r in ex.map(run_one, todo):
                    rows[k] = r
        return [[rows[(slot, g)] for g in range(a.G)] for slot in range(len(cids))]

    def score_task1(self, samples, real_final):
        """v17 §3.2 / S1 (user 2026-09-30): the Brier reward of each Task 1 sample, computed by the learner holding the
        ROLLOUT policy (pi_old) under the env's GPU lock, eval mode, before the row is logged:
          valid, value located (mask_ok)  P_end = p(true) / (p(true) + p(false)) on the sample's OWN prefix,
                                          R = 1 - (P_end - y)^2, y = 1[t == n]         status "valid"
          valid, value not located        dropped from the group (our parsing failure): no reward, no gradient
                                                                                     status "dropped", stop_mask_mismatch
          not a decision                  R = 0 (the worst Brier score): format penalty  status "invalid"."""
        y = 1.0 if real_final else 0.0
        items, idx = [], []
        for i, x in enumerate(samples):
            if not x.get("decision_valid"):
                x.update(status="invalid", p_end=None, reward=0.0, dropped=False, drop_reason=None)
            elif not x.get("mask_ok"):
                x.update(status="dropped", p_end=None, reward=None, dropped=True, drop_reason="stop_mask_mismatch")
            else:
                items.append({"prompt_ids": x["planner_gen"]["prompt_ids"], "prefix_ids": x["prefix_ids"],
                              "target_true": x["target_true"], "target_false": x["target_false"]})
                idx.append(i)
        if items:
            with self.gpu():
                pe = self.learner.p_end_batch(items)
            for i, p in zip(idx, pe):
                p = float(p)
                assert 0.0 <= p <= 1.0, "P_end %r outside [0, 1]" % p
                samples[i].update(status="valid", p_end=p, reward=1.0 - (p - y) ** 2, dropped=False, drop_reason=None)
        return samples

    def task1_rollouts(self, u):
        """v17 §3.2 (user 2026-09-30): Task 1 groups on REAL train_all conversations: --task1-convs conversations (rng
        seeded by (seed, "task1", u)), EVERY decision point t = 2..n (a point after a capped emitted message is skipped
        and counted), --task1-G sampled plans each from the state the current policy reaches greedily (T 1, top-p 1);
        each sample scored by score_task1 before its row is logged. No refill (v16's dynamic sampling is gone)."""
        a, pv, psha = self.a, u - 1, self.learner.policy_sha()
        self.t1_rec = {"convs": [], "groups": [], "skipped_capped": []}
        if a.task1_convs <= 0:
            return []
        reuse = {}
        aborted = os.path.exists(os.path.join(a.out, "ABORTED_u%05d.json" % u))
        for r in read_jsonl(self.p_roll_t1):
            if not aborted and r["update"] == u and r["policy_version"] == pv and r["policy_sha"] == psha:
                reuse[(r["conversation_id"], r["t"])] = r
        rng = random.Random(seed_of(a.seed, "task1", u))
        pool = sorted(self.split["train_all"])     # Task 1 needs no requirement shards: every train session
        cids = rng.sample(pool, min(a.task1_convs, len(pool)))
        for cid in cids:
            assert_train_all_id(cid, self.split)
        skipped = []

        def run_conv(cid):
            n = self.env.human_turns(cid)
            if n < 2:
                return []                    # turn 1 never ends: a one-message conversation has no stop decision
            pos = list(range(2, n + 1))
            rows = [reuse[(cid, t)] for t in pos if (cid, t) in reuse]
            missing = [t for t in pos if (cid, t) not in reuse]
            if not missing:
                return rows
            prompts = self.env.task1_prompts(cid)
            assert len(prompts) == n
            capped = [t for t in missing if any(p.get("emitted_capped") for p in prompts[: t - 1])]
            with self.io_lock:
                skipped.extend([cid, t] for t in capped)
            for t in [t for t in missing if t not in capped]:
                pr = prompts[t - 1]
                assert pr["t"] == t and pr["real_final"] == (t == n)
                smp = self.env.task1_sample(cid, t, pr["user_prompt"], pr["real_final"], a.task1_G,
                                            a.temperature, a.top_p, seed_of(a.seed, "t1s", u))
                self.score_task1(smp, t == n)
                row = {"update": u, "conversation_id": cid, "t": t, "n_real": n, "real_final": t == n,
                       "split": "train", "policy_version": pv, "policy_sha": psha, "samples": smp,
                       "time": time.time()}
                with self.io_lock:
                    append_jsonl(self.p_roll_t1, row)
                rows.append(row)
            return rows

        out = []
        if a.rollout_workers <= 1:
            for cid in cids:
                out += run_conv(cid)
        else:
            with ThreadPoolExecutor(max_workers=a.rollout_workers) as ex:
                for rows in ex.map(run_conv, cids):
                    out += rows
        self.t1_rec = {"convs": sorted(cids), "groups": sorted([r["conversation_id"], r["t"]] for r in out),
                       "skipped_capped": sorted(skipped)}
        return sorted(out, key=lambda r: (r["conversation_id"], r["t"]))

    def task1_samples(self, t1rows, pv, psha):
        """v17 §3.2 / B7 / S2: the GRPO samples of the Task 1 groups (one Planner step per sample). The dropped samples
        (value not located) leave the group first; a group left with < 2 samples is skipped (n_groups_lt2), a group of
        invalid samples only is skipped without a format penalty (n_groups_all_invalid), a group whose rewards have no
        spread is skipped (groups_skipped_zero_std). A = R - mean(R) (Dr. GRPO) within the kept samples: a valid sample
        carries it as adv_prefix on the tokens before its value (prefix_mask), an invalid one as adv on every token;
        both x (1 - note); never on the value tokens; stop credit does not apply.
        -> (samples, counts, per-group advantage records)."""
        samples, recs = [], []
        cnt = {"groups_skipped_zero_std": 0, "n_groups_lt2": 0, "n_groups_all_invalid": 0, "n_groups_used": 0,
               "n_groups_used_final": 0, "n_groups_used_nonfinal": 0, "n_groups_used_adv_gt_1e3": 0}
        for r in t1rows:
            assert r["policy_version"] == pv and r["policy_sha"] == psha, "off-policy task1 rollout"
            assert r["t"] >= 2, "Task 1 stop group at turn 1"
            kept = [x for x in r["samples"] if not x["dropped"]]
            if len(kept) < 2:
                cnt["n_groups_lt2"] += 1
                recs.append([r["conversation_id"], r["t"], "lt2", []])
                continue
            if all(x["status"] == "invalid" for x in kept):
                cnt["n_groups_all_invalid"] += 1
                recs.append([r["conversation_id"], r["t"], "all_invalid", []])
                continue
            adv = RA.group_advantages([x["reward"] for x in kept], self.acfg["adv_eps"], self.acfg["min_group_std"],
                                      self.acfg["grpo_std_norm"])
            if adv is None:
                cnt["groups_skipped_zero_std"] += 1
                recs.append([r["conversation_id"], r["t"], "zero_std", []])
                continue
            cnt["n_groups_used"] += 1
            cnt["n_groups_used_final" if r["real_final"] else "n_groups_used_nonfinal"] += 1
            cnt["n_groups_used_adv_gt_1e3"] += int(max(abs(A) for A in adv) > 1e-3)
            used = []
            for x, A in zip(kept, adv):
                g = dict(x["planner_gen"])
                where = "all"
                if x["status"] == "valid":
                    g["prefix_mask"] = RA.prefix_mask_of(g.get("stop_mask"))
                    assert g["prefix_mask"] is not None, "a valid Task 1 sample without a located value"
                    where = "prefix"
                pseudo = {"conversation_id": r["conversation_id"], "replicate": x["replicate"],
                          "trace": [{"t": x["t"], "planner_gen": g}]}
                for s_ in RA.episode_samples(pseudo, policy_version=pv):
                    s_["ret"], s_["source"], s_["adv_stop"] = x["reward"], "task1", 0.0
                    if where == "prefix":
                        s_["adv"], s_["adv_prefix"] = 0.0, A
                    else:
                        s_["adv"], s_["adv_prefix"] = A, 0.0
                    samples.append(s_)
                used.append([x["replicate"], x["status"], A, where])
            recs.append([r["conversation_id"], r["t"], "used", used])
        return samples, cnt, recs

    def aux_examples(self, t1rows):
        """v17 §3.3 (user 2026-09-30): one stop-supervision example per decision point -- the first valid sample with a
        located value, its own prefix, the value re-encoded with the human's decision (y) -- weight --aux-weight."""
        w = self.aux_weight()
        aux = []
        if w <= 0:
            return aux
        for r in t1rows:
            x = next((x for x in r["samples"] if x.get("status") == "valid"), None)
            if x is None:
                continue
            aux.append({"prompt_ids": list(x["planner_gen"]["prompt_ids"]), "prefix_ids": list(x["prefix_ids"]),
                        "target_ids": list(x["target_true"] if r["real_final"] else x["target_false"]),
                        "want_end": bool(r["real_final"]), "gen_len": len(x["planner_gen"]["gen_ids"]), "weight": w,
                        "conversation_id": r["conversation_id"], "t": r["t"], "replicate": x["replicate"]})
        return aux

    def task1_eval_row(self, u, psha, cid):
        """Validation Task 1 of one conversation: the greedy Task 1 run (term_f1 etc. via task1_stop) and, at every
        decision point t >= 2, the teacher-forced end probability (v16 item 7). -> the validation.jsonl row."""
        t1r = self.env.run_task1(cid, keep_prompts=True)
        probs = []
        for x in t1r["turns"]:
            if x["t"] < 2:
                continue               # turn 1 cannot end: not a decision point
            pr = self.env.task1_end_probe(cid, x["t"], x["user_prompt"], x["real_final"])
            if pr["valid"]:
                # the learner's forward shares the GPU with the Speaker / Planner calls of Task2Env
                with self.gpu():
                    pe = float(self.learner.end_prob(pr))
            else:
                # no value to score: an invalid plan reads as "not ending" (the benchmark's reading); a valid
                # decision whose value tokens could not be located counts as its greedy decision
                pe = 1.0 if (pr.get("decision_valid") and pr["greedy_end"]) else 0.0
            probs.append({"t": x["t"], "real_final": bool(x["real_final"]), "p_end": pe, "valid": bool(pr["valid"]),
                          "decision_valid": bool(pr.get("decision_valid", pr["valid"])),
                          "greedy_end": bool(pr["greedy_end"]), "gen_adapter": pr.get("gen_adapter")})
        for x in t1r["turns"]:
            x.pop("user_prompt", None)
        return {"kind": "task1", "update": u, "policy_sha": psha, "conversation_id": cid,
                "split": "validation", "task1": t1r, "end_probs": probs, "time": time.time()}

    # -------------------------------------------------------- one update
    def one_update(self, u):
        t0 = time.time()
        cfg = copy.deepcopy(self.cfg)
        a = self.a
        _t = time.time()
        groups_all = self.rollouts(u)
        timing = {"task2_rollouts_s": round(time.time() - _t, 1)}
        pv, psha = u - 1, self.learner.policy_sha()
        for grp in groups_all:
            for row in grp:
                assert row["policy_version"] == pv and row["policy_sha"] == psha, "off-policy rollout"
        # an episode with a cut R0 reply, a lost ledger verdict or an emitted capped message never enters a
        # reward group (its reward would be wrong); a group left with < 2 episodes has no baseline
        groups = [[row for row in grp if row["episode"]["clean"]] for grp in groups_all]
        n_unclean = sum(len(g0) - len(g1) for g0, g1 in zip(groups_all, groups))
        n_singletons = sum(1 for g in groups if len(g) == 1)       # a lone clean episode has no baseline
        all_clean = [row["episode"] for grp in groups for row in grp]   # q: every clean rollout of this update
        groups = [g for g in groups if len(g) >= 2]
        clean_eps = [row["episode"] for grp in groups for row in grp]
        # note: the G replicates of a group share the scenario seed (Speaker, act-RNG, few-shot draws), so
        # only the sampled Planner (and R0) differ within a group: common random numbers, by design
        if not clean_eps:
            raise SystemExit("update %d: no clean episode (R0 / ledger judge failing?) -- stopping, not training on it" % u)
        ctx = self.reward_ctx(all_clean, cfg)
        shadow_ctx = self.reward_ctx(all_clean, self.selection_cfg)
        samples, rewarded, rew_groups, stop_groups, shadow = [], [], [], [], []
        for grp in groups:
            rs, sp = [], []
            for row in grp:
                rw = RR.reward(row["episode"], cfg, ctx)
                rs.append(rw["total"])
                sp.append(float(rw["components"].get("stop_part", 0.0)))
                rewarded.append((row["episode"], rw))
                # fixed-weight shadow reward (selection_cfg): comparable across controller changes
                shadow.append(RR.reward(row["episode"], self.selection_cfg, shadow_ctx)["total"])
            rew_groups.append(rs)
            stop_groups.append(sp)
        stop_credit = a.stop_credit and cfg["version"] in ("v3", "v4")
        if stop_credit:
            # stop credit assignment (Task 2 only, v17 S11): the length term (v4 log p_h(T) - log q(T)) is caused
            # only by the end_session decisions, so its group advantage goes to the end_session value tokens
            # of every decision step; coverage and the format constraints keep the sequence-level advantage;
            # both parts are centred on the TOTAL reward's group mean (Dr. GRPO: not divided by its std), so they add
            # up to the plain advantage and w_cov / w_dist keep their effect
            advs, advs_stop = [], []
            for rs, sp in zip(rew_groups, stop_groups):
                a1, a2 = RA.split_group_advantages(rs, sp, self.acfg["adv_eps"], self.acfg["min_group_std"],
                                                   self.acfg["grpo_std_norm"])
                advs.append(a1)
                advs_stop.append(a2)
            skipped = sum(1 for a1 in advs if a1 is None)
        else:
            advs, skipped = RA.advantages_for_groups(rew_groups, a.algo, self.acfg)
            advs_stop = [None] * len(groups)
        for grp, rs, ad, ads in zip(groups, rew_groups, advs, advs_stop):
            if ad is None and ads is None:
                continue
            for j, (row, R) in enumerate(zip(grp, rs)):
                for s_ in RA.episode_samples(row["episode"], policy_version=row["policy_version"]):
                    s_["ret"], s_["adv"] = R, (ad[j] if ad is not None else 0.0)
                    s_["adv_stop"] = ads[j] if ads is not None else 0.0
                    s_["source"] = "task2"
                    samples.append(s_)
        # Task 1 groups (same policy version; one Planner step per sample; Brier reward, v17 §3.2)
        _t = time.time()
        t1rows = self.task1_rollouts(u)
        timing["task1_groups_s"] = round(time.time() - _t, 1)
        t1_samples, t1_cnt, t1_advs = self.task1_samples(t1rows, pv, psha)
        samples += t1_samples
        t1_all = [x for r in t1rows for x in r["samples"]]
        t1_stats, t1_hist = None, None
        if t1_all or a.task1_convs > 0:
            scored = [x for x in t1_all if not x["dropped"]]
            fin = [x for x in t1_all if x["real_final"]]
            mid = [x for x in t1_all if not x["real_final"]]
            pf = [x["p_end"] for x in fin if x["p_end"] is not None]
            pm = [x["p_end"] for x in mid if x["p_end"] is not None]
            kept_std = [RA.pstd([x["reward"] for x in r["samples"] if not x["dropped"]]) for r in t1rows
                        if sum(1 for x in r["samples"] if not x["dropped"]) >= 2]
            t1_hist = {"n": len(t1_all), "n_points": len(t1rows),
                       "brier_mean": (sum(x["reward"] for x in scored) / len(scored)) if scored else None,
                       "end_at_final": (sum(x["ended_planner"] for x in fin) / len(fin)) if fin else None,
                       "end_at_nonfinal": (sum(x["ended_planner"] for x in mid) / len(mid)) if mid else None,
                       "p_end_final_mean": (sum(pf) / len(pf)) if pf else None,
                       "p_end_nonfinal_mean": (sum(pm) / len(pm)) if pm else None,
                       "n_dropped_mask": sum(1 for x in t1_all if x["dropped"]),
                       "n_not_decisions": sum(1 for x in t1_all if x["status"] == "invalid"),
                       "n_skipped_capped_history": len(self.t1_rec["skipped_capped"]),
                       "reward_std_mean": (sum(kept_std) / len(kept_std)) if kept_std else None,
                       "groups_skipped_zero_std": t1_cnt["groups_skipped_zero_std"],
                       "n_groups_lt2": t1_cnt["n_groups_lt2"], "n_groups_used": t1_cnt["n_groups_used"],
                       # D-N1: the used groups at the final / an earlier position, and with a non-negligible advantage
                       "n_groups_used_final": t1_cnt["n_groups_used_final"],
                       "n_groups_used_nonfinal": t1_cnt["n_groups_used_nonfinal"],
                       "n_groups_used_adv_gt_1e3": t1_cnt["n_groups_used_adv_gt_1e3"],
                       # D-N7: scored samples whose re-encoded own value differs from the sampled value tokens
                       "n_value_roundtrip_fail": sum(1 for x in t1_all if x.get("value_roundtrip_ok") is False),
                       "n_groups_no_decision": t1_cnt["n_groups_all_invalid"]}
            # the full record under the spec's names (S15 n_invalid, B7 n_groups_all_invalid) lives in the update row's
            # task1_stats: the controller history may not carry a key that contains "valid" (rl_controllers
            # FORBIDDEN_KEY_PARTS), so the history copy above says n_not_decisions / n_groups_no_decision
            t1_stats = dict(t1_hist, n_invalid=t1_hist["n_not_decisions"],
                            n_groups_all_invalid=t1_cnt["n_groups_all_invalid"], convs=self.t1_rec["convs"],
                            groups=self.t1_rec["groups"], skipped_capped=self.t1_rec["skipped_capped"],
                            advantages=t1_advs)
        assert all(s_["policy_version"] == pv for s_ in samples), "sample from another policy version"
        w_aux = self.aux_weight()
        aux = self.aux_examples(t1rows)
        if t1_stats is not None:
            t1_stats["aux_points"] = sorted([x["conversation_id"], x["t"]] for x in aux)
        aux_orders = []
        for ep in range(int(self.acfg["epochs"])):
            o = list(range(len(aux)))
            random.Random(seed_of(a.seed, "aux_mb", u, ep)).shuffle(o)
            aux_orders.append(o)
        _t = time.time()
        if getattr(self, "gen_name", None):
            bad = sorted({s_.get("gen_adapter") for s_ in samples if s_.get("gen_adapter") != self.gen_name})
            if bad:
                raise AssertionError("samples generated by adapter(s) %r, the policy is %r: off-policy generation"
                                     % (bad[:3], self.gen_name))
        try:
            stats = self.learner.update(samples, cfg, seed=seed_of(a.seed, "update", u), aux=aux or None,
                                        aux_orders=aux_orders if aux else None,
                                        **({"mismatch_abort": a.behav_mismatch_abort} if not a.dry_run else {}))
        except RA.MismatchAbort as e:
            # before any optimizer step and before the checkpoint; the marker makes a resume regenerate update u's
            # rollouts instead of reusing the same mismatched ones
            write_json_atomic(os.path.join(a.out, "ABORTED_u%05d.json" % u),
                              {"update": u, "mismatch": e.value, "threshold": a.behav_mismatch_abort,
                               "adapter": getattr(self, "gen_name", None), "time": time.time()})
            raise SystemExit("update %d: %s > %.4f (phase-0 measured 0.017): the served adapter does not match the "
                             "learner?" % (u, e, a.behav_mismatch_abort))
        timing["learner_s"] = round(time.time() - _t, 1)
        n_mb = min(int(self.acfg["minibatches"]), len(samples)) if samples else 0
        want_steps = int(self.acfg["epochs"]) * n_mb if samples else (1 if aux else 0)
        if stats.get("optimizer_steps", 0) != want_steps:
            raise AssertionError("update %d: %s optimizer steps, expected %d" % (u, stats.get("optimizer_steps"), want_steps))

        def adv_abs(src):
            xs = [abs(float(s_.get("adv") or 0.0)) + abs(float(s_.get("adv_prefix") or 0.0))
                  + abs(float(s_.get("adv_stop") or 0.0)) for s_ in samples if s_.get("source") == src]
            return (sum(xs) / len(xs)) if xs else None
        stats["adv_abs_mean"] = (sum(abs(float(s_.get("adv") or 0.0)) + abs(float(s_.get("adv_prefix") or 0.0))
                                     + abs(float(s_.get("adv_stop") or 0.0)) for s_ in samples)
                                 / len(samples)) if samples else None
        stats["adv_abs_mean_by_source"] = {"task1": adv_abs("task1"), "task2": adv_abs("task2")}
        stats["n_samples_by_source"] = {"task1": sum(1 for s_ in samples if s_.get("source") == "task1"),
                                        "task2": sum(1 for s_ in samples if s_.get("source") == "task2")}
        stats["expected_optimizer_steps"] = want_steps
        agg = RR.aggregate(rewarded)
        tm = int(cfg["t_max"])
        turn_hist = [0] * (tm + 1)
        for e in all_clean:
            turn_hist[min(max(int(e["emitted_user_turns"]), 0), tm)] += 1
        # v17 §3.5 / S5: the length drift of this update = mean(T - min(human turns, t_max)) over the clean episodes
        # that entered a group
        drift = sum(int(e["emitted_user_turns"]) - min(int(e["human_turns"]), tm) for e in clean_eps) / len(clean_eps)
        t2_std = [RA.pstd(rs) for rs in rew_groups]
        hist = {"update": u, "split": "train", "reward_version": cfg["version"], "task1_train": t1_hist, **agg,
                "n_groups": len(groups), "n_groups_skipped_zero_std": skipped, "n_unclean_episodes": n_unclean,
                "n_dropped_singleton_episodes": n_singletons,
                "shadow_reward_mean": sum(shadow) / len(shadow), "turn_hist": turn_hist, "p_h": self.p_h,
                "task2_turns_mean": sum(int(e["emitted_user_turns"]) for e in all_clean) / len(all_clean),
                "kl_q_ph": kl_div(ctx["q"], ctx["p_h"]) if ctx else None,
                "drift_stat": drift, "n_drift_episodes": len(clean_eps),
                "reward_std_mean": {"task2": (sum(t2_std) / len(t2_std)) if t2_std else None,
                                    "task1": (t1_hist or {}).get("reward_std_mean")},
                "aux_weight": w_aux, "lr": cfg["lr"], "kl_coef": cfg["kl_coef"],
                "aux_stats": {k: stats.get(k) for k in ("aux_n", "aux_loss", "aux_grad_norm", "aux_grad_norm_max",
                                                         "aux_p_correct_before", "aux_p_correct_before_final")},
                **{k: stats.get(k) for k in ("loss", "kl", "ratio_mean", "clip_frac", "grad_norm", "n_tokens",
                                              "value_mse", "ratio_init_maxdev", "rl_grad_norm", "rl_grad_norm_max",
                                              "kl_step_mean", "kl_step_max", "clip_frac_step_mean",
                                              "clip_frac_step_max", "tis_w_mean", "tis_capped_frac",
                                              "behav_mismatch_mean", "optimizer_steps")}}
        self.history.append(hist)
        self.cfg = self.controller.propose(self.history)
        self.update_done = u
        row = {"update": u, "policy_version_rollouts": pv, "policy_version_after": u, "algo": a.algo,
               "controller": a.controller, "cfg_used": cfg, "cfg_used_sha256": RR.cfg_sha(cfg),
               "reward_ctx": ctx, "next_cfg": self.cfg, "scenarios": [g[0]["conversation_id"] for g in groups_all],
               "train_aggregate": hist, "task1_stats": t1_stats, "learner_stats": stats, "n_samples": len(samples),
               "timing": timing,
               "controller_failures": getattr(self.controller, "n_failures", None),
               "controller_rollbacks": getattr(self.controller, "n_rollbacks", None),
               "policy_sha_after": self.learner.policy_sha(), "time": time.time(), "update_s": time.time() - t0}
        self.save_checkpoint(u, row)
        append_jsonl(self.p_upd, row)
        return row

    # -------------------------------------------------------- validation
    def validate(self, u, task2):
        """v17 §4 (user 2026-09-30): Task 1 (greedy + teacher-forced end probabilities) on validation_all at every
        validated update; with task2=True (u0 = SFT and the final update) also Task 2 episodes on validation with the
        SAMPLED Planner (--val-temperature, one replicate per --val-seeds entry, 0..7). Logged only to validation.jsonl;
        no checkpoint selection. The summary is keyed by (update, task2): a Task-1-only summary never stands in for the
        final validation (S6). Never enters history or the controller."""
        _tv = time.time()
        self.sync_generation_policy(u)          # validation generates with the policy after update u
        a = self.a
        seeds = list(a.val_seeds) if task2 else []
        prev = read_jsonl(self.p_val)
        for r in prev:
            if r.get("kind") == "summary" and r["update"] == u and r.get("task2") == bool(task2):
                return r
        done = {(r["conversation_id"], r["seed"]): r for r in prev if r.get("kind") == "episode" and r["update"] == u}
        done_t1 = {r["conversation_id"]: r for r in prev if r.get("kind") == "task1" and r["update"] == u}
        psha = self.learner.policy_sha()
        jobs = []
        for cid in sorted(self.split["validation"]):
            assert cid in self.split["validation"] and cid not in self.split["train"] \
                and cid not in self.split["train_all"], "validation id %r is also a training id" % cid
            for s in seeds:
                jobs.append((cid, s))
        t1_ids = sorted(self.split["validation_all"])
        for cid in t1_ids:
            assert cid in self.split["forbidden"] and cid not in self.split["train"] \
                and cid not in self.split["train_all"], "validation_all id %r is also a training id" % cid

        def run_val(job):
            cid, s = job
            r = done.get((cid, s))              # the latest attempt (later rows overwrite earlier ones)
            if r is not None and r["policy_sha"] == psha and (r["episode"]["clean"] or r.get("attempt", 0) >= VAL_RETRIES):
                return r
            attempt = r.get("attempt", 0) + 1 if (r is not None and r["policy_sha"] == psha) else 0
            while True:
                ep = self.env.run_episode(cid, seed=s, replicate=0, planner_temperature=a.val_temperature,
                                          planner_top_p=a.val_top_p, record_generation=False)
                r = {"kind": "episode", "update": u, "policy_sha": psha, "conversation_id": cid, "seed": s,
                     "attempt": attempt, "split": "validation", "episode": ep, "time": time.time()}
                with self.io_lock:
                    append_jsonl(self.p_val, r)
                if ep["clean"] or attempt >= VAL_RETRIES:
                    return r
                attempt += 1                    # an infrastructure incident, not the policy: run it again

        def run_t1(cid):
            r = done_t1.get(cid)
            if r is None or r["policy_sha"] != psha or "end_probs" not in r:
                r = self.task1_eval_row(u, psha, cid)
                with self.io_lock:
                    append_jsonl(self.p_val, r)
            return r

        if not hasattr(self.env, "run_task1"):
            raise RuntimeError("validation needs Task 1 (run_task1)")
        workers = max(1, a.rollout_workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            vrows = list(ex.map(run_val, jobs))
            t1rows = list(ex.map(run_t1, t1_ids))
        eps = [r["episode"] for r in vrows if r["episode"]["clean"]]
        n_unclean = len(vrows) - len(eps)
        vctx = self.reward_ctx(eps, self.selection_cfg) if eps else None
        totals = [RR.reward(e, self.selection_cfg, vctx)["total"] for e in eps]
        score = sum(totals) / len(totals) if totals else None
        turn_stats = None
        if eps and all("human_turns" in e for e in eps):
            turn_stats = turn_stats_of(eps, int(self.selection_cfg["t_max"]))
        t1 = T1.task1_stop_metrics([r["task1"] for r in t1rows])
        t1.update(T1.task1_prob_metrics([p_ for r in t1rows for p_ in r["end_probs"]]))
        if u == 0 and self.task1_base is None:
            # the comparison base: SFT (u0)'s validation Task 1 (persisted now: a crash in update 1 must not lose it)
            self.task1_base = {"update": u, **t1}
            self.update_state(0, task1_base=self.task1_base)
        t1_ok = True if self.task1_base is None else T1.within_tolerance(t1, self.task1_base, a.task1_tol)
        summary = {"kind": "summary", "update": u, "task2": bool(task2), "policy_sha": psha, "split": "validation",
                   "validation_s": round(time.time() - _tv, 1),
                   "n_episodes": len(totals), "n_unclean_episodes": n_unclean,
                   "unclean_after_retries": n_unclean > 0, "val_temperature": a.val_temperature, "val_seeds": seeds,
                   "task2_ids": sorted(self.split["validation"]) if task2 else [], "task1_ids": t1_ids,
                   "mean_reward_fixed_cfg": score, "turn_stats": turn_stats, "task1": t1,
                   "task1_base": self.task1_base, "task1_within_tol": t1_ok, "task1_tol": a.task1_tol,
                   "fixed_cfg_sha256": RR.cfg_sha(self.selection_cfg), "time": time.time()}
        append_jsonl(self.p_val, summary)
        return summary

    # -------------------------------------------------------- the end of the run (v17 §3.5, S5, S6, S13)
    def read_final(self):
        return json.load(open(self.p_final, encoding="utf-8")) if os.path.exists(self.p_final) else None

    def drift_stats(self):
        return {h["update"]: h.get("drift_stat") for h in self.history}

    def finalize(self, u, reason):
        """final.json + a run_meta "stop" row; the checkpoint u records its stop reason."""
        st = json.load(open(os.path.join(self.ckpt_dir(u), "state.json")))
        self.stop_reason = reason
        self.update_state(u, stop_reason=reason)
        fin = {"final_update": u, "stop_reason": reason, "policy_sha": st["policy_sha"], "validated": False,
               "length_drift_margin": self.a.length_drift_margin, "drift_stats": self.drift_stats(),
               "time": time.time()}
        # fix round 1 (A-3): the run_meta stop row first, then final.json (a resume adds a missing stop row)
        self.ensure_stop_row(u, reason)
        write_json_atomic(self.p_final, fin)
        print(json.dumps({"final_update": u, "stop_reason": reason}), flush=True)
        return fin

    def ensure_stop_row(self, u, reason):
        if not any(m.get("kind") == "stop" for m in read_jsonl(self.p_meta)):
            append_jsonl(self.p_meta, {**self.meta("stop"), "final_update": u, "stop_reason": reason})

    def ensure_sft_row(self):
        """Fix round 1 (B-S1): a crash between save_checkpoint(0) and the run_meta "sft" row is repaired on resume from
        u0's own state (the row names the ref = u0 sha)."""
        if self.sft_info is not None and not any(m.get("kind") == "sft" for m in read_jsonl(self.p_meta)):
            append_jsonl(self.p_meta, {**self.meta("sft"), "sft_info": self.sft_info, "repaired_on_resume": True})

    def val_due(self, u):
        return self.a.val_every > 0 and u % self.a.val_every == 0

    # -------------------------------------------------------- main loop
    def run(self):
        a = self.a
        fin = self.read_final()
        if fin is not None and fin.get("validated"):
            raise SystemExit("%s is finished (final.json: u%d, %s): no further training; use a new --out"
                             % (a.out, fin["final_update"], fin["stop_reason"]))
        if not a.resume:
            if os.path.exists(os.path.join(self.ckpt_root, "LATEST.json")):
                raise SystemExit("%s already has checkpoints; pass --resume or use a new --out" % a.out)
            left = [p for p in (self.p_sft_ex, self.p_sft_meta, self.p_sft, self.p_base_pend)
                    if os.path.exists(p)] + [d for d in sorted(os.listdir(self.ckpt_root)) if d.startswith("sft_e")]
            if left:
                raise SystemExit("%s holds SFT leftovers %s; pass --resume (reuses the cached examples) or use a new "
                                 "--out" % (a.out, left))
        self.build()
        resumed = self.load_checkpoint() if a.resume else False
        row = self.meta("resume" if a.resume else "start")
        self.check_provenance(row)
        append_jsonl(self.p_meta, row)
        if not resumed:
            random.seed(a.seed)
            self.sft_stage()                          # -> u0 = SFT, LATEST, the ref adapter
        else:
            self.ensure_sft_row()
            if fin is not None:
                self.ensure_stop_row(fin["final_update"], fin["stop_reason"])
        if fin is None:
            # recomputed from the records (a crash between the last checkpoint and final.json)
            fu, reason = drift_decision(self.drift_stats(), a.length_drift_margin, a.updates)
            if fu is not None:
                assert fu == self.update_done, "stop condition at u%d but the latest checkpoint is u%d" % (fu, self.update_done)
                fin = self.finalize(fu, reason)
        if fin is None:
            if self.update_done == 0:
                self.validate(0, task2=True)
            elif self.val_due(self.update_done):
                self.validate(self.update_done, task2=False)   # completes a validation cut short by a crash
            for u in range(self.update_done + 1, a.updates + 1):
                r = self.one_update(u)
                print(json.dumps({"update": u, "reward_mean": r["train_aggregate"]["reward_mean"],
                                  "skipped": r["train_aggregate"]["n_groups_skipped_zero_std"],
                                  "drift": round(r["train_aggregate"]["drift_stat"], 3),
                                  "n_samples": r["n_samples"], "update_s": round(r["update_s"], 1)}), flush=True)
                fu, reason = drift_decision(self.drift_stats(), a.length_drift_margin, a.updates)
                if fu is not None:
                    fin = self.finalize(fu, reason)
                    break
                if self.val_due(u):
                    self.validate(u, task2=False)             # u1 .. u(final-1): Task 1 only
        assert fin["final_update"] == self.update_done, "final.json names u%d, the latest checkpoint is u%d" % (
            fin["final_update"], self.update_done)
        self.validate(fin["final_update"], task2=True)
        fin["validated"] = True
        fin["validated_time"] = time.time()
        write_json_atomic(self.p_final, fin)
        return self


def turn_stats_of(eps, t_max):
    """Task 2 turn statistics of clean episodes. Fix round 1 (D-N5): the human side is min(human_turns, t_max) (a
    simulated session cannot exceed t_max), in human_turns_mean, abs_diff_mean and turn_w1 alike; the uncapped mean is
    kept as human_turns_mean_uncapped."""
    hum = [min(int(e["human_turns"]), t_max) for e in eps]
    sim = [int(e["emitted_user_turns"]) for e in eps]
    return {"n_episodes": len(eps), "sim_turns_mean": sum(sim) / len(sim), "human_turns_mean": sum(hum) / len(hum),
            "human_turns_mean_uncapped": sum(int(e["human_turns"]) for e in eps) / len(eps),
            "human_turns_capped_at": t_max,
            "abs_diff_mean": sum(abs(a - b) for a, b in zip(sim, hum)) / len(sim),
            "coverage_mean": sum(float(e["coverage"]) for e in eps) / len(eps),
            "turn_w1": turn_w1(sim, hum),
            "end_kinds": {k: sum(e["end_kind"] == k for e in eps) for k in sorted({e["end_kind"] for e in eps})}}


def turn_w1(sim, human):
    """Wasserstein-1 between two samples of integer conversation lengths (sum of |CDF difference|)."""
    if not sim or not human:
        return None
    hi = max(max(sim), max(human))
    w, cs, ch = 0.0, 0.0, 0.0
    for k in range(0, hi + 1):
        cs += sum(1 for x in sim if x == k) / len(sim)
        ch += sum(1 for x in human if x == k) / len(human)
        w += abs(cs - ch)
    return w


SPEC = {"implicit_profile": 1, "fewshot": "fold", "selector": "borda"}      # the pend design (user, 2026-09-25)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--arm", choices=("pend",), default="pend",
                    help="policy env: pend = E1.6 base + fixes, Planner judges the goal and ends (no goal judge)")
    ap.add_argument("--ablation", default=None,
                    help="name of a declared ablation; required to run with any setting that differs from the pend spec")
    ap.add_argument("--batch", type=int, choices=(0, 1), default=1, help="cross-episode dynamic batching of Planner / Ditto generation")
    ap.add_argument("--implicit-profile", type=int, choices=(0, 1), default=SPEC["implicit_profile"])
    ap.add_argument("--fewshot", choices=("off", "fold"), default=SPEC["fewshot"], help="fold: examples from splits[fold].train_all only")
    ap.add_argument("--selector", choices=("length", "borda"), default=SPEC["selector"])
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--planner-backend", choices=("vllm", "hf"), default="vllm",
                    help="spec: vllm (Planner generation on the vLLM server; the HF model is the learner); hf needs --ablation")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8031/v1")
    ap.add_argument("--behav-mismatch-abort", type=float, default=0.1,
                    help="stop when mean |log pi_learner - log pi_vllm| over an update exceeds this (phase 0: 0.017)")
    ap.add_argument("--rollout-workers", type=int, default=1,
                    help="episodes run in threads; GPU calls are serialised inside Task2Env, R0/ledger calls overlap")
    ap.add_argument("--task1-tol", type=float, default=0.05,
                    help="report only: whether validation Task 1 term_f1 / premature stay within this of SFT (u0)")
    # ---- v17 (user 2026-09-30)
    ap.add_argument("--sft-lr", type=float, default=SPEC_V17["sft_lr"], help="SFT warm-up AdamW lr (spec 5e-5)")
    ap.add_argument("--sft-epochs-max", type=int, default=SPEC_V17["sft_epochs_max"],
                    help="SFT epochs (spec 3); the epoch (0 = none) with the lowest validation_all NLL becomes u0")
    ap.add_argument("--sft-samples-per-point", type=int, default=SPEC_V17["sft_samples_per_point"],
                    help="sampled SFT plans per decision point, besides the greedy one (spec 2)")
    ap.add_argument("--task1-convs", type=int, default=SPEC_V17["task1_convs"],
                    help="Task 1 groups per update: train_all conversations, every decision point (spec 8); 0 = off")
    ap.add_argument("--task1-G", type=int, default=SPEC_V17["task1_G"], help="samples per Task 1 group (spec 4)")
    ap.add_argument("--task1-reward", choices=("brier",), default=SPEC_V17["task1_reward"],
                    help="Task 1 reward: brier = 1 - (P_end - y)^2 on the sample's own prefix")
    ap.add_argument("--task1-positions", choices=("all",), default=SPEC_V17["task1_positions"],
                    help="Task 1 decision points: all = t = 2..n")
    ap.add_argument("--aux-weight", type=float, default=SPEC_V17["aux_weight"],
                    help="stop supervision weight (constant; spec 0.5; 0 = off)")
    ap.add_argument("--length-drift-margin", type=float, default=SPEC_V17["length_drift_margin"],
                    help="stop after two consecutive updates with mean(T - human turns) < -margin (spec 1.0)")
    ap.add_argument("--stop-credit", type=int, choices=(0, 1), default=1,
                    help="1: the Task 2 turn-count advantage acts only on the end_session value tokens (Task 1 never)")
    ap.add_argument("--splits", default="/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json")
    ap.add_argument("--algo", choices=RA.ALGOS, default="grpo")
    ap.add_argument("--controller", choices=("fixed", "dual", "llm"), default=SPEC_V17["controller"],
                    help="spec v17: fixed (weights never move); dual/llm need --ablation")
    ap.add_argument("--planner-path")
    ap.add_argument("--planner-dtype", default="bfloat16")
    ap.add_argument("--planner-nf4", action="store_true")
    ap.add_argument("--judge-adapter", default=None)
    ap.add_argument("--judge-base", default=None)
    ap.add_argument("--judge-manifest-train-all", action="store_true",
                    help="accept a judge whose train_scenarios lie in train_all (without shards) rather than train")
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--scenarios-per-update", type=int, default=4)
    ap.add_argument("--updates", type=int, default=SPEC_V17["updates"], help="spec v17: 5, fixed (never changed on resume)")
    ap.add_argument("--lr", type=float, default=SPEC_V17["lr"])
    ap.add_argument("--kl", type=float, default=SPEC_V17["kl"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=SPEC_V17["val_every"])
    ap.add_argument("--val-seeds", type=int, nargs="+", default=list(SPEC_V17["val_seeds"]),
                    help="validation Task 2 seeds of u0 and the final update (spec 0..7)")
    ap.add_argument("--val-temperature", type=float, default=0.7, help="D5: validation Task 2 uses the sampled Planner")
    ap.add_argument("--val-top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None, help='JSON: {"reward": {...}, "algo": {...}, "dual": {...}, "llm": {...}}')
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-code-change", action="store_true")
    ap.add_argument("--intervention", default=None, help="not available in v17 (refused; SPEC v17 S9)")
    ap.add_argument("--keep-optimizer-last", type=int, default=3,
                    help="delete optimizer.pt of checkpoints older than this many updates (adapters kept); 0 = keep all")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="fake env + pure-python learner (no torch, no GPU)")
    ap.add_argument("--dry-run-crash-after-episodes", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.intervention:
        ap.error("--intervention is not available in v17: the controller is fixed (SPEC v17 S9)")
    if a.G < 2:
        ap.error("--G must be >= 2 (group baselines)")
    if not (a.temperature > 0):
        ap.error("training rollouts need --temperature > 0")
    if a.stop_credit and a.algo != "grpo":
        ap.error("--stop-credit 1 is implemented for grpo")
    if a.task1_G < 2:
        ap.error("--task1-G must be >= 2 (group baselines)")
    if a.sft_epochs_max < 0 or a.sft_samples_per_point < 0 or not (a.sft_lr > 0):
        ap.error("--sft-epochs-max / --sft-samples-per-point must be >= 0 and --sft-lr > 0")
    if not (0.0 <= a.aux_weight <= 10.0):
        ap.error("--aux-weight must lie in [0, 10]")
    if not (a.length_drift_margin > 0):
        ap.error("--length-drift-margin must be > 0")
    if a.updates < 1:
        ap.error("--updates must be >= 1")
    off = {k: getattr(a, k) for k in SPEC if getattr(a, k) != SPEC[k]}
    for k, want in (("G", 4), ("behav_mismatch_abort", 0.1), ("batch", 1), ("task1_tol", 0.05),
                    ("val_temperature", 0.7), ("algo", "grpo")):
        if getattr(a, k) != want:
            off[k] = getattr(a, k)
    for k in ("lr", "kl", "sft_lr", "sft_epochs_max", "sft_samples_per_point", "task1_G", "task1_convs", "task1_reward",
              "task1_positions", "aux_weight", "updates", "length_drift_margin", "controller", "stop_credit"):
        if getattr(a, k) != SPEC_V17[k]:
            off[k] = getattr(a, k)
    if sorted(a.val_seeds) != SPEC_V17["val_seeds"]:
        off["val_seeds"] = a.val_seeds
    if not a.dry_run:
        for k, want in (("scenarios_per_update", 4), ("val_every", SPEC_V17["val_every"])):
            if getattr(a, k) != want:
                off[k] = getattr(a, k)
    if a.planner_path and "Qwen3-4B-Instruct-2507" not in a.planner_path:
        off["planner_path"] = a.planner_path          # the spec's Planner is Qwen3-4B-Instruct-2507
    if a.config:
        cj = json.load(open(a.config, encoding="utf-8"))
        for sect in ("reward", "algo", "llm", "dual"):
            if cj.get(sect):
                off["config." + sect] = cj[sect]            # any override of the declared defaults (incl. epochs 2 x 4)
    if a.planner_backend != "vllm" and not a.dry_run:
        off["planner_backend"] = a.planner_backend
    if a.planner_backend == "vllm" and (a.temperature != 1.0 or a.top_p != 1.0):
        ap.error("with the vLLM backend the training rollouts must use --temperature 1 --top-p 1: the server's "
                 "behaviour log-probs are the raw distribution, so any other value would bias the TIS weights")
    if off and not a.ablation:
        ap.error("settings %r differ from the pend spec v17; name the ablation with --ablation" % (off,))
    if not (a.val_temperature > 0):
        ap.error("--val-temperature must be > 0 (D5: sampled Planner at validation)")
    if not a.dry_run:
        for k in (("planner_path",) if a.arm == "pend" else ("planner_path", "judge_adapter", "judge_base")):
            if getattr(a, k) is None:
                ap.error("--%s is required (except with --dry-run)" % k.replace("_", "-"))
    return a


def main(argv=None):
    a = parse_args(argv)
    return Trainer(a).run()


if __name__ == "__main__":
    main()
