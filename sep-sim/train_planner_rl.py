#!/usr/bin/env python3
"""Planner RL for the pend arm (GRPO; v4 LLM factor controller). Design: ops/AUDIT_SPEC_pend_grpo.md.

Loop (update u = 1, 2, ...; the policy that generates update u's rollouts has policy_version u-1):
  1. sample --scenarios-per-update scenarios (seeded by (seed, u)) from splits[fold]["train"] ONLY;
     every id is asserted to be in train and not in forbidden_for_training;
  2. G rollouts each: Task2Env(arm="pend").run_episode(..., planner_temperature>0, record_generation=True);
     each row is appended to rollouts.jsonl with update, policy_version and policy_sha; episodes that are
     not clean (cut R0 reply, lost ledger verdict, emitted capped message, compacted prompt) are dropped;
  3. rewards = rl_reward.reward(episode, cfg, ctx) -- v4: coverage + log p_h(T) - log q(T) - penalties, p_h
     from splits[fold]["train_all"], q from this update's clean rollouts;
  4. advantages normalised once per group, split into the stop part (end_session tokens) and the rest;
     Task 1 stop groups on train_all conversations + the annealed stop supervision (D2);
     assert every sample was produced by the current policy_version (on-policy);
  5. checkpoint EVERY update (adapter, optimizer, controller state, RNG states, update index, policy_version,
     history, rl_manifest.json) atomically; then updates.jsonl;
  6. controller.propose(train aggregates only) -> cfg for the next update;
  7. every --val-every updates: SAMPLED-Planner episodes (D5) and greedy Task 1 on splits[fold]["validation"]
     -> validation.jsonl only; best.json = best checkpoint by w_sel_cov*coverage - w_sel_w1*W1(turn counts)
     + w_sel_task1*Task 1 term_f1 (M2); unclean validation episodes are re-run, else the score is withheld.
Validation never enters history, reward statistics or the controller; the test ids are never read.

--resume continues from ckpt/LATEST; rollouts of an interrupted update that were produced by the same
policy (policy_version and policy_sha match) are reused, the rest regenerated. Code SHA256s must match
the first launch unless --allow-code-change. --dry-run runs the whole loop with a fake env and a
pure-python learner (no torch, no GPU) to test the loop, checkpointing and resume.
"""
from __future__ import annotations

import argparse
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

CODE_FILES = ("train_planner_rl.py", "rl_reward.py", "rl_controllers.py", "rl_algos.py", "task2_env.py",
              "task2_episode.py", "goal_judge.py", "planner_prompt_v3.py", "fit_prompts.py", "ditto_e16.py",
              "task1_stop.py", "batching.py", "implicit_profile.py", "style_select.py")
TIME_KEYS = ("time", "wall_s", "rollout_s", "update_s", "timing", "validation_s",
             "planner_s", "speaker_s", "r0_s", "ledger_s")
VAL_RETRIES = 2          # an unclean validation episode (infrastructure incident) is re-run up to this many times
RESUME_MAY_CHANGE = ("resume", "allow_code_change", "updates", "rollout_workers", "gpu", "max_batch",
                     "keep_optimizer_last", "dry_run_crash_after_episodes", "vllm_url")


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


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ leakage control
def load_split(path, fold):
    d = json.load(open(path, encoding="utf-8"))
    folds = {int(f["fold"]): f for f in d["folds"]}
    f = folds[int(fold)]
    train, val = list(f["train"]), list(f["validation"])
    forbidden = set(f["forbidden_for_training"])
    assert train, "empty train split"
    assert not set(train) & forbidden, "train intersects forbidden_for_training"
    assert set(val) <= forbidden, "validation must be listed as forbidden for training"
    assert not set(train) & set(val), "train/validation overlap"
    assert not set(f.get("train_all", train)) & forbidden, "train_all intersects forbidden_for_training"
    for k, n in (f.get("sizes") or {}).items():          # the declared sizes of the split file
        assert len(f[k]) == n, "splits fold %d: %s has %d ids, declared %d" % (fold, k, len(f[k]), n)
    # the test ids are not kept in memory at all (only through 'forbidden')
    return {"fold": int(fold), "train": train, "train_all": list(f.get("train_all", train)),
            "validation": val, "forbidden": forbidden, "sha256": sha_file(path)}


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


# ------------------------------------------------------------------ dry-run fakes (pure python)
STATUS_IDX = {"NOT ASSESSED": 0, "NOT": 1, "PARTIAL": 2, "SATISFIED": 3, "UNKNOWN": 4}


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


class FakeLearner:
    """Tabular stop policy: p(stop | status) = sigmoid(theta[status]); samples carry the status in
    prompt_ids[0] and the action (1 = stop) in gen_ids[0]. Same interface as rl_algos.TorchLearner."""

    def __init__(self, algo, acfg, lr, seed=0, lr_scale=1e4):
        self.algo, self.acfg, self.lr_scale = algo, RA.algo_cfg(**acfg), lr_scale
        self.theta = [-1.0, -1.5, -0.5, 0.0, -1.0]
        self.value = [0.0] * 5
        self.m = [0.0] * 5            # momentum = "optimizer state"

    def policy_sha(self):
        return sha_bytes(json.dumps([round(x, 12) for x in self.theta]).encode())

    def trainable_names(self):
        return ["theta.%d" % i for i in range(5)]

    def logp(self, s):
        p = _sig(self.theta[s["prompt_ids"][0]])
        return math.log(p if s["gen_ids"][0] == 1 else 1 - p)

    def prepare(self, samples):
        for s in samples:
            s["old_logp"] = self.logp(s)
        if self.algo == "ppo":
            adv = RA.ppo_advantages([s["ret"] for s in samples], [self.value[s["prompt_ids"][0]] for s in samples])
            if self.acfg["ppo_adv_norm"]:
                adv = RA.normalize(adv, self.acfg["adv_eps"])
            for s, a in zip(samples, adv):
                s["adv"] = a

    def update(self, samples, cfg, seed, aux=None):
        if not samples and not aux:
            return {"n_samples": 0, "n_tokens": 0, "skipped_update": True}
        if not samples:
            out = {"n_samples": 0, "n_tokens": 0, "loss": 0.0, "kl": 0.0, "ratio_mean": 1.0, "clip_frac": 0.0,
                   "grad_norm": 0.0, "ratio_init_maxdev": 0.0, "optimizer_steps": 0}
            self._aux_step(aux, cfg, out, own_step=True)
            return out
        st = self._update(samples, cfg, seed)
        if aux:
            self._aux_step(aux, cfg, st)
        return st

    def _aux_step(self, aux, cfg, st, own_step=False):
        lr = cfg["lr"] * self.lr_scale
        n_t = sum(int(x["gen_len"]) for x in aux)            # normalised per generated token, like the real learner
        ps = []
        for x in aux:
            k = x["prompt_ids"][0]
            p = _sig(self.theta[k])
            ps.append(p if x["target_ids"][0] == 1 else 1 - p)
            grad = (1 - p) if x["target_ids"][0] == 1 else -p
            self.theta[k] += lr * float(x["weight"]) * grad / n_t
        st["aux_n"], st["aux_p_correct_before"] = len(aux), sum(ps) / len(ps)
        if own_step:                                           # aux-only update: its own step
            st["optimizer_steps"] = st.get("optimizer_steps", 0) + 1

    def _update(self, samples, cfg, seed):
        self.prepare(samples)
        lr, eps = cfg["lr"] * self.lr_scale, self.acfg["clip_eps"]
        maxdev, steps, ratios = 0.0, 0, []
        for ep in range(int(self.acfg["epochs"])):
            for mb in RA.minibatches(len(samples), int(self.acfg["minibatches"]), seed * 1000 + ep):
                g = [0.0] * 5
                for i in mb:
                    s = samples[i]
                    r = math.exp(self.logp(s) - s["old_logp"])
                    if ep == 0 and steps == 0:
                        maxdev = max(maxdev, abs(r - 1))
                    ratios.append(r)
                    k = s["prompt_ids"][0]
                    p = _sig(self.theta[k])
                    dlogp = (1 - p) if s["gen_ids"][0] == 1 else -p
                    clipped = self.algo in ("grpo", "ppo") or self.acfg["rloo_clipped"]
                    adv = s["adv"] + (s.get("adv_stop") or 0.0) * ((s.get("stop_mask") or [0])[0])
                    active = (not clipped) or RA.clipped_surrogate(r, adv, eps) == r * adv
                    if active:
                        g[k] += adv * r * dlogp / len(mb)
                    if self.algo == "ppo":
                        self.value[k] += 0.1 * (s["ret"] - self.value[k]) / len(mb)
                for k in range(5):
                    self.m[k] = 0.9 * self.m[k] + g[k]
                    self.theta[k] += lr * self.m[k]
                steps += 1
        if maxdev > self.acfg["ratio_init_tol"]:
            raise AssertionError("off-policy start")
        return {"n_samples": len(samples), "n_tokens": len(samples), "loss": 0.0, "kl": 0.0,
                "ratio_mean": sum(ratios) / len(ratios), "clip_frac": 0.0, "grad_norm": 0.0,
                "ratio_init_maxdev": maxdev, "optimizer_steps": steps}

    def save(self, d):
        write_json_atomic(os.path.join(d, "fake_learner.json"),
                          {"theta": self.theta, "value": self.value, "m": self.m})

    def load(self, d):
        s = json.load(open(os.path.join(d, "fake_learner.json")))
        self.theta, self.value, self.m = s["theta"], s["value"], s["m"]


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
                step["planner_gen"] = {"prompt_ids": [k, t], "gen_ids": [int(stop), 7], "stop_mask": [1, 0],
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
    n = 2 + seed_of("human", conversation_id) % 6
    return [{"t": t, "n_real": n, "real_final": t == n, "user_prompt": "fake"} for t in range(1, n + 1)]


def _fake_task1_sample(self, conversation_id, t, user_prompt, real_final, G, temperature, top_p, seed):
    out = []
    for g in range(G):
        rng = random.Random(seed_of("fake-t1", conversation_id, t, g, seed))
        k = 3 if real_final else 1
        stop = rng.random() < _sig(self.learner.theta[k])
        out.append({"t": t, "replicate": g, "real_final": bool(real_final), "ended_planner": stop,
                    "planner_unparsed": False, "planner_hit_max_new": False,
                    "reward": float(stop == bool(real_final)),
                    "planner_gen": {"prompt_ids": [k, t], "gen_ids": [int(stop), 7], "stop_mask": [1, 0],
                                    "temperature": temperature, "top_p": top_p, "seed": g}})
    out[0]["aux"] = {"prompt_ids": [k, t], "prefix_ids": [], "target_ids": [int(bool(real_final))],
                     "want_end": bool(real_final), "gen_len": 2}
    return out


FakeEnv.task1_prompts = _fake_task1_prompts
FakeEnv.task1_sample = _fake_task1_sample


def stub_llm_transport(request):
    """Deterministic offline stand-in for the LLM controller (dry run / tests)."""
    cur = json.loads(request["messages"][1]["content"])["current"]
    if "factors" in request["messages"][0]["content"]:            # v4 factor controller
        prop = {"factors": {k: 1.25 for k in cur}, "rationale": "stub"}
    else:
        prop = {k: v * 1.5 if v else 0.01 for k, v in cur.items()}
        prop["rationale"] = "stub"
    return {"choices": [{"message": {"content": json.dumps(prop)}}]}


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
        self.p_best = os.path.join(a.out, "best.json")
        self.split = load_split(a.splits, a.fold)
        self.judge_info = check_judge_manifest(a.judge_adapter, self.split, strict=not a.judge_manifest_train_all)
        self.io_lock = threading.Lock()
        user_cfg = json.load(open(a.config, encoding="utf-8")) if a.config else {}
        # pend: reward v4 (D1(b): human length DISTRIBUTION matching) with format constraints on
        # (declared defaults; a --config file may override)
        base_reward = {"version": "v4", "lambda_unparsed": 1.0, "lambda_hit_max_new": 1.0} if a.arm == "pend" else {}
        cfg0 = RC.initial_cfg(**{**base_reward, **user_cfg.get("reward", {}), "lr": a.lr, "kl_coef": a.kl,
                                 "w_aux": a.stop_sup_weight})
        self.selection_cfg = copy.deepcopy(cfg0)          # fixed forever: comparable validation scores
        self.acfg = RA.algo_cfg(**user_cfg.get("algo", {}))
        self.controller = RC.make_controller(
            a.controller, cfg0, log_path=os.path.join(a.out, "llm_controller.jsonl"),
            transport=stub_llm_transport if (a.dry_run and a.controller == "llm") else None,
            **user_cfg.get(a.controller, {}))
        self.cfg = copy.deepcopy(cfg0)
        self.history, self.best, self.update_done = [], None, 0
        self.task1_base = None       # Task 1 stop metrics of the starting policy (update 0 validation)
        self.aux_anneal_start = None # D2: first update at which validation Task 1 term_f1 beat task1_base
        self.p_h = None              # v4: smoothed human length distribution of the TRAIN conversations
        self.env = self.learner = None
        self.n_new_episodes = 0
        self.config_record = {"args": vars(a), "cfg0": cfg0, "selection_cfg": self.selection_cfg,
                              "algo_cfg": self.acfg, "algo_bounds": RA.ALGO_BOUNDS, "lora": RA.LORA,
                              "controller": self.controller.describe(), "user_config": user_cfg}

    # -------------------------------------------------------- setup
    def build(self):
        a = self.a
        if a.dry_run:
            self.learner = FakeLearner(a.algo, self.acfg, self.cfg["lr"], seed=a.seed)
            self.env = FakeEnv(self.learner)
            self.set_p_h()
            return
        from task2_env import PlannerLM, Task2Env
        from goal_judge import GoalJudge
        planner = PlannerLM(a.planner_path, gpu=a.gpu, nf4=a.planner_nf4, dtype=a.planner_dtype,
                            adapter=None, trainable=False)
        model = RA.setup_policy(planner, init_adapter=a.init_adapter)
        planner.adapter = "rl:%s" % a.out
        if a.planner_backend == "vllm":
            # Planner generation by the vLLM server; the HF model stays the learner (it re-scores every token)
            import vllm_planner
            if a.init_adapter:
                raise SystemExit("--init-adapter is merged into the HF base; the vLLM server serves the plain base")
            planner.remote = vllm_planner.VLLMPlanner(a.vllm_url)
        judge = GoalJudge(a.judge_base, adapter=a.judge_adapter, gpu=a.gpu) if a.arm != "pend" else None
        self.env = Task2Env(arm=a.arm, gpu=a.gpu, planner=planner, judge=judge, batch=bool(a.batch), max_batch=a.max_batch,
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

    def aux_weight(self, u, cfg):
        """Effective stop-supervision weight = cfg["w_aux"] (initial --stop-sup-weight, then tuned by the LLM
        controller from TRAIN statistics) x the D2 anneal: 1 until validation Task 1 term_f1 first beats the
        untrained policy's, then linearly to 0 over --stop-sup-anneal updates."""
        w = float(cfg["w_aux"])
        if self.aux_anneal_start is None or self.a.stop_sup_anneal <= 0:
            return w
        return w * max(0.0, 1.0 - (u - self.aux_anneal_start) / float(self.a.stop_sup_anneal))

    def meta(self, kind):
        a = self.a
        row = {"kind": kind, "time": time.time(), "code_sha256": code_shas(),
               "config_sha256": sha_bytes(json.dumps(self.config_record, sort_keys=True, default=str).encode()),
               "config": self.config_record, "splits_sha256": self.split["sha256"], "fold": a.fold,
               "judge_manifest": self.judge_info,
               "judge_adapter_sha256": None if a.dry_run else sha_path(a.judge_adapter),
               "init_adapter_sha256": sha_path(a.init_adapter) if a.init_adapter else None,
               "config_file_sha256": sha_file(a.config) if a.config else None}
        return row

    def check_provenance(self, row):
        prev = read_jsonl(os.path.join(self.a.out, "run_meta.jsonl"))
        if not prev:
            return
        first = prev[0]
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
                 "best": self.best, "update_row": update_row, "task1_base": self.task1_base,
                 "aux_anneal_start": self.aux_anneal_start,
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

    def manifest(self):
        """What this policy was trained on (checked by the evaluation CLIs before any validation/test run)."""
        a = self.a
        return {"kind": "planner_rl", "fold": self.split["fold"], "splits_sha256": self.split["sha256"],
                "train_scenarios": sorted(self.split["train"]), "train_conversations": sorted(self.split["train_all"]),
                "fewshot_pool": sorted(self.split["train_all"]) if a.fewshot == "fold" else [],
                "p_h_source": "train_all", "validation_used_for": "checkpoint selection only",
                "planner_path": a.planner_path, "init_adapter": a.init_adapter,
                "init_adapter_sha256": sha_path(a.init_adapter) if a.init_adapter else None,
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
        self.cfg, self.history, self.best, self.update_done = st["cfg"], st["history"], st["best"], u
        self.task1_base = st.get("task1_base")
        self.aux_anneal_start = st.get("aux_anneal_start")
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

    # -------------------------------------------------------- rollouts
    def scenarios_for(self, u):
        rng = random.Random(seed_of(self.a.seed, "scenarios", u))
        k = self.a.scenarios_per_update
        pool = sorted(self.split["train"])
        if k <= len(pool):
            return rng.sample(pool, k)
        return [rng.choice(pool) for _ in range(k)]

    def sync_generation_policy(self, pv):
        """The vLLM server must generate with the policy the learner holds: the adapter of checkpoint pv (saved
        from this learner) is loaded under a name carrying pv and its sha; every generation records that name."""
        remote = getattr(getattr(self.env, "planner", None), "remote", None)
        if remote is None:
            self.gen_name = None
            return None
        self.gen_name = remote.use_adapter(os.path.join(self.ckpt_dir(pv), "adapter"), "p%d" % pv)
        return self.gen_name

    def rollouts(self, u):
        a, pv, psha = self.a, u - 1, self.learner.policy_sha()
        self.sync_generation_policy(pv)
        reuse = {}
        for r in read_jsonl(self.p_roll):
            if r["update"] == u and r["policy_version"] == pv and r["policy_sha"] == psha:
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

    def task1_rollouts(self, u):
        """Task 1 stop groups on REAL train conversations: for --task1-convs conversations (seeded by u,
        train split only), at the person's last message and at one earlier message (t >= 2), G sampled
        Planner decisions from the state the current policy reaches greedily; reward 1 when
        end_session agrees with the real person (the real last message is the only positive)."""
        a, pv, psha = self.a, u - 1, self.learner.policy_sha()
        self.t1_skipped_capped = 0
        if a.task1_convs <= 0:
            return []
        reuse = {}
        for r in read_jsonl(self.p_roll_t1):
            if r["update"] == u and r["policy_version"] == pv and r["policy_sha"] == psha:
                reuse[(r["conversation_id"], r["t"])] = r
        rng = random.Random(seed_of(a.seed, "task1", u))
        pool = sorted(self.split["train_all"])     # Task 1 needs no requirement shards: every train session
        cids = rng.sample(pool, min(a.task1_convs, len(pool)))
        jobs = []
        for cid in cids:
            assert cid in self.split["train_all"] and cid not in self.split["forbidden"], "non-train conversation %r in a Task 1 group" % cid
            jobs.append((cid, rng.random()))

        def run_conv(job):
            cid, x = job
            n = self.env.human_turns(cid)
            if n < 2:
                return []                    # turn 1 never ends: a one-message conversation has no stop decision
            pos = [n] + ([2 + int(x * (n - 2))] if n >= 3 else [])
            rows = [reuse[(cid, t)] for t in pos if (cid, t) in reuse]
            missing = [t for t in pos if (cid, t) not in reuse]
            if not missing:
                return rows
            prompts = self.env.task1_prompts(cid)
            assert len(prompts) == n
            capped_before = [t for t in missing if any(p.get("emitted_capped") for p in prompts[: t - 1])]
            if capped_before:
                with self.io_lock:
                    self.t1_skipped_capped += len(capped_before)
            for t in [t for t in missing if t not in capped_before]:
                pr = prompts[t - 1]
                assert pr["t"] == t and pr["real_final"] == (t == n)
                smp = self.env.task1_sample(cid, t, pr["user_prompt"], pr["real_final"], a.G,
                                            a.temperature, a.top_p, seed_of(a.seed, "t1s", u))
                row = {"update": u, "conversation_id": cid, "t": t, "n_real": n, "real_final": t == n,
                       "split": "train", "policy_version": pv, "policy_sha": psha, "samples": smp, "time": time.time()}
                with self.io_lock:
                    append_jsonl(self.p_roll_t1, row)
                rows.append(row)
            return rows

        out = []
        if a.rollout_workers <= 1:
            for job in jobs:
                out += run_conv(job)
        else:
            with ThreadPoolExecutor(max_workers=a.rollout_workers) as ex:
                for rows in ex.map(run_conv, jobs):
                    out += rows
        return sorted(out, key=lambda r: (r["conversation_id"], r["t"]))

    # -------------------------------------------------------- one update
    def one_update(self, u):
        t0 = time.time()
        cfg = copy.deepcopy(self.cfg)
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
        stop_credit = self.a.stop_credit and cfg["version"] in ("v3", "v4")
        if stop_credit:
            # stop credit assignment: the length term (v3 |T - target|, v4 log p_h(T) - log q(T)) is caused
            # only by the end_session decisions, so its group advantage goes to the end_session value tokens
            # of every decision step; coverage and the format constraints keep the sequence-level advantage
            # normalised once by the group's std of the TOTAL reward (not per part), so w_cov / w_dist and
            # the controller's factors keep their effect on the gradient
            advs, advs_stop = [], []
            for rs, sp in zip(rew_groups, stop_groups):
                a1, a2 = RA.split_group_advantages(rs, sp, self.acfg["adv_eps"], self.acfg["min_group_std"])
                advs.append(a1)
                advs_stop.append(a2)
            skipped = sum(1 for a1 in advs if a1 is None)
        else:
            advs, skipped = RA.advantages_for_groups(rew_groups, self.a.algo, self.acfg)
            advs_stop = [None] * len(groups)
        for grp, rs, ad, ads in zip(groups, rew_groups, advs, advs_stop):
            if ad is None and ads is None:
                continue
            for j, (row, R) in enumerate(zip(grp, rs)):
                for s_ in RA.episode_samples(row["episode"], policy_version=row["policy_version"]):
                    s_["ret"], s_["adv"] = R, (ad[j] if ad is not None else 0.0)
                    s_["adv_stop"] = ads[j] if ads is not None else 0.0
                    samples.append(s_)
        # Task 1 stop groups (same policy version; one Planner step per sample). A sample that is not a
        # decision (unparsed / capped / no valid end_session) has reward 0 and no stop mask.
        _t = time.time()
        t1rows = self.task1_rollouts(u)
        timing["task1_groups_s"] = round(time.time() - _t, 1)
        t1_rewards = [[x["reward"] for x in r["samples"]] for r in t1rows]
        t1_advs, t1_skipped = RA.advantages_for_groups(t1_rewards, self.a.algo, self.acfg) if t1rows else ([], 0)
        for r, rs, ad in zip(t1rows, t1_rewards, t1_advs):
            assert r["policy_version"] == pv and r["policy_sha"] == psha, "off-policy task1 rollout"
            assert r["t"] >= 2, "Task 1 stop group at turn 1"
            if ad is None:
                continue
            for x, R, A in zip(r["samples"], rs, ad):
                pseudo = {"conversation_id": r["conversation_id"], "replicate": x["replicate"],
                          "trace": [{"t": x["t"], "planner_gen": x["planner_gen"]}]}
                for s_ in RA.episode_samples(pseudo, policy_version=pv):
                    if self.a.stop_credit:
                        s_["ret"], s_["adv"], s_["adv_stop"], s_["source"] = R, 0.0, A, "task1"
                    else:
                        s_["ret"], s_["adv"], s_["source"] = R, A, "task1"
                    samples.append(s_)
        t1_all = [x for r in t1rows for x in r["samples"]]
        t1_hist = None
        if t1_all:
            fin = [x for x in t1_all if x["real_final"]]
            mid = [x for x in t1_all if not x["real_final"]]
            t1_hist = {"n": len(t1_all), "acc": sum(x["reward"] for x in t1_all) / len(t1_all),
                       "end_at_final": (sum(x["ended_planner"] for x in fin) / len(fin)) if fin else None,
                       "end_at_nonfinal": (sum(x["ended_planner"] for x in mid) / len(mid)) if mid else None,
                       "n_not_decisions": sum(1 for x in t1_all if not x.get("decision_valid", True)),
                       "n_skipped_capped_history": self.t1_skipped_capped,
                       "groups_skipped_zero_std": t1_skipped}
        assert all(s_["policy_version"] == pv for s_ in samples), "sample from another policy version"
        aux, w_aux = [], self.aux_weight(u, cfg)
        if w_aux > 0:
            for r in t1rows:
                x = (r["samples"] or [{}])[0].get("aux")
                if x is not None:
                    assert x["want_end"] == r["real_final"], "stop-supervision label disagrees with the human"
                    aux.append(dict(x, weight=w_aux))
        _t = time.time()
        if getattr(self, "gen_name", None):
            bad = sorted({s_.get("gen_adapter") for s_ in samples if s_.get("gen_adapter") != self.gen_name})
            if bad:
                raise AssertionError("samples generated by adapter(s) %r, the policy is %r: off-policy generation"
                                     % (bad[:3], self.gen_name))
        stats = self.learner.update(samples, cfg, seed=seed_of(self.a.seed, "update", u), aux=aux or None)
        timing["learner_s"] = round(time.time() - _t, 1)
        mm = stats.get("behav_mismatch_mean")
        if mm is not None and mm > self.a.behav_mismatch_abort:
            # before the checkpoint is written: a resume redoes this update
            raise SystemExit("update %d: mean |log pi_learner - log pi_vllm| = %.4f > %.4f (phase-0 measured 0.017): "
                             "the served adapter does not match the learner?" % (u, mm, self.a.behav_mismatch_abort))
        agg = RR.aggregate(rewarded)
        tm = int(cfg["t_max"])
        turn_hist = [0] * (tm + 1)
        for e in all_clean:
            turn_hist[min(max(int(e["emitted_user_turns"]), 0), tm)] += 1
        hist = {"update": u, "split": "train", "reward_version": cfg["version"], "task1_train": t1_hist, **agg,
                "n_groups": len(groups), "n_groups_skipped_zero_std": skipped, "n_unclean_episodes": n_unclean,
                "n_dropped_singleton_episodes": n_singletons,
                "shadow_reward_mean": sum(shadow) / len(shadow), "turn_hist": turn_hist, "p_h": self.p_h,
                "aux_weight": w_aux, "lr": cfg["lr"], "kl_coef": cfg["kl_coef"],
                "aux_stats": {k: stats.get(k) for k in ("aux_n", "aux_loss", "aux_grad_norm", "aux_p_correct_before",
                                                         "aux_p_correct_end")},
                **{k: stats.get(k) for k in ("loss", "kl", "ratio_mean", "clip_frac", "grad_norm", "n_tokens",
                                              "value_mse", "ratio_init_maxdev", "rl_grad_norm", "tis_w_mean",
                                              "tis_capped_frac", "behav_mismatch_mean")}}
        self.history.append(hist)
        self.cfg = self.controller.propose(self.history)
        self.update_done = u
        row = {"update": u, "policy_version_rollouts": pv, "policy_version_after": u, "algo": self.a.algo,
               "controller": self.a.controller, "cfg_used": cfg, "cfg_used_sha256": RR.cfg_sha(cfg),
               "reward_ctx": ctx, "next_cfg": self.cfg, "scenarios": [g[0]["conversation_id"] for g in groups_all],
               "train_aggregate": hist, "learner_stats": stats, "n_samples": len(samples),
               "timing": timing,
               "controller_failures": getattr(self.controller, "n_failures", None),
               "controller_rollbacks": getattr(self.controller, "n_rollbacks", None),
               "policy_sha_after": self.learner.policy_sha(), "time": time.time(), "update_s": time.time() - t0}
        self.save_checkpoint(u, row)
        append_jsonl(self.p_upd, row)
        return row

    # -------------------------------------------------------- validation
    def validate(self, u):
        """Task 2 episodes on the validation ids with the SAMPLED Planner (D5: --val-temperature, one
        replicate per --val-seeds entry) + Task 1 (greedy, as the benchmark); logged only to validation.jsonl;
        drives best.json and the D2 annealing trigger. Never enters history or the controller."""
        _tv = time.time()
        self.sync_generation_policy(u)          # validation generates with the policy after update u
        a = self.a
        prev = read_jsonl(self.p_val)
        if any(r.get("kind") == "summary" and r["update"] == u for r in prev):
            return
        done = {(r["conversation_id"], r["seed"]): r for r in prev if r.get("kind") == "episode" and r["update"] == u}
        done_t1 = {r["conversation_id"]: r for r in prev if r.get("kind") == "task1" and r["update"] == u}
        psha = self.learner.policy_sha()
        jobs = []
        for cid in sorted(self.split["validation"]):
            assert cid in self.split["validation"] and cid not in self.split["train"] \
                and cid not in self.split["train_all"], "validation id %r is also a training id" % cid
            for s in a.val_seeds:
                jobs.append((cid, s))

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
            if r is None or r["policy_sha"] != psha:
                r = {"kind": "task1", "update": u, "policy_sha": psha, "conversation_id": cid,
                     "split": "validation", "task1": self.env.run_task1(cid), "time": time.time()}
                with self.io_lock:
                    append_jsonl(self.p_val, r)
            return r

        if not hasattr(self.env, "run_task1"):
            raise RuntimeError("validation needs Task 1 (run_task1): D2, task1_base and the selection score use it")
        workers = max(1, a.rollout_workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            vrows = list(ex.map(run_val, jobs))
            t1rows = list(ex.map(run_t1, sorted(self.split["validation"])))
        eps = [r["episode"] for r in vrows if r["episode"]["clean"]]
        n_unclean = len(vrows) - len(eps)
        withheld = n_unclean > 0                # every checkpoint is scored on the SAME validation set, or not at all
        vctx = self.reward_ctx(eps, self.selection_cfg) if eps else None
        totals = [RR.reward(e, self.selection_cfg, vctx)["total"] for e in eps]
        score = sum(totals) / len(totals) if totals else float("nan")
        turn_stats = None
        if eps and all("human_turns" in e for e in eps):
            d = [e["emitted_user_turns"] - e["human_turns"] for e in eps]
            turn_stats = {"sim_turns_mean": sum(e["emitted_user_turns"] for e in eps) / len(eps),
                          "human_turns_mean": sum(e["human_turns"] for e in eps) / len(eps),
                          "abs_diff_mean": sum(abs(x) for x in d) / len(d),
                          "coverage_mean": sum(float(e["coverage"]) for e in eps) / len(eps),
                          "turn_w1": turn_w1([e["emitted_user_turns"] for e in eps],
                                             [min(int(e["human_turns"]), int(self.selection_cfg["t_max"])) for e in eps]),
                          "end_kinds": {k: sum(e["end_kind"] == k for e in eps) for k in sorted({e["end_kind"] for e in eps})}}
        t1 = T1.task1_stop_metrics([r["task1"] for r in t1rows]) if t1rows else None
        if t1 is not None and self.task1_base is None:
            self.task1_base = {"update": u, **t1}
            sp = os.path.join(self.ckpt_dir(self.update_done), "state.json")
            st = json.load(open(sp))
            st["task1_base"] = self.task1_base         # persisted now: a crash in update 1 must not lose it
            write_json_atomic(sp, st)
        if t1 is not None and self.aux_anneal_start is None and u > self.task1_base["update"] \
                and t1["term_f1"] > self.task1_base["term_f1"]:
            self.aux_anneal_start = u            # D2: from here the stop supervision goes linearly to 0
            sp = os.path.join(self.ckpt_dir(self.update_done), "state.json")
            st = json.load(open(sp))
            st["aux_anneal_start"] = u
            write_json_atomic(sp, st)
        t1_ok = True if t1 is None else T1.within_tolerance(t1, self.task1_base, a.task1_tol)
        # checkpoint selection (user 2026-09-26): with ~8 validation episodes the v4 dist term is dominated by
        # the smoothing, so Task 2 is scored by the turn-count distribution W1 against the validation people
        # plus coverage; Task 1 by the M2 term_f1. The v4 validation reward is still logged (not selected on).
        sel = None
        if not withheld and turn_stats is not None and turn_stats.get("turn_w1") is not None:
            sel = (a.w_sel_cov * turn_stats["coverage_mean"] - a.w_sel_w1 * turn_stats["turn_w1"]
                   + (a.w_sel_task1 * t1["term_f1"] if t1 is not None else 0.0))
        summary = {"kind": "summary", "update": u, "policy_sha": psha, "split": "validation",
                                  "selection_withheld": "unclean validation episode(s) after %d re-runs" % VAL_RETRIES
                                  if withheld else None,
                                  "validation_s": round(time.time() - _tv, 1),
                                  "n_episodes": len(totals), "n_unclean_episodes": n_unclean,
                                  "val_temperature": a.val_temperature, "val_seeds": a.val_seeds,
                                  "mean_reward_selection": score, "aux_anneal_start": self.aux_anneal_start,
                                  "turn_stats": turn_stats, "task1": t1, "task1_base": self.task1_base,
                                  "task1_within_tol": t1_ok, "task1_tol": a.task1_tol,
                                  "selection_score": sel, "w_sel_task1": a.w_sel_task1,
                                  "w_sel_w1": a.w_sel_w1, "w_sel_cov": a.w_sel_cov,
                                  "selection_formula": "w_sel_cov*coverage_mean - w_sel_w1*turn_w1 + w_sel_task1*task1.term_f1",
                                  "selection_cfg_sha256": RR.cfg_sha(self.selection_cfg), "time": time.time()}
        if sel is not None and (self.best is None or sel > self.best["selection_score"]):
            self.best = {"update": u, "checkpoint": os.path.relpath(self.ckpt_dir(u), a.out).replace("\\", "/"),
                         "policy_sha": psha, "mean_reward_selection": score, "selection_score": sel,
                         "selection_cfg_sha256": RR.cfg_sha(self.selection_cfg)}
            write_json_atomic(self.p_best, self.best)
            # keep the checkpoint's own record of best in sync
            sp = os.path.join(self.ckpt_dir(self.update_done), "state.json")
            st = json.load(open(sp))
            st["best"] = self.best
            write_json_atomic(sp, st)
        append_jsonl(self.p_val, summary)

    # -------------------------------------------------------- main loop
    def run(self):
        a = self.a
        self.build()
        resumed = self.load_checkpoint() if a.resume else False
        if not resumed and os.path.exists(os.path.join(self.ckpt_root, "LATEST.json")):
            raise SystemExit("%s already has checkpoints; pass --resume or use a new --out" % a.out)
        row = self.meta("resume" if resumed else "start")
        self.check_provenance(row)
        append_jsonl(os.path.join(a.out, "run_meta.jsonl"), row)
        if not resumed:
            random.seed(a.seed)
            self.save_checkpoint(0, None)             # policy_version 0 = the starting policy
        if a.val_every > 0 and self.update_done % a.val_every == 0:
            self.validate(self.update_done)           # also completes a validation cut short by a crash
        for u in range(self.update_done + 1, a.updates + 1):
            r = self.one_update(u)
            print(json.dumps({"update": u, "reward_mean": r["train_aggregate"]["reward_mean"],
                              "skipped": r["train_aggregate"]["n_groups_skipped_zero_std"],
                              "n_samples": r["n_samples"], "update_s": round(r["update_s"], 1)}), flush=True)
            if a.val_every > 0 and u % a.val_every == 0:
                self.validate(u)
        return self


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
                    help="reported: whether Task 1 term_f1 / premature stay within this of update 0")
    ap.add_argument("--task1-convs", type=int, default=4,
                    help="Task 1 stop groups per update: real TRAIN conversations (last + one earlier message, G samples each); 0 = off")
    ap.add_argument("--stop-sup-weight", type=float, default=1.0,  # initial w_aux; the LLM controller tunes it
                    help="auxiliary stop-token supervision on the Task 1 positions (human end/continue); 0 = off (pure GRPO)")
    ap.add_argument("--stop-sup-anneal", type=int, default=10,
                    help="D2: updates over which the stop supervision goes linearly to 0 once validation Task 1 "
                         "term_f1 beats the untrained policy's; 0 = never anneal")
    ap.add_argument("--stop-credit", type=int, choices=(0, 1), default=1,
                    help="1: turn-count and Task 1 advantages act only on the end_session value tokens")
    ap.add_argument("--w-sel-w1", type=float, default=1.0,
                    help="checkpoint selection: weight of the validation turn-count W1 (turns; subtracted)")
    ap.add_argument("--w-sel-cov", type=float, default=1.0, help="checkpoint selection: weight of validation coverage")
    ap.add_argument("--w-sel-task1", type=float, default=1.0,
                    help="checkpoint selection: weight of validation Task 1 term_f1 (M2)")
    ap.add_argument("--splits", default="/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json")
    ap.add_argument("--algo", choices=RA.ALGOS, default="grpo")
    ap.add_argument("--controller", choices=("fixed", "dual", "llm"), default="llm",
                    help="spec: llm (the v4 factor controller); fixed/dual need --ablation")
    ap.add_argument("--planner-path")
    ap.add_argument("--planner-dtype", default="bfloat16")
    ap.add_argument("--planner-nf4", action="store_true")
    ap.add_argument("--init-adapter", default=None, help="SFT warm-start adapter, merged into the base (KL reference)")
    ap.add_argument("--judge-adapter", default=None)
    ap.add_argument("--judge-base", default=None)
    ap.add_argument("--judge-manifest-train-all", action="store_true",
                    help="accept a judge whose train_scenarios lie in train_all (without shards) rather than train")
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--scenarios-per-update", type=int, default=4)
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--lr", type=float, default=RC.TRAIN_DEFAULTS["lr"])
    ap.add_argument("--kl", type=float, default=RC.TRAIN_DEFAULTS["kl_coef"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--val-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--val-temperature", type=float, default=0.7, help="D5: validation Task 2 uses the sampled Planner")
    ap.add_argument("--val-top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None, help='JSON: {"reward": {...}, "algo": {...}, "dual": {...}, "llm": {...}}')
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-code-change", action="store_true")
    ap.add_argument("--keep-optimizer-last", type=int, default=3,
                    help="delete optimizer.pt of checkpoints older than this many updates (adapters kept); 0 = keep all")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="fake env + pure-python learner (no torch, no GPU)")
    ap.add_argument("--dry-run-crash-after-episodes", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.G < 2:
        ap.error("--G must be >= 2 (group baselines)")
    if not (a.temperature > 0):
        ap.error("training rollouts need --temperature > 0")
    if a.stop_credit and a.algo != "grpo":
        ap.error("--stop-credit 1 is implemented for grpo")
    off = {k: getattr(a, k) for k in SPEC if getattr(a, k) != SPEC[k]}
    if a.controller != "llm":
        off["controller"] = a.controller
    if a.planner_path and "Qwen3-4B-Instruct-2507" not in a.planner_path:
        off["planner_path"] = a.planner_path          # the spec's Planner is Qwen3-4B-Instruct-2507
    if a.config:
        cj = json.load(open(a.config, encoding="utf-8"))
        for sect in ("reward", "algo", "llm", "dual"):
            if cj.get(sect):
                off["config." + sect] = cj[sect]            # any override of the declared defaults
    if a.planner_backend != "vllm" and not a.dry_run:
        off["planner_backend"] = a.planner_backend
    for k, want in (("stop_credit", 1), ("kl", RC.TRAIN_DEFAULTS["kl_coef"]), ("val_temperature", 0.7),
                    ("lr", RC.TRAIN_DEFAULTS["lr"]),
                    ("stop_sup_anneal", 10), ("algo", "grpo")):
        if getattr(a, k) != want:
            off[k] = getattr(a, k)
    if sorted(a.val_seeds) != [0, 1]:
        off["val_seeds"] = a.val_seeds
    if a.task1_convs <= 0:
        off["task1_convs"] = a.task1_convs
    if a.stop_sup_weight == 0:
        off["stop_sup_weight"] = 0.0
    elif not (0.01 <= a.stop_sup_weight <= 5.0):
        ap.error("--stop-sup-weight must be 0 (off) or within the controller bounds [0.01, 5]")
    if off and not a.ablation:
        ap.error("settings %r differ from the pend spec %r; name the ablation with --ablation" % (off, SPEC))
    if not (a.val_temperature > 0):
        ap.error("--val-temperature must be > 0 (D5: sampled Planner at validation)")
    if not a.dry_run:
        for k in (("planner_path",) if a.arm == "pend" else ("planner_path", "judge_adapter", "judge_base")):
            if getattr(a, k) is None:
                ap.error("--%s is required (except with --dry-run)" % k.replace("_", "-"))
    return a


def main(argv=None):
    return Trainer(parse_args(argv)).run()


if __name__ == "__main__":
    main()
