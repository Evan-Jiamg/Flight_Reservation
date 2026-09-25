#!/usr/bin/env python3
"""Planner RL on Task 2 (GRPO / RLOO / PPO; fixed / dual-ascent / LLM controller).

Loop (update u = 1, 2, ...; the policy that generates update u's rollouts has policy_version u-1):
  1. sample --scenarios-per-update scenarios (seeded by (seed, u)) from splits[fold]["train"] ONLY;
     every id is asserted to be in train and not in forbidden_for_training;
  2. G rollouts each: Task2Env(arm="a2").run_episode(..., planner_temperature>0, record_generation=True);
     each row is appended to rollouts.jsonl with update, policy_version and policy_sha;
  3. rewards = rl_reward.reward_v2(episode, cfg) with the controller's current cfg;
  4. advantages (rl_algos, pure python) -> samples (ids exactly as recorded) -> learner.update;
     assert every sample was produced by the current policy_version (on-policy);
  5. checkpoint EVERY update (adapter, optimizer, value head, controller state, RNG states, update index,
     policy_version, history) atomically; then updates.jsonl;
  6. controller.propose(train aggregates only) -> cfg for the next update;
  7. every --val-every updates: greedy episodes on splits[fold]["validation"] -> validation.jsonl only;
     best.json = best checkpoint by mean validation reward under the FIXED initial cfg (selection_cfg).
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
TIME_KEYS = ("time", "wall_s", "rollout_s", "update_s")


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
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass          # a torn last line after a crash; the row is regenerated
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
            self._aux_step(aux, cfg, out)
            return out
        st = self._update(samples, cfg, seed)
        if aux:
            self._aux_step(aux, cfg, st)
        return st

    def _aux_step(self, aux, cfg, st):
        lr = cfg["lr"] * self.lr_scale
        ps = []
        for x in aux:
            k = x["prompt_ids"][0]
            p = _sig(self.theta[k])
            ps.append(p if x["target_ids"][0] == 1 else 1 - p)
            grad = (1 - p) if x["target_ids"][0] == 1 else -p
            self.theta[k] += lr * float(x["weight"]) * grad / len(aux)
        st["aux_n"], st["aux_p_correct_before"] = len(aux), sum(ps) / len(ps)
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
            stop = (rng.random() < p) if planner_temperature > 0 else p > 0.5
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
                "human_turns": 2 + seed_of("human", conversation_id) % 6}


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
                     "want_end": bool(real_final)}
    return out


FakeEnv.task1_prompts = _fake_task1_prompts
FakeEnv.task1_sample = _fake_task1_sample


def stub_llm_transport(request):
    """Deterministic offline stand-in for the LLM controller (dry run / tests)."""
    cur = json.loads(request["messages"][1]["content"])["current"]
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
        # pend: reward v3 with format constraints on (declared defaults; a --config file may override)
        base_reward = {"version": "v3", "lambda_unparsed": 1.0, "lambda_hit_max_new": 1.0} if a.arm == "pend" else {}
        cfg0 = RC.initial_cfg(**{**base_reward, **user_cfg.get("reward", {}), "lr": a.lr, "kl_coef": a.kl})
        self.selection_cfg = copy.deepcopy(cfg0)          # fixed forever: comparable validation scores
        self.acfg = RA.algo_cfg(**user_cfg.get("algo", {}))
        self.controller = RC.make_controller(
            a.controller, cfg0, log_path=os.path.join(a.out, "llm_controller.jsonl"),
            transport=stub_llm_transport if (a.dry_run and a.controller == "llm") else None,
            **user_cfg.get(a.controller, {}))
        self.cfg = copy.deepcopy(cfg0)
        self.history, self.best, self.update_done = [], None, 0
        self.task1_base = None       # Task 1 stop metrics of the starting policy (update 0 validation)
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
            return
        from task2_env import PlannerLM, Task2Env
        from goal_judge import GoalJudge
        planner = PlannerLM(a.planner_path, gpu=a.gpu, nf4=a.planner_nf4, dtype=a.planner_dtype,
                            adapter=None, trainable=False)
        model = RA.setup_policy(planner, init_adapter=a.init_adapter)
        planner.adapter = "rl:%s" % a.out
        judge = GoalJudge(a.judge_base, adapter=a.judge_adapter, gpu=a.gpu) if a.arm != "pend" else None
        self.env = Task2Env(arm=a.arm, gpu=a.gpu, planner=planner, judge=judge, batch=bool(a.batch), max_batch=a.max_batch,
                            implicit_profile=bool(a.implicit_profile), selector=a.selector)
        if a.fewshot == "fold":
            from task2_env import make_fewshot_pool
            pool_ids = list(self.split["train_all"])
            assert not set(pool_ids) & self.split["forbidden"], "few-shot pool intersects validation/test"
            self.env.fewshot = make_fewshot_pool(self.env.recs, pool_ids)
        self.learner = RA.TorchLearner(model, a.algo, self.acfg, lr=self.cfg["lr"], seed=a.seed)
        self.config_record["env"] = self.env.describe()
        self.config_record["trainable_params"] = self.learner.trainable_names()[:8] + ["..."]

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
        for k in ("code_sha256", "splits_sha256", "judge_adapter_sha256", "init_adapter_sha256"):
            if first[k] != row[k] and not self.a.allow_code_change:
                raise SystemExit("provenance mismatch on resume (%s); rerun in a new --out or pass --allow-code-change" % k)
        for k in ("algo", "controller", "fold", "G", "scenarios_per_update", "seed", "temperature", "top_p"):
            if first["config"]["args"][k] != row["config"]["args"][k]:
                raise SystemExit("argument %s changed on resume (%r -> %r)" % (k, first["config"]["args"][k],
                                                                             row["config"]["args"][k]))

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
                 "python_rng": [rng["python"][0], list(rng["python"][1]), rng["python"][2]]}
        write_json_atomic(os.path.join(tmp, "state.json"), state)
        if os.path.exists(d):             # only possible if LATEST was not advanced after a crash
            shutil.rmtree(d)
        os.replace(tmp, d)
        write_json_atomic(os.path.join(self.ckpt_root, "LATEST.json"), {"update": u, "dir": os.path.basename(d)})
        if not self.a.dry_run and self.a.keep_optimizer_last > 0:
            old = self.ckpt_dir(u - self.a.keep_optimizer_last)
            for fn in ("optimizer.pt",):
                if u - self.a.keep_optimizer_last >= 1 and os.path.exists(os.path.join(old, fn)):
                    os.remove(os.path.join(old, fn))   # adapters are kept for every update

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

    def rollouts(self, u):
        a, pv, psha = self.a, u - 1, self.learner.policy_sha()
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
            prompts = None
            rows = []
            n = None
            for key in list(reuse):
                if key[0] == cid:
                    rows.append(reuse[key])
            if rows:
                return rows
            prompts = self.env.task1_prompts(cid)
            n = len(prompts)
            if n < 2:
                return []                    # turn 1 never ends: a one-message conversation has no stop decision
            pos = [n] + ([2 + int(x * (n - 2))] if n >= 3 else [])
            for t in pos:
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
        groups = self.rollouts(u)
        pv = u - 1
        samples, rewarded = [], []
        rew_groups = []
        for grp in groups:
            rs = []
            for row in grp:
                assert row["policy_version"] == pv and row["policy_sha"] == self.learner.policy_sha(), "off-policy rollout"
                rw = RR.reward(row["episode"], cfg)
                rs.append(rw["total"])
                rewarded.append((row["episode"], rw))
            rew_groups.append(rs)
        stop_credit = self.a.stop_credit and cfg["version"] == "v3"
        if stop_credit:
            # stop credit assignment: the turn-count term is caused only by the end_session decisions,
            # so its group advantage goes to the end_session value tokens of every step; coverage and
            # the format constraints keep the sequence-level advantage
            comps = {id(row): rw for row, (_, rw) in zip([r for g in groups for r in g], rewarded)}
            len_groups = [[-cfg["w_len"] * comps[id(row)]["components"]["len_err"] for row in grp] for grp in groups]
            seq_groups = [[R - L for R, L in zip(rs, ls)] for rs, ls in zip(rew_groups, len_groups)]
            advs, sk1 = RA.advantages_for_groups(seq_groups, self.a.algo, self.acfg)
            advs_stop, sk2 = RA.advantages_for_groups(len_groups, self.a.algo, self.acfg)
            skipped = sum(1 for a1, a2 in zip(advs, advs_stop) if a1 is None and a2 is None)
        else:
            advs, skipped = RA.advantages_for_groups(rew_groups, self.a.algo, self.acfg)
            advs_stop = [None] * len(groups)
        for grp, rs, ad, ads in zip(groups, rew_groups, advs, advs_stop):
            if ad is None and ads is None:
                continue
            for j, (row, R) in enumerate(zip(grp, rs)):
                for s in RA.episode_samples(row["episode"], policy_version=row["policy_version"]):
                    s["ret"], s["adv"] = R, (ad[j] if ad is not None else 0.0)
                    s["adv_stop"] = ads[j] if ads is not None else 0.0
                    samples.append(s)
        # Task 1 stop groups (same policy version; one Planner step per sample)
        t1rows = self.task1_rollouts(u)
        t1_rewards = [[x["reward"] for x in r["samples"]] for r in t1rows]
        t1_advs, t1_skipped = RA.advantages_for_groups(t1_rewards, self.a.algo, self.acfg) if t1rows else ([], 0)
        for r, rs, ad in zip(t1rows, t1_rewards, t1_advs):
            assert r["policy_version"] == pv and r["policy_sha"] == self.learner.policy_sha(), "off-policy task1 rollout"
            if ad is None:
                continue
            for x, R, A in zip(r["samples"], rs, ad):
                pseudo = {"conversation_id": r["conversation_id"], "replicate": x["replicate"],
                          "trace": [{"t": x["t"], "planner_gen": x["planner_gen"]}]}
                for s in RA.episode_samples(pseudo, policy_version=pv):
                    if self.a.stop_credit:
                        s["ret"], s["adv"], s["adv_stop"], s["source"] = R, 0.0, A, "task1"
                    else:
                        s["ret"], s["adv"], s["source"] = R, A, "task1"
                    samples.append(s)
        t1_all = [x for r in t1rows for x in r["samples"]]
        t1_hist = None
        if t1_all:
            fin = [x for x in t1_all if x["real_final"]]
            mid = [x for x in t1_all if not x["real_final"]]
            t1_hist = {"n": len(t1_all), "acc": sum(x["reward"] for x in t1_all) / len(t1_all),
                       "end_at_final": (sum(x["ended_planner"] for x in fin) / len(fin)) if fin else None,
                       "end_at_nonfinal": (sum(x["ended_planner"] for x in mid) / len(mid)) if mid else None,
                       "groups_skipped_zero_std": t1_skipped}
        assert all(s["policy_version"] == pv for s in samples), "sample from another policy version"
        aux = []
        if self.a.stop_sup_weight > 0:
            for r in t1rows:
                x = (r["samples"] or [{}])[0].get("aux")
                if x is not None:
                    assert x["want_end"] == r["real_final"], "stop-supervision label disagrees with the human"
                    aux.append(dict(x, weight=self.a.stop_sup_weight))
        stats = self.learner.update(samples, cfg, seed=seed_of(self.a.seed, "update", u), aux=aux or None)
        agg = RR.aggregate(rewarded)
        hist = {"update": u, "split": "train", "reward_version": cfg["version"], "task1_train": t1_hist, **agg, "n_groups": len(groups), "n_groups_skipped_zero_std": skipped,
                "lr": cfg["lr"], "kl_coef": cfg["kl_coef"],
                **{k: stats.get(k) for k in ("loss", "kl", "ratio_mean", "clip_frac", "grad_norm", "n_tokens",
                                              "value_mse", "ratio_init_maxdev")}}
        self.history.append(hist)
        self.cfg = self.controller.propose(self.history)
        self.update_done = u
        row = {"update": u, "policy_version_rollouts": pv, "policy_version_after": u, "algo": self.a.algo,
               "controller": self.a.controller, "cfg_used": cfg, "cfg_used_sha256": RR.cfg_sha(cfg),
               "next_cfg": self.cfg, "scenarios": [g[0]["conversation_id"] for g in groups],
               "train_aggregate": hist, "learner_stats": stats, "n_samples": len(samples),
               "policy_sha_after": self.learner.policy_sha(), "time": time.time(), "update_s": time.time() - t0}
        self.save_checkpoint(u, row)
        append_jsonl(self.p_upd, row)
        return row

    # -------------------------------------------------------- validation
    def validate(self, u):
        """Greedy episodes on the validation ids; logged only to validation.jsonl; drives best.json."""
        a = self.a
        prev = read_jsonl(self.p_val)
        if any(r.get("kind") == "summary" and r["update"] == u for r in prev):
            return
        done = {(r["conversation_id"], r["seed"]): r for r in prev if r.get("kind") == "episode" and r["update"] == u}
        done_t1 = {r["conversation_id"]: r for r in prev if r.get("kind") == "task1" and r["update"] == u}
        psha = self.learner.policy_sha()
        jobs = []
        for cid in sorted(self.split["validation"]):
            assert cid in self.split["validation"] and cid not in self.split["train"]
            for s in a.val_seeds:
                jobs.append((cid, s))

        def run_val(job):
            cid, s = job
            r = done.get((cid, s))
            if r is None or r["policy_sha"] != psha:
                ep = self.env.run_episode(cid, seed=s, replicate=0, planner_temperature=0.0, planner_top_p=1.0,
                                          record_generation=False)
                rw = RR.reward(ep, self.selection_cfg)
                r = {"kind": "episode", "update": u, "policy_sha": psha, "conversation_id": cid, "seed": s,
                     "split": "validation", "reward_selection": rw, "episode": ep, "time": time.time()}
                with self.io_lock:
                    append_jsonl(self.p_val, r)
            return r

        def run_t1(cid):
            r = done_t1.get(cid)
            if r is None or r["policy_sha"] != psha:
                r = {"kind": "task1", "update": u, "policy_sha": psha, "conversation_id": cid,
                     "split": "validation", "task1": self.env.run_task1(cid), "time": time.time()}
                with self.io_lock:
                    append_jsonl(self.p_val, r)
            return r

        workers = max(1, a.rollout_workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            vrows = list(ex.map(run_val, jobs))
            t1rows = list(ex.map(run_t1, sorted(self.split["validation"]))) if hasattr(self.env, "run_task1") else []
        totals = [r["reward_selection"]["total"] for r in vrows]
        score = sum(totals) / len(totals) if totals else float("nan")
        eps = [r["episode"] for r in vrows]
        turn_stats = None
        if eps and all("human_turns" in e for e in eps):
            d = [e["emitted_user_turns"] - e["human_turns"] for e in eps]
            turn_stats = {"sim_turns_mean": sum(e["emitted_user_turns"] for e in eps) / len(eps),
                          "human_turns_mean": sum(e["human_turns"] for e in eps) / len(eps),
                          "abs_diff_mean": sum(abs(x) for x in d) / len(d),
                          "coverage_mean": sum(float(e["coverage"]) for e in eps) / len(eps),
                          "end_kinds": {k: sum(e["end_kind"] == k for e in eps) for k in sorted({e["end_kind"] for e in eps})}}
        t1 = T1.task1_stop_metrics([r["task1"] for r in t1rows]) if t1rows else None
        if t1 is not None and self.task1_base is None:
            self.task1_base = {"update": u, **t1}
        t1_ok = True if t1 is None else T1.within_tolerance(t1, self.task1_base, a.task1_tol)
        sel = score + (a.w_sel_task1 * t1["term_f1"] if t1 is not None else 0.0)
        append_jsonl(self.p_val, {"kind": "summary", "update": u, "policy_sha": psha, "split": "validation",
                                  "n_episodes": len(totals), "mean_reward_selection": score,
                                  "turn_stats": turn_stats, "task1": t1, "task1_base": self.task1_base,
                                  "task1_within_tol": t1_ok, "task1_tol": a.task1_tol,
                                  "selection_score": sel, "w_sel_task1": a.w_sel_task1,
                                  "selection_cfg_sha256": RR.cfg_sha(self.selection_cfg), "time": time.time()})
        if totals and (self.best is None or sel > self.best["selection_score"]):
            self.best = {"update": u, "checkpoint": os.path.relpath(self.ckpt_dir(u), a.out).replace("\\", "/"),
                         "policy_sha": psha, "mean_reward_selection": score, "selection_score": sel,
                         "selection_cfg_sha256": RR.cfg_sha(self.selection_cfg)}
            write_json_atomic(self.p_best, self.best)
            # keep the checkpoint's own record of best in sync
            sp = os.path.join(self.ckpt_dir(self.update_done), "state.json")
            st = json.load(open(sp))
            st["best"] = self.best
            write_json_atomic(sp, st)

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


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--arm", choices=("pend", "final", "a2"), default="pend",
                    help="policy env: pend = E1.6 base + fixes, Planner judges the goal and ends (no goal judge)")
    ap.add_argument("--batch", type=int, choices=(0, 1), default=1, help="cross-episode dynamic batching of Planner / Ditto generation")
    ap.add_argument("--implicit-profile", type=int, choices=(0, 1), default=0)
    ap.add_argument("--fewshot", choices=("off", "fold"), default="off", help="fold: examples from splits[fold].train_all only")
    ap.add_argument("--selector", choices=("length", "borda"), default="length")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--rollout-workers", type=int, default=1,
                    help="episodes run in threads; GPU calls are serialised inside Task2Env, R0/ledger calls overlap")
    ap.add_argument("--task1-tol", type=float, default=0.05,
                    help="reported: whether Task 1 term_f1 / premature stay within this of update 0")
    ap.add_argument("--task1-convs", type=int, default=4,
                    help="Task 1 stop groups per update: real TRAIN conversations (last + one earlier message, G samples each); 0 = off")
    ap.add_argument("--stop-sup-weight", type=float, default=1.0,
                    help="auxiliary stop-token supervision on the Task 1 positions (human end/continue); 0 = off (pure GRPO)")
    ap.add_argument("--stop-credit", type=int, choices=(0, 1), default=1,
                    help="1: turn-count and Task 1 advantages act only on the end_session value tokens")
    ap.add_argument("--w-sel-task1", type=float, default=1.0,
                    help="checkpoint selection = validation Task 2 reward + this * validation Task 1 term_f1")
    ap.add_argument("--splits", default="/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json")
    ap.add_argument("--algo", choices=RA.ALGOS, default="grpo")
    ap.add_argument("--controller", choices=("fixed", "dual", "llm"), default="fixed")
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
    ap.add_argument("--val-seeds", type=int, nargs="+", default=[0])
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
    if not a.dry_run:
        for k in (("planner_path",) if a.arm == "pend" else ("planner_path", "judge_adapter", "judge_base")):
            if getattr(a, k) is None:
                ap.error("--%s is required (except with --dry-run)" % k.replace("_", "-"))
    return a


def main(argv=None):
    return Trainer(parse_args(argv)).run()


if __name__ == "__main__":
    main()
