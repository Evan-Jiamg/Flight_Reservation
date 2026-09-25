"""Pure-python tests: advantage math (rl_algos) and the --dry-run full loop of train_planner_rl
(fake env + pure-python learner): checkpoint every update, resume (also after a crash in the middle of an
update's rollouts), on-policy versions, train-only rollouts, validation-only best selection, leak gates.
Run: python test_rl_advantages.py"""
import json
import math
import os
import shutil
import tempfile

import rl_algos as RA
import train_planner_rl as T


def close(a, b, tol=1e-9):
    return math.isclose(a, b, abs_tol=tol)


# ------------------------------------------------------------------ advantage math
def test_group_advantages():
    a = RA.group_advantages([1.0, 0.0, 0.0, 1.0], eps=0.0)
    assert [round(x, 12) for x in a] == [1.0, -1.0, -1.0, 1.0]            # mean .5, population std .5
    a = RA.group_advantages([3.0, 1.0], eps=1e-6)
    assert close(a[0], 1.0 / (1.0 + 1e-6)) and close(sum(a), 0.0)
    assert RA.group_advantages([0.7, 0.7, 0.7]) is None                    # zero spread -> skipped
    assert RA.group_advantages([0.7, 0.7 + 1e-12], min_std=1e-8) is None
    try:
        RA.group_advantages([1.0])
    except ValueError:
        pass
    else:
        raise AssertionError


def test_split_advantages_keep_weights():
    """Stop credit: the two parts sum to the plain GRPO advantage, and scaling the length weight changes them
    (the audit found per-part normalisation made w_dist inert)."""
    cov = [0.2, 0.5, 0.9, 0.4]
    dist = [-1.0, 0.3, -0.2, 0.8]
    outs = []
    for w in (0.1, 1.0, 5.0):
        R = [c + w * d for c, d in zip(cov, dist)]
        S = [w * d for d in dist]
        a_seq, a_stop = RA.split_group_advantages(R, S)
        plain = RA.group_advantages(R)
        assert all(abs(x + y - z) < 1e-9 for x, y, z in zip(a_seq, a_stop, plain))
        outs.append(a_stop)
    assert outs[0] != outs[1] != outs[2]
    assert abs(outs[2][0]) > abs(outs[0][0])          # a larger w_dist puts more of the advantage on the stop tokens
    assert RA.split_group_advantages([1.0, 1.0], [0.0, 0.5]) == (None, None)


def test_rloo_ppo_normalize():
    a = RA.rloo_advantages([1.0, 2.0, 6.0])
    assert [round(x, 12) for x in a] == [1.0 - 4.0, 2.0 - 3.5, 6.0 - 1.5]
    assert close(sum(RA.rloo_advantages([0.3, 0.1, 0.9, 0.4])), 0.0)       # LOO advantages sum to 0 * G/(G-1)
    assert RA.ppo_advantages([1.0, 0.5], [0.25, 0.75]) == [0.75, -0.25]
    n = RA.normalize([1.0, 2.0, 3.0], eps=0.0)
    assert close(sum(n), 0.0) and close(RA.pstd(n), 1.0)


def test_groups_and_skip_count():
    cfg = RA.algo_cfg()
    advs, skipped = RA.advantages_for_groups([[1, 0], [0.5, 0.5], [0, 0, 1]], "grpo", cfg)
    assert skipped == 1 and advs[1] is None and len(advs[2]) == 3
    advs, skipped = RA.advantages_for_groups([[1, 1], [0, 1]], "rloo", cfg)
    assert skipped == 1 and advs[1] == [-1.0, 1.0]


def test_surrogate_and_k3():
    assert RA.clipped_surrogate(1.0, 2.0, 0.2) == 2.0
    assert close(RA.clipped_surrogate(1.5, 1.0, 0.2), 1.2)                 # positive adv: capped at 1+eps
    assert close(RA.clipped_surrogate(0.5, -1.0, 0.2), -0.8)               # negative adv: capped at 1-eps
    assert close(RA.clipped_surrogate(1.5, -1.0, 0.2), -1.5)               # pessimistic side kept
    assert RA.k3(-1.3, -1.3) == 0.0                                        # KL == 0 at init
    assert RA.k3(-1.0, -2.0) > 0 and RA.k3(-2.0, -1.0) > 0


def test_episode_samples_exact_ids():
    ep = {"conversation_id": "c", "replicate": 1, "trace": [
        {"t": 1, "planner_gen": {"prompt_ids": [1, 2, 3], "gen_ids": [4, 5], "temperature": 1.0}},
        {"t": 2, "planner_gen": {"prompt_ids": list(range(40000)), "gen_ids": [9] * 600, "temperature": 1.0}}]}
    s = RA.episode_samples(ep, policy_version=3)
    assert s[0]["prompt_ids"] == [1, 2, 3] and s[0]["gen_ids"] == [4, 5]
    assert len(s[1]["prompt_ids"]) == 40000 and len(s[1]["gen_ids"]) == 600   # never truncated
    assert all(x["policy_version"] == 3 for x in s)
    try:
        RA.episode_samples({"conversation_id": "c", "trace": [{"t": 1}]})
    except ValueError:
        pass
    else:
        raise AssertionError("missing planner_gen must be refused")


def test_algo_cfg_bounds():
    for bad in ({"clip_eps": 2.0}, {"epochs": 0}, {"nope": 1}, {"forward_mode": "x"}):
        try:
            RA.algo_cfg(**bad)
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError(bad)
    assert RA.LORA["lora_dropout"] == 0.0 and RA.LORA["r"] == 16 and RA.LORA["lora_alpha"] == 32


# ------------------------------------------------------------------ dry-run full loop
IDS = ["c%02d" % i for i in range(30)]


def make_splits(d):
    s = {"folds": [{"fold": 0, "train": IDS[:14], "validation": IDS[14:15], "test": IDS[15:24],
                    "train_all": IDS[:14] + ["x_noshard"], "validation_all": IDS[14:15], "test_all": IDS[15:24],
                    "forbidden_for_training": IDS[14:24]}]}
    p = os.path.join(d, "splits.json")
    json.dump(s, open(p, "w"))
    return p


def args(splits, out, *extra, controller="dual", updates=5):
    return ["--dry-run", "--fold", "0", "--splits", splits, "--out", out, "--updates", str(updates),
            "--G", "4", "--scenarios-per-update", "3", "--val-every", "2", "--val-seeds", "0", "1",
            "--controller", controller] + (["--ablation", "test-" + controller] if controller != "llm" else []) + list(extra)


def strip(rows):
    def s(o):
        if isinstance(o, dict):
            return {k: s(v) for k, v in o.items() if k not in T.TIME_KEYS}
        if isinstance(o, list):
            return [s(v) for v in o]
        return o
    return [s(r) for r in rows]


def run_ok(argv):
    return T.main(argv)


def run_crash(argv):
    try:
        T.main(argv)
    except SystemExit as e:
        return str(e)
    raise AssertionError("expected a simulated crash")


def check_outputs(out, splits_p, n_updates):
    sp = json.load(open(splits_p))["folds"][0]
    train, forb = set(sp["train"]), set(sp["forbidden_for_training"])
    rolls = T.read_jsonl(os.path.join(out, "rollouts.jsonl"))
    keys = [(r["update"], r["slot"], r["replicate"]) for r in rolls]
    assert len(keys) == len(set(keys)), "duplicate rollout rows"
    assert len(rolls) == n_updates * 3 * 4
    for r in rolls:
        assert r["conversation_id"] in train and r["conversation_id"] not in forb and r["split"] == "train"
        assert r["policy_version"] == r["update"] - 1
    upd = T.read_jsonl(os.path.join(out, "updates.jsonl"))
    assert [r["update"] for r in upd] == list(range(1, n_updates + 1))
    for r in upd:
        assert r["policy_version_rollouts"] == r["update"] - 1
        assert r["learner_stats"].get("ratio_init_maxdev", 0.0) <= 1e-12       # exactly on-policy at start
        assert r["train_aggregate"]["split"] == "train"
    for u in range(0, n_updates + 1):
        assert os.path.exists(os.path.join(out, "ckpt", "u%05d" % u, "state.json")), u
    val = T.read_jsonl(os.path.join(out, "validation.jsonl"))
    assert val and all(v["split"] == "validation" and v["conversation_id"] in sp["validation"]
                       for v in val if v["kind"] == "episode")
    summ = [v["update"] for v in val if v["kind"] == "summary"]
    assert summ == [u for u in range(0, n_updates + 1) if u % 2 == 0], summ
    best = json.load(open(os.path.join(out, "best.json")))
    vs = {v["update"]: v["mean_reward_selection"] for v in val if v["kind"] == "summary"}
    assert best["mean_reward_selection"] == max(vs.values()) and best["update"] in vs
    # no validation id ever reached the controller history
    st = json.load(open(os.path.join(out, "ckpt", "u%05d" % n_updates, "state.json")))
    for h in st["history"]:
        assert h["split"] == "train"
    return upd, rolls, val


def test_dry_run_resume_equals_uninterrupted():
    for algo in ("grpo",):
        d = tempfile.mkdtemp()
        try:
            sp = make_splits(d)
            full, part = os.path.join(d, "full"), os.path.join(d, "part")
            run_ok(args(sp, full, "--algo", algo))
            u_full, r_full, v_full = check_outputs(full, sp, 5)
            run_ok(args(sp, part, "--algo", algo, updates=3))
            run_ok(args(sp, part, "--algo", algo, "--resume"))
            u_part, r_part, v_part = check_outputs(part, sp, 5)
            assert strip(u_full) == strip(u_part), "resume diverged from the uninterrupted run"
            assert strip(r_full) == strip(r_part)
            assert json.load(open(os.path.join(full, "best.json"))) == json.load(open(os.path.join(part, "best.json")))
            meta = T.read_jsonl(os.path.join(part, "run_meta.jsonl"))
            assert [m["kind"] for m in meta] == ["start", "resume"]
            assert all(len(v) == 64 for v in meta[0]["code_sha256"].values()) and meta[0]["splits_sha256"]
            # the policy actually moved
            assert u_full[-1]["policy_sha_after"] != T.read_jsonl(os.path.join(full, "rollouts.jsonl"))[0]["policy_sha"]
        finally:
            shutil.rmtree(d, ignore_errors=True)


def test_dry_run_crash_mid_rollouts_then_resume():
    d = tempfile.mkdtemp()
    try:
        sp = make_splits(d)
        full, part = os.path.join(d, "full"), os.path.join(d, "part")
        run_ok(args(sp, full, updates=4))
        # crash after 17 new episodes: update 1 complete (12), update 2 has 5 rollouts
        msg = run_crash(args(sp, part, "--dry-run-crash-after-episodes", "17", updates=4))
        assert "simulated crash" in msg
        assert len(T.read_jsonl(os.path.join(part, "rollouts.jsonl"))) == 17
        run_ok(args(sp, part, "--resume", updates=4))
        u_full, r_full, _ = check_outputs(full, sp, 4)
        u_part, r_part, _ = check_outputs(part, sp, 4)
        assert strip(u_full) == strip(u_part)
        assert sorted(json.dumps(x, sort_keys=True) for x in strip(r_full)) == \
            sorted(json.dumps(x, sort_keys=True) for x in strip(r_part))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_dry_run_fixed_and_llm_controllers_run():
    d = tempfile.mkdtemp()
    try:
        sp = make_splits(d)
        o1 = os.path.join(d, "fixed")
        run_ok(args(sp, o1, controller="fixed", updates=2))
        upd = T.read_jsonl(os.path.join(o1, "updates.jsonl"))
        assert upd[0]["cfg_used"] == upd[1]["cfg_used"]                     # fixed: nothing moves
        o2 = os.path.join(d, "llm")
        run_ok(args(sp, o2, controller="llm", updates=5))
        # v4 reward -> the factor controller: one decision every 5 updates, factors from the allowed set
        log = T.read_jsonl(os.path.join(o2, "llm_controller.jsonl"))
        assert len(log) == 1 and log[0]["ok"] and log[0]["changed"]
        assert all(v["factor"] == 1.25 for v in log[0]["applied"].values())
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_refuses_existing_out_without_resume_and_leaks():
    d = tempfile.mkdtemp()
    try:
        sp = make_splits(d)
        o = os.path.join(d, "o")
        run_ok(args(sp, o, updates=1))
        msg = run_crash(args(sp, o, updates=2))
        assert "--resume" in msg
        # leak: a split whose train list contains a forbidden id
        bad = json.load(open(sp))
        bad["folds"][0]["train"].append(IDS[20])
        bp = os.path.join(d, "bad.json")
        json.dump(bad, open(bp, "w"))
        try:
            T.load_split(bp, 0)
        except AssertionError:
            pass
        else:
            raise AssertionError("train/forbidden overlap not caught")
        split = T.load_split(sp, 0)
        try:
            T.assert_train_id(IDS[14], split)                                  # a validation id
        except AssertionError:
            pass
        else:
            raise AssertionError
        # judge manifest gate
        jd = os.path.join(d, "judge")
        os.makedirs(jd)
        for scen, ok_strict, ok_all in ((IDS[:5], True, True), (IDS[:5] + ["x_noshard"], False, True),
                                        (IDS[:5] + [IDS[16]], False, False)):
            json.dump({"train_scenarios": scen}, open(os.path.join(jd, "train_manifest.json"), "w"))
            for strict, ok in ((True, ok_strict), (False, ok_all)):
                try:
                    T.check_judge_manifest(jd, split, strict=strict)
                    assert ok, (scen, strict)
                except AssertionError:
                    assert not ok, (scen, strict)
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print("test_rl_advantages: %d tests passed" % n)
