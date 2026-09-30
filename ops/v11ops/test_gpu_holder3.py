"""Tests of the pure planning part of gpu_holder3.py (no CUDA, no torch). Run: python -m pytest test_gpu_holder3.py"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_holder3 as GH  # noqa: E402

TOTAL = 97887 / 1024.0          # 95.6 GiB per RTX PRO 6000
S = GH.sizes_from_env({})


def free_with_others(others):
    return {g: TOTAL - o for g, o in others.items()}


def roles(plan):
    return plan["roles"]["oss"], plan["roles"]["planner"], plan["roles"]["train"]


def test_import_does_not_pull_torch():
    assert "torch" not in GH.__dict__
    src = open(GH.__file__).read().split("def main(")[0]
    assert "import torch" not in src


def test_sizes_from_env_defaults_and_overrides():
    assert (S["server"], S["oss"], S["planner"], S["train"], S["margin"]) == (91, 77, 16, 45, 1.0)
    s = GH.sizes_from_env({"SERVER_NEED_GIB": "92", "OSS_NEED_GIB": "77", "PLANNER_NEED_GIB": "16",
                           "TRAIN_NEED_GIB": "40", "HOLD_MARGIN_GIB": "2"})
    assert (s["server"], s["oss"], s["planner"], s["train"], s["margin"]) == (92, 77, 16, 40, 2.0)


def test_candidate_plans_cover_both_plans_and_orderings():
    plans = GH.candidate_plans(2, S)
    got = {(p["name"],) + roles(p): p["needs"] for p in plans}
    assert got == {("P1", 0, 0, 1): {0: 91, 1: 45}, ("P2", 0, 1, 1): {0: 77, 1: 61},
                   ("P1", 1, 1, 0): {0: 45, 1: 91}, ("P2", 1, 0, 0): {0: 61, 1: 77}}


@pytest.mark.parametrize("big", [0, 1])
def test_p1_when_one_gpu_has_92_free(big):
    free = {big: 92.0, 1 - big: 50.0}
    p = choose(GH.candidate_plans(2, S), {0: 0, 1: 0}, free)
    assert p["name"] == "P1" and roles(p) == (big, big, 1 - big)
    assert GH.deficit(p["needs"], {0: 0, 1: 0}, free, S["margin"]) == 0


def test_both_gpus_empty_prefers_p1_on_lower_index():
    free = {0: TOTAL - 0.5, 1: TOTAL - 0.5}
    p = GH.choose_plan(GH.candidate_plans(2, S), {0: 0, 1: 0}, free, S["margin"])
    assert p["name"] == "P1" and roles(p) == (0, 0, 1)


def choose(plans, held, free, current=None):
    return GH.choose_plan(plans, held, free, S["margin"], current, S["hysteresis"], S["tie_tol"])


@pytest.mark.parametrize("busy", [0, 1])
def test_p2_user_scenario_and_mirror(busy):
    """Free ~25.6 GiB on one GPU (other users ~70), ~77 on the other -> gpt-oss alone on the emptier GPU, planner +
    trainer on the busy one. P2 is short by 1 + 37 = 38, P1 on the same ordering by 15 + 21 = 36: within the 2 GiB tie
    tolerance (the P1/P2 size difference), and P2's shortfall is (almost) all on the busy GPU -> P2."""
    free = {busy: 25.6, 1 - busy: 77.0}
    held = {0: 0, 1: 0}
    plans = GH.candidate_plans(2, S)
    p = choose(plans, held, free)
    assert p["name"] == "P2" and roles(p) == (1 - busy, busy, busy)
    assert p["needs"] == {1 - busy: 77, busy: 61}
    assert GH.shortfalls(p["needs"], held, free, S["margin"]) == {1 - busy: 1, busy: 37}
    p1 = next(q for q in plans if q["name"] == "P1" and q["roles"]["oss"] == 1 - busy)
    assert GH.deficit(p1["needs"], held, free, S["margin"]) == 36 and GH.n_short(p1["needs"], held, free, S["margin"]) == 2
    # without the tie tolerance the smaller total deficit (P1) would win
    assert GH.choose_plan(plans, held, free, S["margin"])["name"] == "P1"


@pytest.mark.parametrize("busy", [0, 1])
def test_p2_user_scenario_other_users_in_gib(busy):
    """Same with other users' memory given in GiB (70 / 18.6 of 95.6)."""
    free = free_with_others({busy: 70.0, 1 - busy: 18.6})
    p = choose(GH.candidate_plans(2, S), {0: 0, 1: 0}, free)
    assert p["name"] == "P2" and roles(p) == (1 - busy, busy, busy)
    assert GH.shortfalls(p["needs"], {0: 0, 1: 0}, free, S["margin"]) == {1 - busy: 2, busy: 37}


def test_complete_plans_beat_near_ties():
    free = {0: 92.0, 1: 60.0}                       # P1 (gpt-oss + planner on GPU 0) complete; P2 short by 1
    p = choose(GH.candidate_plans(2, S), {0: 0, 1: 0}, free)
    assert p["name"] == "P1" and roles(p) == (0, 0, 1)


def test_incremental_grab_counts_keep_margin_and_stop_at_target():
    t = {0: 60, 1: 76}
    assert GH.grab_counts(t, {0: 0, 1: 0}, {0: 25.6, 1: 77.0}, 1.0) == {0: 24, 1: 76}
    assert GH.grab_counts(t, {0: 24, 1: 76}, {0: 1.6, 1: 1.0}, 1.0) == {0: 0, 1: 0}    # only the margin is free
    assert GH.grab_counts(t, {0: 24, 1: 76}, {0: 11.2, 1: 1.0}, 1.0) == {0: 10, 1: 0}  # 10 GiB freed by others
    assert GH.grab_counts(t, {0: 59, 1: 76}, {0: 30.0, 1: 5.0}, 1.0) == {0: 1, 1: 0}   # never beyond the target
    assert GH.grab_counts({0: 45}, {0: 60}, {0: 0.0}, 1.0) == {0: -15}                  # lowered target: hand over


class FakeGPUs:
    """Other users' memory per GPU + what the holder holds; free = total - others - held."""

    def __init__(self, others):
        self.others, self.held = dict(others), {g: 0 for g in others}

    def free(self):
        return {g: TOTAL - self.others[g] - self.held[g] for g in self.others}

    def apply(self, targets, margin=1.0):
        for g, k in GH.grab_counts(targets, self.held, self.free(), margin).items():
            self.held[g] += k


def run_until_commit(st, gpus, max_ticks=50, others_step=None, fixed=None):
    for i in range(max_ticks):
        if others_step:
            others_step(i, gpus)
        res = st.tick(dict(gpus.held), gpus.free(), fixed=fixed)
        if res["commit"]:
            return res, i
        gpus.apply(res["targets"])
    raise AssertionError("never committed")


def test_incremental_hold_then_commit_when_fully_held():
    gpus = FakeGPUs({0: 70.0, 1: 17.0})         # GPU 1 can hold gpt-oss's 77 at once, GPU 0 only 24 of 61
    st = GH.HolderState(2, S)
    held_g0 = []

    def others_free_gpu0(i, g):                 # the other user on GPU 0 frees 5 GiB per tick from tick 3 on
        if i >= 3:
            g.others[0] = max(10.0, g.others[0] - 5.0)
        held_g0.append(g.held[0])
    res, _ = run_until_commit(st, gpus, others_step=others_free_gpu0)
    assert res["commit"] == {"oss": 1, "planner": 0, "train": 0}
    assert gpus.held == {0: 61, 1: 77} and st.committed
    assert held_g0[1] == 24 and all(b >= a for a, b in zip(held_g0, held_g0[1:]))   # grabbed step by step, never lost
    assert held_g0[2] == 24 and 24 < held_g0[4] < 61


def test_no_commit_while_partially_held():
    st = GH.HolderState(2, S)
    free = {0: 25.6, 1: 77.0}
    res = st.tick({0: 24, 1: 76}, {0: 1.6, 1: 1.0})
    assert res["commit"] is None and not st.committed and res["targets"] == {0: 61, 1: 77}


def test_no_replan_after_commit_targets_follow_the_files():
    gpus = FakeGPUs({0: 0.5, 1: 0.5})
    st = GH.HolderState(2, S)
    res, _ = run_until_commit(st, gpus)
    assert res["commit"] == {"oss": 0, "planner": 0, "train": 1}
    plan = st.plan
    # after the commit the world changes completely: no re-plan, targets unchanged
    res = st.tick({0: 91, 1: 45}, {0: 0.0, 1: 90.0})
    assert st.plan is plan and res["commit"] is None and res["targets"] == {0: 91, 1: 45}
    # the pipeline hands over (lower) and takes back (raise) through target_<g>
    res = st.tick({0: 91, 1: 45}, {0: 1.0, 1: 1.0}, file_targets={0: 16, 1: 45})
    assert res["targets"] == {0: 16, 1: 45}
    assert GH.grab_counts(res["targets"], {0: 91, 1: 45}, {0: 1.0, 1: 1.0}, 1.0) == {0: -75, 1: 0}
    res = st.tick({0: 0, 1: 0}, {0: 3.0, 1: 50.0}, file_targets={0: 0, 1: 45})
    assert res["targets"] == {0: 0, 1: 45}
    assert GH.grab_counts(res["targets"], {0: 0, 1: 0}, {0: 3.0, 1: 50.0}, 1.0) == {0: 0, 1: 45}
    res = st.tick({0: 0, 1: 0}, {0: 3.0, 1: 20.0}, file_targets={0: 0, 1: 45})
    assert GH.grab_counts(res["targets"], {0: 0, 1: 0}, {0: 3.0, 1: 20.0}, 1.0) == {0: 0, 1: 19}   # raise: incremental


def test_plan_change_before_commit_releases_only_surplus():
    st = GH.HolderState(2, S)
    res = st.tick({0: 0, 1: 0}, {0: 46.0, 1: 94.0})
    assert roles(st.plan) == (1, 1, 0) and st.plan["name"] == "P1"
    # others took almost all the rest of GPU 1 while GPU 0 got free room: P2 with gpt-oss alone on GPU 1 is complete
    held, free = {0: 20, 1: 80}, {0: 45.0, 1: 0.5}
    res = st.tick(held, free)
    assert st.plan["name"] == "P2" and roles(st.plan) == (1, 0, 0)
    assert res["targets"] == {0: 61, 1: 77}
    ch = GH.grab_counts(res["targets"], held, free, 1.0)
    assert ch == {0: 41, 1: -3}                  # GPU 1 gives back only the 3 GiB above 77; GPU 0 keeps its 20 and grows


def test_hysteresis_keeps_the_current_plan_on_small_differences():
    st = GH.HolderState(2, S)
    st.tick({0: 0, 1: 0}, {0: 88.0, 1: 50.0})                 # P1 on GPU 0 short by 4, P2 short by 11
    first = st.plan
    assert first["name"] == "P1" and roles(first) == (0, 0, 1)
    st.tick({0: 0, 1: 0}, {0: 88.0, 1: 59.0})                 # P2 now short by 3 vs P1 4: within the 1 GiB hysteresis
    assert st.plan is first
    st.tick({0: 0, 1: 0}, {0: 88.0, 1: 61.0})                 # P2 short by 1 vs 4: switch
    assert st.plan["name"] == "P2" and roles(st.plan) == (0, 1, 1)


def test_complete_plan_always_replaces_an_incomplete_one():
    st = GH.HolderState(2, S)
    st.tick({0: 0, 1: 0}, {0: 25.6, 1: 77.0})                 # P2, gpt-oss on GPU 1
    cur = st.plan
    assert roles(cur) == (1, 0, 0)
    # GPU 1 stays 1 GiB short for gpt-oss (others moved in: deficit 1, within the hysteresis), GPU 0 became empty:
    # P1 and P2 with gpt-oss on GPU 0 are both complete -> switch anyway, P1 preferred
    st.tick({0: 24, 1: 75}, {0: 70.0, 1: 0.9})
    assert roles(st.plan) == (0, 0, 1) and st.plan["name"] == "P1"
    assert GH.grab_counts(st.targets, {0: 24, 1: 75}, {0: 70.0, 1: 0.9}, 1.0) == {0: 67, 1: -30}   # only the surplus


def test_replan_uncommits_keeps_holdings_and_commits_again():
    gpus = FakeGPUs({0: 0.5, 1: 0.5})
    st = GH.HolderState(2, S)
    res, _ = run_until_commit(st, gpus)
    assert res["commit"] == {"oss": 0, "planner": 0, "train": 1}
    # hand-over to gpt-oss lost the race on GPU 0: another user now sits there with 30 GiB; the holder still has 16
    gpus.held = {0: 16, 1: 45}
    gpus.others = {0: 30.0 + 0.5, 1: 0.5}
    res = st.tick(dict(gpus.held), gpus.free(), file_targets={0: 16, 1: 45})
    assert res["commit"] is None and st.committed                      # no re-plan without the pipeline's request
    res = st.tick(dict(gpus.held), gpus.free(), replan=True)
    assert res["uncommit"] and not st.committed and any("UN-COMMIT" in e for e in res["events"])
    # free: GPU 0 = 95.6 - 30.5 - 16 = 49.1 (avail 64.1), GPU 1 = 95.6 - 0.5 - 45 = 50.1 (avail 94.1)
    assert roles(st.plan) == (1, 1, 0) and st.plan["name"] == "P1"
    ch = GH.grab_counts(res["targets"], gpus.held, gpus.free(), 1.0)
    assert all(v >= 0 for v in ch.values())                             # nothing held is released
    res, _ = run_until_commit(st, gpus)
    assert res["commit"] == {"oss": 1, "planner": 1, "train": 0} and gpus.held == {0: 45, 1: 91}


def test_replan_to_p2_after_lost_race():
    """The 23:38 case: gpt-oss hand-over on GPU 1 lost ~14 GiB to another user; GPU 0 (training, 45 held) has room."""
    st = GH.HolderState(2, S)
    st.committed, st.plan = True, None
    held = {0: 45, 1: 16}
    free = {0: 30.0, 1: TOTAL - 14.0 - 16 - 0.5}               # GPU 1 avail = 16 + 64 = 80 >= 77
    res = st.tick(held, free, replan=True)
    assert st.plan["name"] == "P2" and roles(st.plan) == (1, 0, 0)
    assert res["targets"] == {0: 61, 1: 77}
    assert GH.deficit(st.plan["needs"], held, free, 1.0) == 0


def test_servers_already_up_plans_the_training_gpu_only():
    fixed = GH.parse_servers_up("oss=1 planner=0", 2)
    assert fixed == {"oss": 1, "planner": 0}
    plans = GH.candidate_plans(2, S, fixed)
    assert [(roles(p), p["needs"]) for p in plans] == [((1, 0, 0), {0: 45, 1: 0})]
    st = GH.HolderState(2, S)
    res = st.tick({0: 0, 1: 0}, {0: 60.0, 1: 5.0}, fixed=fixed)
    assert res["targets"] == {0: 45, 1: 0}
    res = st.tick({0: 45, 1: 0}, {0: 15.0, 1: 5.0}, fixed=fixed)
    assert res["commit"] == {"oss": 1, "planner": 0, "train": 0}


def test_gpt_oss_up_planner_down_places_planner_and_trainer():
    fixed = GH.parse_servers_up("oss=0 planner=-", 2)
    assert fixed == {"oss": 0, "planner": None}
    plans = GH.candidate_plans(2, S, fixed)
    assert sorted((p["name"],) + roles(p) + (p["needs"][0], p["needs"][1]) for p in plans) == [
        ("P1", 0, 0, 1, 16, 45), ("P2", 0, 1, 1, 0, 61)]
    p = GH.choose_plan(plans, {0: 0, 1: 0}, {0: 17.5, 1: 70.0}, 1.0)
    assert roles(p) == (0, 0, 1)                   # both complete: P1 preferred
    p = GH.choose_plan(plans, {0: 0, 1: 0}, {0: 3.0, 1: 70.0}, 1.0)
    assert roles(p) == (0, 1, 1)                   # no room next to gpt-oss: planner goes to the training GPU


def test_parse_servers_up_ignores_garbage():
    assert GH.parse_servers_up("oss=- planner=-", 2) == {"oss": None, "planner": None}
    assert GH.parse_servers_up("oss=7 planner=x", 2) == {"oss": None, "planner": None}
    assert GH.parse_servers_up(None, 2) == {"oss": None, "planner": None}


def test_single_gpu_has_no_plan():
    st = GH.HolderState(1, S)
    res = st.tick({0: 0}, {0: 95.0})
    assert res["commit"] is None and res["targets"] == {0: 0}



def test_cutover_extra_lets_the_holder_choose_the_plan_that_needs_old_memory():
    """Audit example: other users ~30 GiB on GPU 0 / 5 on GPU 1, our OLD placeholders 42 GiB on GPU 1. Without extra_1
    the holder sees GPU 1 as nearly full and picks P1 with gpt-oss on GPU 0, which never completes (the old 42 GiB are
    never needed there); with extra_1 = 42 it picks P2 with gpt-oss on GPU 1, the cut-over hands the old memory over
    while the holder's target on GPU 1 exceeds what it holds, and the plan completes."""
    others, old, held = {0: 30.0, 1: 5.0}, {0: 0, 1: 42}, {0: 0, 1: 0}

    def free():
        return {g: TOTAL - others[g] - old[g] - held[g] for g in (0, 1)}
    assert roles(choose(GH.candidate_plans(2, S), held, free())) == (0, 0, 1)          # blind to the old memory
    st = GH.HolderState(2, S)
    res = None
    for _ in range(200):
        res = st.tick(dict(held), free(), extra=dict(old))
        if res["commit"]:
            break
        assert roles(st.plan) == (1, 0, 0)
        for g, k in GH.grab_counts(res["targets"], held, free(), 1.0).items():
            held[g] += k                                   # grabs only REAL free memory
        for g in (0, 1):                                   # cut-over step: only while the holder wants more there
            if old[g] > 0 and res["targets"][g] > held[g]:
                old[g] -= min(2, old[g], res["targets"][g] - held[g])
    assert res["commit"] == {"oss": 1, "planner": 0, "train": 0}
    assert held == {0: 61, 1: 77} and old == {0: 0, 1: 12}   # 12 GiB of the old holder are surplus: released after


def test_extra_is_ignored_after_the_commit():
    st = GH.HolderState(2, S)
    st.tick({0: 91, 1: 45}, {0: 1.0, 1: 1.0})
    assert st.committed
    res = st.tick({0: 91, 1: 45}, {0: 1.0, 1: 1.0}, extra={0: 0, 1: 80})
    assert res["targets"] == {0: 91, 1: 45} and res["commit"] is None



def test_read_extra_ignores_stale_files(tmp_path):
    import time
    for g, v in ((0, "12"), (1, "42")):
        (tmp_path / ("extra_%d" % g)).write_text(v)
    now = time.time()
    assert GH.read_extra(str(tmp_path), 2, now=now) == {0: 12, 1: 42}
    old = now - 30
    os.utime(tmp_path / "extra_1", (old, old))                   # an aborted cut-over left it behind
    assert GH.read_extra(str(tmp_path), 2, now=now) == {0: 12, 1: 0}
    (tmp_path / "extra_0").write_text("garbage")
    assert GH.read_extra(str(tmp_path), 3, now=now) == {0: 0, 1: 0, 2: 0}
    # the stale 42 GiB must not make a phantom-complete plan: without it the old example stays P1 on GPU 0
    st = GH.HolderState(2, S)
    free = {0: TOTAL - 30.0, 1: TOTAL - 5.0 - 42}
    st.tick({0: 0, 1: 0}, free, extra=GH.read_extra(str(tmp_path), 2, now=now))
    assert roles(st.plan) == (0, 0, 1)


# ---------------------------------------------------------------- main loop with a fake torch (file protocol end to end)
class _FakeCuda:
    def __init__(self, others):
        self.others = dict(others)            # GiB used by other users
        self.used = {g: 0 for g in others}    # GiB held by the holder's live blocks

    def device_count(self):
        return len(self.others)

    def mem_get_info(self, g):
        return int((TOTAL - self.others[g] - self.used[g]) * GH.GIB), int(TOTAL * GH.GIB)

    def synchronize(self, g=None):
        pass

    def empty_cache(self):
        pass

    def device(self, g):
        import contextlib
        return contextlib.nullcontext()


class _FakeTorch:
    uint8 = "uint8"

    def __init__(self, others):
        self.cuda = _FakeCuda(others)

    def empty(self, n, dtype=None, device=None):
        g = int(device.split(":")[1])
        cuda = self.cuda
        if cuda.mem_get_info(g)[0] < n:
            raise RuntimeError("CUDA out of memory")
        cuda.used[g] += 1

        class Block:
            def __del__(self):
                cuda.used[g] -= 1
        return Block()


def _wait(pred, timeout=10.0):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    raise AssertionError("timeout")


def test_main_loop_file_protocol(tmp_path, monkeypatch):
    import threading
    fake = _FakeTorch({0: 20.0, 1: 17.0})
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setenv("HOLD_TICK_S", "0.01")
    H = str(tmp_path / "hold")

    def rd(n):
        p = os.path.join(H, n)
        return open(p).read().strip() if os.path.exists(p) else None

    def wr(n, v):
        open(os.path.join(H, n + ".tmp"), "w").write(v)
        os.replace(os.path.join(H, n + ".tmp"), os.path.join(H, n))
    os.makedirs(H)
    for stale in ("role_train", "stop", "replan"):                     # left by an earlier instance: cleared at start
        open(os.path.join(H, stale), "w").write("1")
    th = threading.Thread(target=GH.main, args=(H,), daemon=True)
    th.start()
    try:
        _wait(lambda: rd("plan") is not None)                          # first tick done (stale files cleared before)
        _wait(lambda: rd("role_train") is not None)
        # GPU 0 can hold 74, GPU 1 77 -> P2: gpt-oss alone on GPU 1, planner + trainer on GPU 0
        assert (rd("role_oss"), rd("role_planner"), rd("role_train"), rd("role_server")) == ("1", "0", "0", "1")
        assert fake.cuda.used == {0: 61, 1: 77}
        # hand-over (lower) and take-back (raise) through target_<g>
        wr("target_1", "1")
        _wait(lambda: fake.cuda.used[1] == 1 and rd("status_1") == "1")
        wr("target_0", "45")
        _wait(lambda: fake.cuda.used[0] == 45)
        wr("target_1", "3")
        _wait(lambda: fake.cuda.used[1] == 3)
        wr("target_1", "1")
        _wait(lambda: fake.cuda.used[1] == 1)
        # gpt-oss lost the race on GPU 1: another user now has 30 GiB there -> the pipeline asks for a re-plan
        fake.cuda.others[1] = 30.0
        wr("servers_up", "oss=- planner=-")
        for r in ("role_train", "role_oss", "role_planner", "role_server"):
            os.remove(os.path.join(H, r))
        wr("replan", "1")
        _wait(lambda: not os.path.exists(os.path.join(H, "replan")))
        assert rd("role_train") is None                                 # not committable yet (GPU 0 short)
        assert fake.cuda.used[0] >= 45 and fake.cuda.used[1] >= 1       # nothing held was given away
        fake.cuda.others[0] = 10.0                                      # somebody frees 10 GiB on GPU 0
        _wait(lambda: rd("role_train") is not None)
        assert (rd("role_oss"), rd("role_planner"), rd("role_train")) == ("0", "1", "1")
        assert fake.cuda.used == {0: 77, 1: 61}
    finally:
        wr("stop", "")
        th.join(timeout=10)
    assert not th.is_alive() and fake.cuda.used == {0: 0, 1: 0}
