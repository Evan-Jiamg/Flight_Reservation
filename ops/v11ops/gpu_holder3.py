# GPU placeholder on cfda5, INCREMENTAL + DYNAMIC PLACEMENT (user 2026-09-30: "don't insist on a specific GPU; as long
# as GPU0 + GPU1 together have room, run"; earlier: "don't let others grab the cards -- reserve resources first").
# Never touches other users' processes. Replaces gpu_holder2.py (all-at-once, one fixed split) for the v17 scripts.
#
# Components: gpt-oss vLLM (OSS_NEED 77 GiB = 0.78 x 95.6 + ~3 start slack - margin), Planner vLLM (PLANNER_NEED 16 =
# 0.15 x 95.6 + ~2 slack - margin), trainer (TRAIN_NEED 45); gpt-oss and the trainer never share a GPU. Plans, over both
# GPU orderings (a, b):
#   P1  a = gpt-oss + Planner vLLM (SERVER_NEED 91)   b = trainer (45)
#   P2  a = gpt-oss (77)                               b = Planner vLLM + trainer (16 + 45 = 61)
# Every tick, before the commit: deficit(plan) = sum_g max(0, need_g - (held_g + floor(free_g - MARGIN))); among the
# complete plans (deficit 0), or else the plans within PLAN_TIE_TOL_GIB (2 = the P1/P2 size difference) of the smallest
# deficit, the one whose shortfall sits on fewer GPUs wins, then P1, then fewer new GiB, then the lower GPU indices (see
# choose_plan). The current plan is kept unless another one is better by more than REPLAN_HYSTERESIS_GIB or is
# complete while the current one is not. Memory is grabbed in
# 1 GiB blocks, always leaving MARGIN free, up to each GPU's target; a plan change releases only the surplus above the
# new target. When the chosen plan is FULLY held: COMMIT -> role files (role_server = role_oss for old readers,
# role_oss, role_planner, role_train LAST = the commit marker); no re-planning after the commit ...
# ... unless the pipeline asks for it: start_servers6.sh writes `replan` (after a vLLM start lost the hand-over race,
# or when the committed placement no longer matches our running servers) -> UN-COMMIT, keep what is held, plan again.
# `servers_up` (written by start_servers6.sh; "oss=<g|-> planner=<g|->"): our vLLM servers already running on those
# GPUs -- they are fixed and need nothing from the holder (e.g. a fresh holder while the servers are up: only the
# training GPU is planned). Read only while not committed.
# `extra_<g>` (written by cutover_holder3.sh): GiB our OLD placeholders still hold on GPU g -- counted as available when
# choosing the plan (not for grabbing / committing), so the cut-over can hand exactly that memory over; ignored when
# older than 10 s (EXTRA_MAX_AGE_S).
# Files in $H:  role_*  status_<g> (GiB held)  target_<g> (holder-owned before the commit; after it the pipeline lowers
#               it to hand memory over and raises it to take memory back -- a raise is taken incrementally)
#               plan (one status line)  servers_up  replan  stop
# The planning part (everything above main()) is pure Python, tested without CUDA by test_gpu_holder3.py.
import itertools
import os
import sys
import time

GIB = 1 << 30


def sizes_from_env(env=None):
    env = os.environ if env is None else env
    return {"server": int(env.get("SERVER_NEED_GIB", "91")), "oss": int(env.get("OSS_NEED_GIB", "77")),
            "planner": int(env.get("PLANNER_NEED_GIB", "16")), "train": int(env.get("TRAIN_NEED_GIB", "45")),
            "margin": float(env.get("HOLD_MARGIN_GIB", "1")),
            "hysteresis": float(env.get("REPLAN_HYSTERESIS_GIB", "1")),
            "tie_tol": float(env.get("PLAN_TIE_TOL_GIB", "2"))}


def parse_servers_up(text, n):
    """'oss=1 planner=-' -> {'oss': 1, 'planner': None}; unknown / out-of-range values count as not running."""
    fixed = {"oss": None, "planner": None}
    for tok in (text or "").split():
        k, _, v = tok.partition("=")
        if k in fixed and v.isdigit() and int(v) < n:
            fixed[k] = int(v)
    return fixed


def read_extra(H, n, now=None, max_age=10.0):
    """extra_<g> files of a running cut-over -> {g: GiB}; a file older than max_age s counts as 0 (the cut-over rewrites
    them every second -- a stale one after an aborted cut-over must not create phantom-complete plans)."""
    now = time.time() if now is None else now
    out = {}
    for g in range(n):
        p = os.path.join(H, "extra_%d" % g)
        try:
            if now - os.path.getmtime(p) > max_age:
                out[g] = 0
                continue
            with open(p) as f:
                out[g] = max(0, int(float(f.read().strip())))
        except (OSError, ValueError):
            out[g] = 0
    return out


def candidate_plans(n, sizes, fixed=None):
    """All placements: gpt-oss on a, trainer on b != a, Planner vLLM on a (P1) or b (P2). A component in `fixed`
    (already running there) keeps its GPU and needs nothing."""
    fixed = fixed or {}
    fo, fp = fixed.get("oss"), fixed.get("planner")
    plans, seen = [], set()
    for a, b in itertools.permutations(range(n), 2):
        if fo is not None and a != fo:
            continue
        for p in ([fp] if fp is not None else [a, b]):
            key = (a, p, b)
            if key in seen:
                continue
            seen.add(key)
            needs = {g: 0 for g in range(n)}
            if fo is None and fp is None and p == a:
                needs[a] += sizes["server"]
            else:
                if fo is None:
                    needs[a] += sizes["oss"]
                if fp is None:
                    needs[p] += sizes["planner"]
            needs[b] += sizes["train"]
            plans.append({"name": "P1" if p == a else "P2", "roles": {"oss": a, "planner": p, "train": b},
                          "needs": needs})
    return plans


def grabbable(g, free, margin):
    """Whole 1 GiB blocks the holder could take on GPU g now (always leaving `margin` free)."""
    return max(0, int(free.get(g, 0.0) - margin + 1e-9))


def avail(g, held, free, margin):
    return held.get(g, 0) + grabbable(g, free, margin)


def shortfalls(needs, held, free, margin):
    return {g: max(0, need - avail(g, held, free, margin)) for g, need in needs.items()}


def deficit(needs, held, free, margin):
    return sum(shortfalls(needs, held, free, margin).values())


def n_short(needs, held, free, margin):
    return sum(1 for v in shortfalls(needs, held, free, margin).values() if v > 0)


def new_gib(needs, held):
    return sum(max(0, need - held.get(g, 0)) for g, need in needs.items())


def plan_key(plan, held, free, margin):
    """Order among near-equal plans: shortfall concentrated on fewer GPUs first (larger sum of squared shortfalls: e.g.
    gpt-oss complete on GPU1 and only GPU0 waiting beats both GPUs waiting for other users -- the user's example
    2026-09-30), then P1, then fewer new GiB to grab, then the lower GPU indices."""
    r, sf = plan["roles"], shortfalls(plan["needs"], held, free, margin)
    return (-sum(v * v for v in sf.values()), 0 if plan["name"] == "P1" else 1,
            new_gib(plan["needs"], held), (r["oss"], r["train"], r["planner"]))


def same_plan(p, q):
    return p is not None and q is not None and p["roles"] == q["roles"] and p["needs"] == q["needs"]


def best_plan(plans, held, free, margin, tie_tol=0.0):
    """Complete plans if any, else the plans within tie_tol GiB of the smallest deficit; then plan_key."""
    d = [deficit(p["needs"], held, free, margin) for p in plans]
    dmin = min(d)
    near = [p for p, x in zip(plans, d) if (x == 0 if dmin == 0 else x <= dmin + tie_tol + 1e-9)]
    return min(near, key=lambda p: plan_key(p, held, free, margin))


def choose_plan(plans, held, free, margin, current=None, hysteresis=0.0, tie_tol=0.0):
    """best_plan; the current plan is kept unless the best one is better by more than `hysteresis` GiB or is complete
    (deficit 0) while the current one is not."""
    if not plans:
        return None
    best = best_plan(plans, held, free, margin, tie_tol)
    cur = next((p for p in plans if same_plan(p, current)), None)
    if cur is None:
        return best
    db, dc = deficit(best["needs"], held, free, margin), deficit(cur["needs"], held, free, margin)
    if db < dc - hysteresis or (db == 0 and dc > 0):
        return best
    return cur


def grab_counts(targets, held, free, margin):
    """Per GPU: > 0 = 1 GiB blocks to grab now (never below MARGIN free), < 0 = blocks to release (down to the target)."""
    out = {}
    for g, t in targets.items():
        h = held.get(g, 0)
        if h > t:
            out[g] = t - h
        else:
            out[g] = min(t - h, grabbable(g, free, margin))
    return out


def describe(plan):
    r, nd = plan["roles"], plan["needs"]
    if r["oss"] == r["planner"]:
        s = "g%d = gpt-oss + planner (%d GiB), g%d = train (%d GiB)" % (r["oss"], nd[r["oss"]], r["train"], nd[r["train"]])
    elif r["planner"] == r["train"]:
        s = "g%d = gpt-oss (%d GiB), g%d = planner + train (%d GiB)" % (r["oss"], nd[r["oss"]], r["train"], nd[r["train"]])
    else:
        s = "gpt-oss g%d, planner g%d, train g%d (%s GiB)" % (r["oss"], r["planner"], r["train"],
                                                              ", ".join("g%d=%d" % kv for kv in sorted(nd.items())))
    return "%s: %s" % (plan["name"], s)


class HolderState:
    """Plan selection + targets, no CUDA. tick() is called once per loop with the measured state."""

    def __init__(self, n, sizes):
        self.n, self.sizes = n, sizes
        self.committed = False
        self.plan = None
        self.targets = {g: 0 for g in range(n)}

    def tick(self, held, free, fixed=None, file_targets=None, replan=False, extra=None):
        """-> {'targets': {g: GiB}, 'commit': roles or None, 'uncommit': bool, 'events': [str], 'deficit': float}
        extra: GiB per GPU still held by OUR old placeholders during a cut-over (hold3/extra_<g>); counted as available
        for choosing the plan only -- grabbing (main) and the commit use the real free / held memory."""
        ev, uncommit = [], False
        if extra:
            free = {g: free.get(g, 0.0) + max(0, extra.get(g, 0)) for g in range(self.n)}
        margin = self.sizes["margin"]
        if replan:
            if self.committed:
                ev.append("UN-COMMIT (replan requested by the pipeline); keeping %s GiB held" %
                          " ".join("g%d=%d" % (g, held.get(g, 0)) for g in range(self.n)))
                uncommit = True
            self.committed, self.plan = False, None
        if self.committed:
            for g, t in (file_targets or {}).items():
                if t is not None:
                    self.targets[g] = int(t)
            return {"targets": dict(self.targets), "commit": None, "uncommit": False, "events": ev, "deficit": 0.0}
        plans = candidate_plans(self.n, self.sizes, fixed)
        best = choose_plan(plans, held, free, margin, self.plan, self.sizes["hysteresis"], self.sizes.get("tie_tol", 0.0))
        if best is None:
            if self.plan is not None:
                ev.append("no placement possible with %d GPU(s)" % self.n)
            self.plan = None
            self.targets = {g: 0 for g in range(self.n)}
            return {"targets": dict(self.targets), "commit": None, "uncommit": uncommit, "events": ev, "deficit": float("inf")}
        d = deficit(best["needs"], held, free, margin)
        if not same_plan(best, self.plan):
            ev.append("plan -> %s; deficit %.1f GiB" % (describe(best), d))
            self.plan = best
        self.targets = {g: best["needs"].get(g, 0) for g in range(self.n)}
        commit = None
        if all(held.get(g, 0) >= t for g, t in self.targets.items()):
            self.committed = True
            commit = dict(best["roles"])
            ev.append("COMMIT %s" % describe(best))
        return {"targets": dict(self.targets), "commit": commit, "uncommit": uncommit, "events": ev, "deficit": d}


# ---------------------------------------------------------------- CUDA layer (torch imported only here)
def main(H):
    import torch

    sizes = sizes_from_env()
    tick_s = float(os.environ.get("HOLD_TICK_S", "0.25"))
    extra_age = float(os.environ.get("EXTRA_MAX_AGE_S", "10"))
    os.makedirs(H, exist_ok=True)
    n = torch.cuda.device_count()
    blocks = {g: [] for g in range(n)}
    margin = sizes["margin"]

    def path(name):
        return os.path.join(H, name)

    def rd(name, default=None):
        try:
            return open(path(name)).read().strip()
        except OSError:
            return default

    def wr(name, value):
        tmp = path(name + ".tmp")
        with open(tmp, "w") as f:
            f.write(str(value))
        os.replace(tmp, path(name))

    def rm(name):
        try:
            os.remove(path(name))
        except OSError:
            pass

    def log(msg):
        print("%s %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

    def free_gib(g):
        return torch.cuda.mem_get_info(g)[0] / GIB

    def grab(g, k):
        got = 0
        for _ in range(k):
            try:
                blocks[g].append(torch.empty(GIB, dtype=torch.uint8, device="cuda:%d" % g))
            except RuntimeError:            # somebody took it meanwhile: keep what we have, try again next tick
                with torch.cuda.device(g):
                    torch.cuda.empty_cache()
                break
            got += 1
        return got

    def shrink(g, to):
        if len(blocks[g]) > to:
            del blocks[g][max(0, to):]
            torch.cuda.synchronize(g)
            with torch.cuda.device(g):
                torch.cuda.empty_cache()

    def file_target_name(name):
        t = rd(name)
        try:
            return int(float(t)) if t is not None else None
        except ValueError:
            return None

    def file_target(g):
        return file_target_name("target_%d" % g)

    stale_files = ["role_train", "role_oss", "role_planner", "role_server", "replan", "servers_up", "stop"]
    for stale in stale_files + ["extra_%d" % g for g in range(n)]:
        rm(stale)                          # a fresh holder never inherits another instance's placement / requests
    st = HolderState(n, sizes)
    log("holder up: %d GPUs, incremental, server %d / oss %d / planner %d / train %d GiB, margin %.1f, tick %.2f s"
        % (n, sizes["server"], sizes["oss"], sizes["planner"], sizes["train"], margin, tick_s))
    last_wait_log = 0.0
    while not os.path.exists(path("stop")):
        held = {g: len(blocks[g]) for g in range(n)}
        free = {g: free_gib(g) for g in range(n)}
        replan = os.path.exists(path("replan"))
        was_committed = st.committed
        fixed = None if st.committed and not replan else parse_servers_up(rd("servers_up"), n)
        ft = {g: file_target(g) for g in range(n)} if st.committed and not replan else None
        extra = read_extra(H, n, max_age=extra_age) if not st.committed or replan else None
        res = st.tick(held, free, fixed, ft, replan, extra)
        if replan:
            for r in ("role_train", "role_oss", "role_planner", "role_server"):
                rm(r)
            rm("replan")
        for e in res["events"]:
            log(e)
        for g in range(n):
            t = res["targets"][g]
            k = grab_counts({g: t}, held, free, margin)[g]
            if k < 0:
                shrink(g, t)
                log(("g%d handed over, now holds %d GiB" if was_committed and not replan
                     else "g%d released surplus, now holds %d GiB") % (g, len(blocks[g])))
            elif k > 0:
                got = grab(g, k)
                if got and was_committed and not replan:
                    log("g%d taken back, holds %d/%d GiB" % (g, len(blocks[g]), t))
                elif got:
                    log("g%d grabbed %d GiB, holds %d/%d GiB" % (g, got, len(blocks[g]), t))
        if not st.committed or res["commit"]:
            for g in range(n):
                wr("target_%d" % g, res["targets"][g])            # holder-owned until the commit
        if res["commit"]:
            r = res["commit"]
            wr("role_server", r["oss"]); wr("role_oss", r["oss"]); wr("role_planner", r["planner"])
            wr("role_train", r["train"])                          # LAST: the commit marker readers wait for
        held = {g: len(blocks[g]) for g in range(n)}
        plan_line = ("%s %s; held %s; targets %s; free %s" % (
            "COMMITTED" if st.committed else "planning",
            describe(st.plan) if st.plan else "-",
            " ".join("g%d=%d" % (g, held[g]) for g in range(n)),
            " ".join("g%d=%d" % (g, res["targets"][g]) for g in range(n)),
            " ".join("g%d=%.1f" % (g, free[g]) for g in range(n))))
        if not st.committed:
            plan_line += "; deficit %.1f GiB" % res["deficit"]
            if time.time() - last_wait_log >= 600:
                log("waiting: " + plan_line)
                last_wait_log = time.time()
        wr("plan", plan_line)
        for g in range(n):
            wr("status_%d" % g, held[g])
        time.sleep(tick_s)
    for g in range(n):
        shrink(g, 0)
        wr("status_%d" % g, 0)
    log("holder stopped, all released")


if __name__ == "__main__":
    main(sys.argv[1])
