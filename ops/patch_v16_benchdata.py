"""Review T point 2 (+ 1, 4): the live benchmark data files are sha-checked at start; provenance comment updated with the
reflog evidence; bench_pin records an override honestly."""
import sys

root = sys.argv[1]


def patch(fn, pairs):
    p = root + "/" + fn
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (fn, s.count(old), old[:80])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


patch("sep-sim/task2_env.py", [
    ('''# The benchmark's Python tools (r0_client: R0 client + requirement Ledger; metrics.judge) are loaded from a PINNED copy
# of benchmark commit ca13b33 (2026-09-17), the version every run up to v15 used: the shared tree was rewritten on
# 2026-09-27 (names R0Client / Ledger / R0_* / JUDGE_BASE_URL retired). Data files (req_shards_v1.json, folds) are read
# from the live tree; they did not change (sha checked in the pin's manifest). Override only with PEND_BENCH_TOOLS.
BENCH_PIN = "ca13b33"
BENCH_TOOLS = os.environ.get("PEND_BENCH_TOOLS") or "/tmp2/mzjiang_usersim/grpo_planner/bench_pin/%s/tools" % BENCH_PIN''',
     '''# The benchmark's Python tools (r0_client: R0 client + requirement Ledger; metrics.judge) are loaded from a PINNED copy
# of benchmark commit ca13b33 (2026-09-17). Every run up to v15 used the shared tree at 391f05b (its reflog: HEAD stayed
# there from 2026-09-14 until the pull of 2026-09-27 22:57), whose tools/ is byte-identical to ca13b33's (git diff: no
# file). The pull renamed R0Client / Ledger and retired the R0_* / JUDGE_BASE_URL variables. Override only with
# PEND_BENCH_TOOLS. The data files are read live but must keep the sha they had at 391f05b (check_bench_data).
BENCH_PIN = "ca13b33"
BENCH_TOOLS = os.environ.get("PEND_BENCH_TOOLS") or "/tmp2/mzjiang_usersim/grpo_planner/bench_pin/%s/tools" % BENCH_PIN
BENCH_DATA_SHA256 = {
    "data/req_shards_v1.json": "d42adf63cc17a5130c9a7eaa6f0936621f455bc919a154ae7ee88dd7920d7828",
    "domains/main_dataset_search/folds3_goal_persona_v1.json": "b610ba6b3b680e7d28b371f8207383dc4a8cafec0770089cff9533a87bfdec1f"}


def check_bench_data():
    """The live benchmark data files must be the ones every earlier run read (sha at 391f05b); skipped only where the
    benchmark tree does not exist at all (local tests)."""
    if not os.path.isdir(BENCH):
        return None
    for f, want in BENCH_DATA_SHA256.items():
        got = hashlib.sha256(open(os.path.join(BENCH, f), "rb").read()).hexdigest()
        if got != want:
            raise RuntimeError("benchmark data %s changed (sha %s, expected %s): refusing to run on other data" % (f, got[:12], want[:12]))
    return dict(BENCH_DATA_SHA256)'''),
    ('''        self.reqs = json.load(open(os.path.join(BENCH, "data/req_shards_v1.json")))''',
     '''        self.bench_data = check_bench_data()
        self.reqs = json.load(open(os.path.join(BENCH, "data/req_shards_v1.json")))'''),
    ('''                "bench_tools": BENCH_TOOLS, "bench_pin": BENCH_PIN,''',
     '''                "bench_tools": BENCH_TOOLS, "bench_pin": BENCH_PIN if not os.environ.get("PEND_BENCH_TOOLS") else "override",
                "bench_data_sha256": getattr(self, "bench_data", None),'''),
])
patch("sep-sim/step0_coverage.py", [
    ('''    TE.check_bench_module(_r0mod)                     # the pinned benchmark copy (as Task 2)''',
     '''    TE.check_bench_module(_r0mod)                     # the pinned benchmark copy (as Task 2)
    TE.check_bench_data()                             # the requirement shards every earlier run read'''),
])
print("patched")
