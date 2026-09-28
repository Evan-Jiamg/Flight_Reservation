"""Pin the benchmark tools (r0_client, metrics.judge) to commit ca13b33, the version every earlier run used (the shared
benchmark was rewritten on 2026-09-27 22:57: R0Client -> SystemAgentClient, Ledger -> CoverageState, R0_* / JUDGE_BASE_URL
refused). task2_env loads the tools ONLY from the pinned copy and records it in describe()."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''BENCH = "/tmp2/hchsu/trec2026-usersim-benchmark"
''', '''BENCH = "/tmp2/hchsu/trec2026-usersim-benchmark"
# The benchmark's Python tools (r0_client: R0 client + requirement Ledger; metrics.judge) are loaded from a PINNED copy
# of benchmark commit ca13b33 (2026-09-17), the version every run up to v15 used: the shared tree was rewritten on
# 2026-09-27 (names R0Client / Ledger / R0_* / JUDGE_BASE_URL retired). Data files (req_shards_v1.json, folds) are read
# from the live tree; they did not change (sha checked in the pin's manifest). Override only with PEND_BENCH_TOOLS.
BENCH_PIN = "ca13b33"
BENCH_TOOLS = os.environ.get("PEND_BENCH_TOOLS") or "/tmp2/mzjiang_usersim/grpo_planner/bench_pin/%s/tools" % BENCH_PIN
''')
rep('''    for p in (HERE, tree, os.path.join(tree, "scripts"), os.path.join(BENCH, "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)''', '''    if not os.path.isfile(os.path.join(BENCH_TOOLS, "r0_client.py")):
        raise RuntimeError("pinned benchmark tools missing at %s (never falling back to the live tree)" % BENCH_TOOLS)
    live = os.path.join(BENCH, "tools")
    if live in sys.path:
        raise RuntimeError("the live benchmark tools are on sys.path; only the pinned copy may be used")
    for p in (HERE, tree, os.path.join(tree, "scripts"), BENCH_TOOLS):
        if p not in sys.path:
            sys.path.insert(0, p)''')
rep('''                "system_prompt_sha256": hashlib.sha256(self.system.encode()).hexdigest(),''',
    '''                "system_prompt_sha256": hashlib.sha256(self.system.encode()).hexdigest(),
                "bench_tools": BENCH_TOOLS, "bench_pin": BENCH_PIN,
                "bench_tools_sha256": {f: hashlib.sha256(open(os.path.join(BENCH_TOOLS, f), "rb").read()).hexdigest()
                                       for f in ("r0_client.py", "metrics/judge.py")},''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
