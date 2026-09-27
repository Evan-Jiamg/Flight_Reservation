"""Bench pin check moved to where r0_client is imported: the loaded module file must lie in the pinned copy."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''    if not os.path.isfile(os.path.join(BENCH_TOOLS, "r0_client.py")):
        raise RuntimeError("pinned benchmark tools missing at %s (never falling back to the live tree)" % BENCH_TOOLS)
    live = os.path.join(BENCH, "tools")
    if live in sys.path:
        raise RuntimeError("the live benchmark tools are on sys.path; only the pinned copy may be used")
    for p in (HERE, tree, os.path.join(tree, "scripts"), BENCH_TOOLS):''', '''    for p in (HERE, tree, os.path.join(tree, "scripts"), BENCH_TOOLS):''')
rep('''def make_fewshot_pool(''', '''def check_bench_module(mod):
    """The benchmark module actually imported must be the PINNED copy (a test stub without a file is accepted)."""
    f = getattr(mod, "__file__", None)
    if f and not os.path.abspath(f).startswith(os.path.abspath(BENCH_TOOLS) + os.sep):
        raise RuntimeError("benchmark module %s loaded from %s, not from the pinned copy %s" % (mod.__name__, f, BENCH_TOOLS))
    return mod


def make_fewshot_pool(''')
rep('''        from r0_client import R0Client
''', '''        import r0_client as _r0mod
        import metrics.judge as _judgemod
        check_bench_module(_r0mod)
        check_bench_module(_judgemod)
        from r0_client import R0Client
''')
rep('''                "bench_tools_sha256": {f: hashlib.sha256(open(os.path.join(BENCH_TOOLS, f), "rb").read()).hexdigest()
                                       for f in ("r0_client.py", "metrics/judge.py")},''',
    '''                "bench_tools_sha256": {f: hashlib.sha256(open(os.path.join(BENCH_TOOLS, f), "rb").read()).hexdigest()
                                       for f in ("r0_client.py", "metrics/judge.py")
                                       if os.path.isfile(os.path.join(BENCH_TOOLS, f))},''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
