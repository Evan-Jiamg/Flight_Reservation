"""Round-4 audit fixes (2026-09-26), batch 4: evaluation CLIs. Run from sep-sim/."""


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == s.count(old) and s.count(old) >= 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


def patch1(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


CHECK_OLD = '''            diff = {k: (m.get(k), v) for k, v in settings.items() if k in m and m.get(k) != v}
            if diff:
                raise SystemExit("adapter trained with other settings than this evaluation: %r" % diff)
            return m
    return None'''
CHECK_NEW = '''            norm = lambda k, x: os.path.normpath(str(x)) if (k == "planner_path" and x) else x
            diff = {k: (m.get(k), v) for k, v in settings.items() if k in m and norm(k, m.get(k)) != norm(k, v)}
            if diff:
                raise SystemExit("adapter trained with other settings than this evaluation: %r" % diff)
            return m
    raise SystemExit("LEAK GATE: RL adapter without rl_manifest.json (next to it or in its directory)")'''

for p in ("task1_v4.py", "rollout_v4.py"):
    patch1(p, [(CHECK_OLD, CHECK_NEW)])

patch1("task1_v4.py", [
    ('''    if off and not a.ablation:
        raise SystemExit("settings %r differ from the pend spec; name the ablation with --ablation" % off)''',
     '''    if "Qwen3-4B-Instruct-2507" not in a.planner_path:
        off["planner_path"] = a.planner_path            # the spec's Planner
    if off and not a.ablation:
        raise SystemExit("settings %r differ from the pend spec; name the ablation with --ablation" % off)'''),
    ('''    env = Task2Env("pend", a.gpu, planner, judge=None, batch=bool(a.batch), max_batch=a.max_batch,
                   implicit_profile=bool(a.implicit_profile), selector=a.selector)''',
     '''    env = Task2Env("pend", a.gpu, planner, judge=None, batch=bool(a.batch), max_batch=a.max_batch,
                   implicit_profile=bool(a.implicit_profile), selector=a.selector, task1_only=True)'''),
    ('''            check_rl_settings(a.planner_adapter, {"arm": "pend", "implicit_profile": a.implicit_profile,
                                                  "selector": a.selector})''',
     '''            check_rl_settings(a.planner_adapter, {"arm": "pend", "implicit_profile": a.implicit_profile,
                                                  "selector": a.selector, "fewshot": a.fewshot,
                                                  "planner_path": a.planner_path})'''),
    ('''                "planner_temperature": 0.0, "end_mapping": "M2", "simcse": getattr(SS, "SIMCSE", None)}''',
     '''                "planner_temperature": 0.0, "end_mapping": "M2", "simcse": getattr(SS, "SIMCSE", None),
                "sessions": a.sessions, "fold": a.fold, "limit": a.limit,
                "splits_sha256": sha_file(a.splits) if a.sessions != "all" else None}'''),
])

patch1("rollout_v4.py", [
    ('''    off = {k: getattr(args, k) for k in spec if getattr(args, k) != spec[k]}
    if off and not args.ablation:''',
     '''    off = {k: getattr(args, k) for k in spec if getattr(args, k) != spec[k]}
    if args.arm == "pend" and "Qwen3-4B-Instruct-2507" not in args.planner_path:
        off["planner_path"] = args.planner_path          # the spec's Planner
    if off and not args.ablation:'''),
    ('''        check_rl_settings(args.planner_adapter, {"arm": args.arm, "implicit_profile": args.implicit_profile,
                                                 "fewshot": args.fewshot, "selector": args.selector})''',
     '''        check_rl_settings(args.planner_adapter, {"arm": args.arm, "implicit_profile": args.implicit_profile,
                                                 "fewshot": args.fewshot, "selector": args.selector,
                                                 "planner_path": args.planner_path})'''),
    ('''    done = set()
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except ValueError:
                continue                  # a torn last line after a crash: that episode is regenerated
            done.add((r["conversation_id"], r["seed"], r.get("replicate", 0)))''',
     '''    done = set()
    if os.path.exists(out):
        lines = [l for l in open(out, encoding="utf-8") if l.strip()]
        for i, l in enumerate(lines):
            try:
                r = json.loads(l)
            except ValueError:
                if i != len(lines) - 1:
                    raise SystemExit("%s: undecodable line %d (not the last one): the file is corrupt" % (out, i + 1))
                # a torn last line after a crash: cut it off (the next row must not merge into it); regenerated
                with open(out, "rb+") as f:
                    data = f.read()
                    cut = data.rstrip(b"\\n").rfind(b"\\n")
                    f.seek(0)
                    f.truncate(cut + 1 if cut >= 0 else 0)
                continue
            done.add((r["conversation_id"], r["seed"], r.get("replicate", 0)))'''),
])
print("ok")
