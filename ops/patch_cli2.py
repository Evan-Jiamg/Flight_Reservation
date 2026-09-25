p = "rollout_v4.py"
s = open(p, encoding="utf-8").read()


def rep_(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:80])
    s = s.replace(old, new)


rep_('''    bad = train & set(split_fold["forbidden_for_training"])
    if bad:
        raise SystemExit("LEAK GATE: %s trained on validation/test scenarios %s" % (what, sorted(bad)[:3]))''',
'''    bad = train & set(split_fold["forbidden_for_training"])
    if bad:
        raise SystemExit("LEAK GATE: %s trained on validation/test scenarios %s" % (what, sorted(bad)[:3]))
    # RL manifest: Task 1 conversations, few-shot pool (train_all) must avoid validation/test too
    other = set(m.get("train_conversations") or []) | set(m.get("fewshot_pool") or [])
    bad = other & set(split_fold["forbidden_for_training"])
    if bad:
        raise SystemExit("LEAK GATE: %s used validation/test conversations %s" % (what, sorted(bad)[:3]))''')
rep_('''    ap.add_argument("--implicit-profile", type=int, choices=(0, 1), default=0)
    ap.add_argument("--fewshot", choices=("off", "fold"), default="off",
                    help="fold: Speaker few-shot examples from splits[fold].train_all only (pend arm)")
    ap.add_argument("--selector", choices=("length", "borda"), default="length")''',
'''    ap.add_argument("--implicit-profile", type=int, choices=(0, 1), default=None, help="pend spec: 1")
    ap.add_argument("--fewshot", choices=("off", "fold"), default=None,
                    help="fold: Speaker few-shot examples from splits[fold].train_all only (pend arm; pend spec)")
    ap.add_argument("--selector", choices=("length", "borda"), default=None, help="pend spec: borda")
    ap.add_argument("--planner-temperature", type=float, default=None,
                    help="pend spec (D5): 0.7, the sampled Planner, one episode per seed 0 and 1; other arms: 0")
    ap.add_argument("--planner-top-p", type=float, default=1.0)
    ap.add_argument("--ablation", default=None, help="name of a declared ablation; required for any non-spec pend setting")''')
rep_('''    args = ap.parse_args()
    if args.split == "test" and not args.final:''',
'''    args = ap.parse_args()
    spec = {"implicit_profile": 1, "fewshot": "fold", "selector": "borda", "planner_temperature": 0.7} \\
        if args.arm == "pend" else {"implicit_profile": 0, "fewshot": "off", "selector": "length", "planner_temperature": 0.0}
    for k, v in spec.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    off = {k: getattr(args, k) for k in spec if getattr(args, k) != spec[k]}
    if off and not args.ablation:
        raise SystemExit("settings %r differ from the %s spec; name the ablation with --ablation" % (off, args.arm))
    if args.arm != "pend" and (args.implicit_profile or args.fewshot != "off" or args.selector != "length"):
        raise SystemExit("Implicit Profile / few-shot / Borda selector exist only for the pend arm")
    if args.split == "test" and not args.final:''')
rep_('''    if args.planner_adapter:
        found = None
        for name in ("train_manifest.json", "rl_manifest.json", "run_manifest.json"):
            found = check_manifest(os.path.join(args.planner_adapter, name), args.fold, sf, "planner adapter") or found''',
'''    if args.planner_adapter:
        found = None
        # train_planner_rl writes ckpt/uNNNNN/rl_manifest.json next to ckpt/uNNNNN/adapter
        for d in (args.planner_adapter, os.path.dirname(os.path.abspath(args.planner_adapter.rstrip("/\\\\")))):
            for name in ("train_manifest.json", "rl_manifest.json", "run_manifest.json"):
                found = check_manifest(os.path.join(d, name), args.fold, sf, "planner adapter") or found''')
rep_('''    done = set()
    if os.path.exists(out):
        done = {(json.loads(l)["conversation_id"], json.loads(l)["seed"]) for l in open(out)}''',
'''    done = set()
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except ValueError:
                continue                  # a torn last line after a crash: that episode is regenerated
            done.add((r["conversation_id"], r["seed"], r.get("replicate", 0)))''')
rep_('''            "planner_nf4": args.planner_nf4, "planner_dtype": args.planner_dtype,''',
'''            "planner_nf4": args.planner_nf4, "planner_dtype": args.planner_dtype,
            "settings": {"implicit_profile": args.implicit_profile, "fewshot": args.fewshot, "selector": args.selector,
                         "planner_temperature": args.planner_temperature, "planner_top_p": args.planner_top_p,
                         "ablation": args.ablation},''')
rep_('''            row = env.run_episode(cid, seed, replicate=args.replicate)''',
'''            row = env.run_episode(cid, seed, replicate=args.replicate, planner_temperature=args.planner_temperature,
                                  planner_top_p=args.planner_top_p)''')
rep_('''    todo = [j for j in jobs if j not in done]''',
'''    todo = [j for j in jobs if (j[0], j[1], args.replicate) not in done]''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
