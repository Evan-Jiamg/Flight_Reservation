p = "task1_v4.py"
s = open(p, encoding="utf-8").read()


def rep_(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:80])
    s = s.replace(old, new)


rep_('''A trained --planner-adapter must carry a manifest whose training scenarios avoid the sessions scored here.
"""''',
'''A trained --planner-adapter must carry a manifest whose training scenarios avoid the sessions scored here.
Defaults are the pend spec (Implicit Profile on, few-shot auto = loo for all / fold for fold runs, Borda
selector); any other setting needs --ablation <name>. END flags follow M2 (task2_env.task1_generate).
"""''')
rep_('''    ap.add_argument("--implicit-profile", type=int, choices=(0, 1), default=0)
    ap.add_argument("--selector", choices=("length", "borda"), default="length",
                    help="borda: length + style rank sum (style_select.py); length: the E1.6 rule")
    ap.add_argument("--fewshot", choices=("off", "loo", "fold"), default="off",
                    help="loo: all finished sessions (leave-one-out: never the same conversation, goal or persona); "
                         "fold: splits[fold].train_all only")''',
'''    ap.add_argument("--implicit-profile", type=int, choices=(0, 1), default=1)
    ap.add_argument("--selector", choices=("length", "borda"), default="borda",
                    help="borda: length + style rank sum (style_select.py); length: the E1.6 rule")
    ap.add_argument("--fewshot", choices=("auto", "off", "loo", "fold"), default="auto",
                    help="auto (spec): loo for --sessions all, fold otherwise; loo: all finished sessions "
                         "(leave-one-out: never the same conversation, goal or persona); fold: splits[fold].train_all only")
    ap.add_argument("--ablation", default=None, help="name of a declared ablation; required for any non-spec setting")''')
rep_('''    a = ap.parse_args(argv)
    if a.sessions == "fold-test" and not a.final:''',
'''    a = ap.parse_args(argv)
    if a.fewshot == "auto":
        a.fewshot = "loo" if a.sessions == "all" else "fold"
    spec_fs = "loo" if a.sessions == "all" else "fold"
    off = {k: v for k, v in (("implicit_profile", a.implicit_profile), ("selector", a.selector), ("fewshot", a.fewshot))
           if v != {"implicit_profile": 1, "selector": "borda", "fewshot": spec_fs}[k]}
    if off and not a.ablation:
        raise SystemExit("settings %r differ from the pend spec; name the ablation with --ablation" % off)
    if a.sessions == "fold-test" and not a.final:''')
rep_('''        if a.planner_adapter:
            man = None
            for name in ("rl_manifest.json", "train_manifest.json", "run_manifest.json"):
                p = os.path.join(a.planner_adapter, name)
                if os.path.exists(p):
                    man = json.load(open(p, encoding="utf-8"))''',
'''        if a.planner_adapter:
            man = None
            # train_planner_rl writes ckpt/uNNNNN/rl_manifest.json next to ckpt/uNNNNN/adapter
            for d in (a.planner_adapter, os.path.dirname(os.path.abspath(a.planner_adapter.rstrip("/\\\\")))):
                for name in ("rl_manifest.json", "train_manifest.json", "run_manifest.json"):
                    p = os.path.join(d, name)
                    if man is None and os.path.exists(p):
                        man = json.load(open(p, encoding="utf-8"))
            if man is not None and "fold" in man and int(man["fold"]) != a.fold:
                raise SystemExit("LEAK GATE: adapter trained on fold %s, scored on fold %s" % (man["fold"], a.fold))''')
rep_('''            used = set(man.get("train_scenarios", [])) | set(man.get("train_conversations", []))''',
'''            used = set(man.get("train_scenarios", [])) | set(man.get("train_conversations", [])) | set(man.get("fewshot_pool", []))''')
rep_('''    lock = threading.Lock()
    meta = {"describe": env.describe(), "sessions": a.sessions, "fold": a.fold, "n_sessions": len(cids),
            "code_sha256": {n: sha_file(os.path.join(HERE, n)) for n in
                            ("task1_v4.py", "task2_env.py", "planner_prompt_v3.py", "fit_prompts.py", "ditto_e16.py",
                             "implicit_profile.py", "batching.py", "style_select.py")},
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)''',
'''    lock = threading.Lock()
    import implicit_profile as IP
    import style_select as SS
    settings = {"implicit_profile": a.implicit_profile, "selector": a.selector, "fewshot": a.fewshot,
                "ablation": a.ablation, "planner_adapter": a.planner_adapter or None, "planner_path": a.planner_path,
                "fewshot_k": 3, "copy_ngram": IP.COPY_NGRAM,
                "fewshot_backoff": "same style+proficiency; fewer than k -> same interaction style",
                "planner_temperature": 0.0, "end_mapping": "M2", "simcse": getattr(SS, "SIMCSE", None)}
    code = {n: sha_file(os.path.join(HERE, n)) for n in
            ("task1_v4.py", "task2_env.py", "planner_prompt_v3.py", "fit_prompts.py", "ditto_e16.py",
             "implicit_profile.py", "batching.py", "style_select.py")}
    mp = a.out + ".meta.json"
    if os.path.exists(mp) and (done or k1_done):
        old = json.load(open(mp, encoding="utf-8"))
        if old.get("settings") != settings or old.get("code_sha256") != code:
            raise SystemExit("resume refused: %s was written with other settings or code; use a new --out" % a.out)
    meta = {"describe": env.describe(), "sessions": a.sessions, "fold": a.fold, "n_sessions": len(cids),
            "settings": settings, "code_sha256": code, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(mp, "w"), indent=1)''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
