"""Evaluation CLIs: Planner generation through the same vLLM backend as training (user decision 2026-09-26)."""


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


VLLM_PLANNER = '''
def make_planner(PlannerLM, path, gpu, adapter, backend, url, **hf_kw):
    """HF: the model on this GPU (legacy / ablation). vLLM (spec): only the tokenizer here; generation on the vLLM
    server, the adapter (if any) loaded there under a name carrying its sha."""
    if backend == "hf":
        return PlannerLM(path, gpu, adapter=adapter or None, **hf_kw)
    import vllm_planner
    planner = PlannerLM(path, gpu, load_model=False)
    planner.remote = vllm_planner.VLLMPlanner(url)
    if adapter:
        planner.remote.use_adapter(adapter, "eval")
    planner.adapter = adapter or None
    return planner

'''

s = open("task1_v4.py", encoding="utf-8").read()
s = s.replace("\n\ndef main(argv=None):", "\n" + VLLM_PLANNER + "\ndef main(argv=None):", 1)
open("task1_v4.py", "w", encoding="utf-8", newline="\n").write(s)
patch("task1_v4.py", [
    ('''    ap.add_argument("--planner-adapter", default="")''',
     '''    ap.add_argument("--planner-adapter", default="")
    ap.add_argument("--planner-backend", choices=("vllm", "hf"), default="vllm",
                    help="spec: vllm (the same generation backend as training); hf needs --ablation")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8031/v1")'''),
    ('''    if "Qwen3-4B-Instruct-2507" not in a.planner_path:''',
     '''    if a.planner_backend != "vllm":
        off["planner_backend"] = a.planner_backend
    if "Qwen3-4B-Instruct-2507" not in a.planner_path:'''),
    ('''    planner = PlannerLM(a.planner_path, a.gpu, adapter=a.planner_adapter or None)''',
     '''    planner = make_planner(PlannerLM, a.planner_path, a.gpu, a.planner_adapter, a.planner_backend, a.vllm_url)'''),
    ('''                                                  "planner_path": a.planner_path})''',
     '''                                                  "planner_path": a.planner_path, "planner_backend": a.planner_backend})'''),
    ('''                "sessions": a.sessions, "fold": a.fold, "limit": a.limit,''',
     '''                "sessions": a.sessions, "fold": a.fold, "limit": a.limit, "planner_backend": a.planner_backend,'''),
])

s = open("rollout_v4.py", encoding="utf-8").read()
s = s.replace("\n\ndef main():", "\n" + VLLM_PLANNER + "\ndef main():", 1)
open("rollout_v4.py", "w", encoding="utf-8", newline="\n").write(s)
patch("rollout_v4.py", [
    ('''    ap.add_argument("--planner-adapter", default="")''',
     '''    ap.add_argument("--planner-adapter", default="")
    ap.add_argument("--planner-backend", choices=("vllm", "hf"), default=None,
                    help="pend spec: vllm (the same generation backend as training); other arms: hf")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8031/v1")'''),
    ('''    spec = {"implicit_profile": 1, "fewshot": "fold", "selector": "borda", "planner_temperature": 0.7} \\
        if args.arm == "pend" else {"implicit_profile": 0, "fewshot": "off", "selector": "length", "planner_temperature": 0.0}''',
     '''    spec = {"implicit_profile": 1, "fewshot": "fold", "selector": "borda", "planner_temperature": 0.7,
            "planner_backend": "vllm"} \\
        if args.arm == "pend" else {"implicit_profile": 0, "fewshot": "off", "selector": "length", "planner_temperature": 0.0,
                                    "planner_backend": "hf"}'''),
    ('''    planner = PlannerLM(args.planner_path, args.gpu, nf4=args.planner_nf4, dtype=args.planner_dtype,
                        adapter=args.planner_adapter or None)''',
     '''    planner = make_planner(PlannerLM, args.planner_path, args.gpu, args.planner_adapter, args.planner_backend,
                           args.vllm_url, nf4=args.planner_nf4, dtype=args.planner_dtype)'''),
    ('''                                                     "planner_path": args.planner_path})''',
     '''                                                     "planner_path": args.planner_path,
                                                     "planner_backend": args.planner_backend})'''),
    ('''                         "planner_temperature": args.planner_temperature, "planner_top_p": args.planner_top_p,''',
     '''                         "planner_temperature": args.planner_temperature, "planner_top_p": args.planner_top_p,
                         "planner_backend": args.planner_backend,'''),
])
print("ok")
