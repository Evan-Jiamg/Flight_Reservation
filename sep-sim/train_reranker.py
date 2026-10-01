#!/usr/bin/env python3
"""SPEC v18 §6 (user 2026-10-01): the human-likeness reranker of the Selector -- train side only.

  generate  (GPU: Planner vLLM + Ditto-8B on --gpu) the candidates of the teacher-forced Task 1 run of the v17 u0 policy
            with the CURRENT selector (Borda, no reranker) on every conversation of splits[fold][--split]: train_all with
            seeds 0 and 1 (the training pairs), validation_all with seed 0 (the §6.3 gate only). Exactly the Task 1
            path of training / test (Task2Env.task1_generate: Implicit Profile, few-shot = train_all, guards, task1_only:
            no R0, no ledger). One row per (conversation, turn, seed): every candidate, its guard reason, the selected
            index, the person's real message at that turn. + <out>.provenance.json.
  fit       (CPU) pairs (human message, our candidate) of the same state (cid, t): the candidates of both seeds,
            without blanks and guard-rejected ones, de-duplicated within the state (whitespace-normalised text);
            phi(x) = [PCA_32(SimCSE(x)), 5 style features] standardised; pairwise logistic regression (Bradley-Terry, no
            intercept, L2: minimise 0.5 |w|^2 + C sum log(1 + exp(-w (phi(h) - phi(c))))) by Newton's method.
            Leave-one-conversation-out (17 folds): PCA and standardisation fitted INSIDE each fold's training part; C in
            {0.01, 0.1, 1} by the LOCO pairwise accuracy (ties: the smaller C); the final model refitted on all train
            pairs. Platt scaling of the out-of-fold scores (Brier, reliability; diagnostics only). Gate (§6.3, pre-
            registered): LOCO accuracy >= 0.60 AND validation accuracy >= 0.55 (point estimates), with conversation-
            cluster bootstrap 95% CIs (10000 resamples; validation has 4 clusters). Output reranker_v18_f<F>.json (the
            runtime fields of style_select.HumanLikenessScorer + the record). Every file the fit opens is logged (no
            benchmark instrument / data file may appear).

  python train_reranker.py generate --fold 2 --split train_all --seeds 0 1 --init-adapter RUN17/ckpt/u00000 \
      --planner-path $Q4 --gpu $GPU --out $L/rerank_cands_train_f2.jsonl
  python train_reranker.py generate --fold 2 --split validation_all --seeds 0 ... --out $L/rerank_cands_val_f2.jsonl
  python train_reranker.py fit --fold 2 --cands-train ... --cands-val ... --out $L/reranker_v18_f2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import style_select as SS  # noqa: E402

SPLITS = "/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json"
CORPUS = "/home/mzjiang/v5-latency/data.jsonl"
BENCH = "/tmp2/hchsu/trec2026-usersim-benchmark"
INIT_POLICY_SHA = "f67d643796951c6bac4aeec7ae1e826fb3f34d6139edaaf167d52e57d81e1ac6"
PCA_DIM = 32
CS = (0.01, 0.1, 1.0)
GATE = {"loco_acc": 0.60, "val_acc": 0.55}
N_BOOT = 10000
OPENED = []
CODE_FILES = ("train_reranker.py", "style_select.py", "task2_env.py", "implicit_profile.py", "planner_prompt_v3.py",
              "fit_prompts.py", "ditto_e16.py", "batching.py", "vllm_planner.py", "task1_v4.py", "task2_episode.py")


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        OPENED.append(os.path.abspath(os.fsdecode(args[0])))


def sha_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha_dir(p):
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(p)):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            h.update(os.path.relpath(fp, p).replace("\\", "/").encode())
            h.update(sha_file(fp).encode())
    return h.hexdigest()


def norm_text(t):
    return " ".join((t or "").split())


def fold_of(splits, fold):
    return {int(x["fold"]): x for x in json.load(open(splits, encoding="utf-8"))["folds"]}[int(fold)]


# ================================================================== generate (GPU)
def full_candidates(row):
    """task1_generate row -> the whole candidate list (the selected one at selected_index, the others in order)."""
    others, idx = list(row.get("samples") or []), int(row["selected_index"])
    cands = others[:idx] + [row["greedy"]] + others[idx:]
    reasons = row.get("guard_reasons")
    if not isinstance(reasons, list) or len(reasons) != len(cands):
        raise ValueError("%s t%s: guard reasons not aligned with %d candidates" % (row.get("conversation_id"),
                                                                                  row.get("turn_index"), len(cands)))
    return cands, [r or None for r in reasons]


def generate(a):
    f = fold_of(a.splits, a.fold)
    ids = sorted(f[a.split])
    forb = set(f["forbidden_for_training"])
    if a.split == "train_all":
        assert not set(ids) & forb, "train_all intersects forbidden"
    else:
        assert set(ids) <= forb and not set(ids) & set(f["train_all"]), "validation_all must be forbidden / not train"
    st = json.load(open(os.path.join(a.init_adapter, "state.json"), encoding="utf-8"))
    if st.get("policy_sha") != a.init_policy_sha:
        raise SystemExit("the init adapter %s is not u0 (policy sha %s)" % (a.init_adapter, str(st.get("policy_sha"))[:12]))
    adapter = os.path.join(a.init_adapter, "adapter")
    if os.path.exists(a.out):
        raise SystemExit("%s exists: use a new --out" % a.out)
    if a.dry_run:
        env = DryEnv(a.corpus)
        name = "p0-dry"
    else:
        import torch
        torch.cuda.set_device(int(a.gpu))
        from task2_env import Task2Env, PlannerLM, setup_environment, make_fewshot_pool
        import task1_v4
        import bench_tf_generate as B
        setup_environment("pend")
        planner = task1_v4.make_planner(PlannerLM, a.planner_path, a.gpu, None, "vllm", a.vllm_url)
        name = "p0-%s" % sha_dir(adapter)[:12]
        foreign = B.foreign_adapters(planner.remote, name)
        if foreign and not a.force_unload:
            raise SystemExit("the Planner vLLM server holds other adapter(s) %s; pass --force-unload only if unused" % foreign)
        planner.remote.use_adapter(adapter, "p0")
        planner.adapter = adapter
        # the selector of the reranker's data is the CURRENT one (Borda, no reranker), as u0 ran it
        env = Task2Env("pend", a.gpu, planner, judge=None, corpus=a.corpus, batch=True, max_batch=8,
                       implicit_profile=True, selector="borda", task1_only=True)
        env.fewshot = make_fewshot_pool(env.recs, sorted(f["train_all"]))
    prov = {"kind": "rerank_candidates", "split": a.split, "fold": a.fold, "seeds": list(a.seeds), "ids": ids,
            "init_adapter": a.init_adapter, "adapter_dir_sha256": sha_dir(adapter) if os.path.isdir(adapter) else None,
            "init_policy_sha": a.init_policy_sha, "gen_adapter": name, "splits_sha256": sha_file(a.splits),
            "corpus_sha256": sha_file(a.corpus), "selector": "borda", "env": env.describe(),
            "code_sha256": {c: sha_file(os.path.join(HERE, c)) for c in CODE_FILES if os.path.exists(os.path.join(HERE, c))},
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    rows = []
    from concurrent.futures import ThreadPoolExecutor

    def one(job):
        cid, seed = job
        users = [m["text"] for m in env.recs[cid]["chat_messages"] if str(m.get("participant_name", "")).lower() == "user"]
        trows, _ = env.task1_generate(cid, seed=seed)
        out = []
        for r in trows:
            t = int(r["turn_index"])
            cands, reasons = full_candidates(r)
            out.append({"conversation_id": cid, "t": t, "seed": seed, "n_turns": len(users), "human_text": users[t - 1],
                        "candidates": cands, "guard_reasons": reasons, "selected_index": r["selected_index"],
                        "speaker_hit_max_new": r.get("speaker_hit_max_new"),
                        "gen_adapter": (r.get("planner_fit") or {}).get("gen_adapter")})
        return out
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for res in ex.map(one, [(c, s) for c in ids for s in a.seeds]):
            rows += res
    if not a.dry_run:
        bad = sorted({r["gen_adapter"] for r in rows} - {name})
        if bad:
            raise SystemExit("candidates generated by %s, expected only %s" % (bad, name))
    rows.sort(key=lambda r: (r["conversation_id"], r["t"], r["seed"]))
    with open(a.out + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    os.replace(a.out + ".tmp", a.out)
    prov.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), n_rows=len(rows), out_sha256=sha_file(a.out))
    json.dump(prov, open(a.out + ".provenance.json", "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({"out": a.out, "n_rows": len(rows), "sha256": prov["out_sha256"][:12]}), flush=True)
    return prov


class DryEnv:
    """Pure-python stand-in (tests): task1_generate rows with 4 candidates per turn."""

    def __init__(self, corpus):
        self.recs = {}
        for l in open(corpus, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                self.recs[r["conversation_id"]] = r

    def describe(self):
        return {"dry_run": True, "selector": "borda"}

    def task1_generate(self, cid, seed=0):
        users = [m["text"] for m in self.recs[cid]["chat_messages"] if str(m.get("participant_name", "")).lower() == "user"]
        rows = []
        for t in range(1, len(users) + 1):
            rng = random.Random("%s|%d|%d" % (cid, t, seed))
            c0 = "Could you share a dataset about %s please." % cid[:4]           # slot 0: the same for both seeds
            cands = [c0] + ["Option %d for turn %d %s" % (rng.randrange(5), t, "!" if rng.random() < 0.3 else ".")
                            for _ in range(3)]
            idx = rng.randrange(4)
            reasons = ["" if rng.random() > 0.15 else "near_copy" for _ in range(4)]
            reasons[idx] = ""
            rows.append({"turn_index": t, "greedy": cands[idx], "selected_index": idx,
                         "samples": [c for j, c in enumerate(cands) if j != idx], "guard_reasons": reasons,
                         "speaker_hit_max_new": [False] * 4, "planner_fit": {"gen_adapter": "p0-dry"}})
        return rows, {}


# ================================================================== fit (CPU)
def build_pairs(rows, allowed, forbidden):
    """-> (pairs [(cid, t, human, candidate)], stats). Candidates of both seeds, blanks and guard-rejected removed,
    de-duplicated within the state; a candidate identical to the person's message is dropped (counted)."""
    states = {}
    st = {"n_rows": len(rows), "n_cands": 0, "n_blank": 0, "n_guard_rejected": 0, "n_duplicate": 0, "n_equal_human": 0}
    for r in rows:
        cid = r["conversation_id"]
        if cid not in allowed or cid in forbidden:
            raise SystemExit("candidate row of conversation %s outside the allowed split" % cid)
        s = states.setdefault((cid, int(r["t"])), {"human": r["human_text"], "cands": [], "seen": set()})
        if norm_text(s["human"]) != norm_text(r["human_text"]):
            raise SystemExit("%s t%s: two different human texts" % (cid, r["t"]))
        for c, why in zip(r["candidates"], r["guard_reasons"]):
            st["n_cands"] += 1
            if not (c or "").strip():
                st["n_blank"] += 1
                continue
            if why:
                st["n_guard_rejected"] += 1
                continue
            k = norm_text(c)
            if k in s["seen"]:
                st["n_duplicate"] += 1
                continue
            s["seen"].add(k)
            if k == norm_text(s["human"]):
                st["n_equal_human"] += 1
                continue
            s["cands"].append(c)
    pairs = [(cid, t, s["human"], c) for (cid, t), s in sorted(states.items()) for c in s["cands"]]
    st.update(n_pairs=len(pairs), n_states=sum(1 for s in states.values() if s["cands"]),
              n_conversations=len({p[0] for p in pairs}))
    return pairs, st


def pca_fit(X, k):
    import numpy as np
    mu = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    k = min(k, vt.shape[0])
    return mu, vt[:k]


class Featurizer:
    """phi(x) = [PCA(E(x)), style(x)], standardised -- fitted on the given texts (one fold's training part, or all)."""

    def __init__(self, emb_of, texts, k=PCA_DIM):
        import numpy as np
        E = np.asarray([emb_of[t] for t in texts])
        self.mu, self.comp = pca_fit(E, k)
        raw = self.raw(emb_of, texts)
        self.f_mean = raw.mean(axis=0)
        sd = raw.std(axis=0)
        self.f_std = np.where(sd > 1e-12, sd, 1.0)

    def raw(self, emb_of, texts):
        import numpy as np
        E = np.asarray([emb_of[t] for t in texts])
        P = (E - self.mu) @ self.comp.T
        S = np.asarray([SS.style_features(t) for t in texts], dtype=float)
        return np.concatenate([P, S], axis=1)

    def __call__(self, emb_of, texts):
        return (self.raw(emb_of, texts) - self.f_mean) / self.f_std


def fit_pairwise(D, C, iters=100):
    """Newton's method for min_w 0.5 |w|^2 + C sum_i log(1 + exp(-w . d_i)) (no intercept). D: n x p differences."""
    import numpy as np
    n, p = D.shape
    w = np.zeros(p)
    for _ in range(iters):
        z = D @ w
        s = 1.0 / (1.0 + np.exp(z))                     # sigma(-z)
        g = w - C * (D.T @ s)
        H = np.eye(p) + C * (D.T * (s * (1 - s))) @ D
        step = np.linalg.solve(H, g)
        w = w - step
        if float(np.abs(step).max()) < 1e-10:
            break
    return w


def acc_of(scores):
    """Pairwise accuracy of decision values (ties count 1/2)."""
    return sum(1.0 if s > 0 else (0.5 if s == 0 else 0.0) for s in scores) / len(scores) if scores else None


def cluster_ci(correct_by_conv, n_boot=N_BOOT, seed=0):
    """Conversation-cluster bootstrap 95% CI of the pairwise accuracy. correct_by_conv: {cid: [0/0.5/1 per pair]}."""
    convs = sorted(correct_by_conv)
    if not convs:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        pick = [rng.choice(convs) for _ in convs]
        xs = [x for c in pick for x in correct_by_conv[c]]
        vals.append(sum(xs) / len(xs))
    vals.sort()
    return {"lo": vals[int(0.025 * n_boot)], "hi": vals[int(0.975 * n_boot) - 1], "n_clusters": len(convs),
            "n_boot": n_boot}


def platt(scores):
    """Platt scaling of out-of-fold decision values on the symmetric pairs (d, 1), (-d, 0): P = sigmoid(a d + b).
    -> {a, b, brier, reliability}."""
    import numpy as np
    d = np.asarray(list(scores) + [-x for x in scores], dtype=float)
    y = np.asarray([1.0] * len(scores) + [0.0] * len(scores))
    a_, b_ = 1.0, 0.0
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(a_ * d + b_)))
        g = np.array([np.sum((p - y) * d), np.sum(p - y)])
        wv = p * (1 - p)
        H = np.array([[np.sum(wv * d * d), np.sum(wv * d)], [np.sum(wv * d), np.sum(wv)]]) + 1e-9 * np.eye(2)
        st = np.linalg.solve(H, g)
        a_, b_ = a_ - st[0], b_ - st[1]
        if float(np.abs(st).max()) < 1e-10:
            break
    p = 1.0 / (1.0 + np.exp(-(a_ * d + b_)))
    rel = []
    for k in range(10):
        m = (p >= k / 10) & (p < (k + 1) / 10 if k < 9 else p <= 1.0)
        if m.any():
            rel.append({"bin": k, "n": int(m.sum()), "p_mean": float(p[m].mean()), "y_mean": float(y[m].mean())})
    return {"a": float(a_), "b": float(b_), "brier": float(np.mean((p - y) ** 2)), "reliability": rel}


def fit(a, embed=None):
    import numpy as np
    f = fold_of(a.splits, a.fold)
    train_all, val_all, forb = set(f["train_all"]), set(f["validation_all"]), set(f["forbidden_for_training"])
    rows_tr = [json.loads(l) for l in open(a.cands_train, encoding="utf-8") if l.strip()]
    rows_va = [json.loads(l) for l in open(a.cands_val, encoding="utf-8") if l.strip()]
    pairs, st = build_pairs(rows_tr, train_all, forb)
    vpairs, vst = build_pairs(rows_va, val_all, set())
    if not pairs:
        raise SystemExit("no training pair")
    texts = sorted({p[2] for p in pairs} | {p[3] for p in pairs} | {p[2] for p in vpairs} | {p[3] for p in vpairs})
    if embed is None:
        scorer = SS.StyleScorer(a.embed_model)
        embed = lambda ts: scorer.embed(ts).numpy()
    E = np.asarray(embed(texts), dtype=float)
    emb_of = {t: E[i] for i, t in enumerate(texts)}
    convs = sorted({p[0] for p in pairs})
    folds, oof = [], {C: {} for C in CS}
    for held in convs:
        tr = [p for p in pairs if p[0] != held]
        te = [p for p in pairs if p[0] == held]
        tr_texts = sorted({p[2] for p in tr} | {p[3] for p in tr})
        fz = Featurizer(emb_of, tr_texts)
        Dtr = fz(emb_of, [p[2] for p in tr]) - fz(emb_of, [p[3] for p in tr])
        Dte = fz(emb_of, [p[2] for p in te]) - fz(emb_of, [p[3] for p in te])
        for C in CS:
            w = fit_pairwise(Dtr, C)
            oof[C][held] = [float(x) for x in Dte @ w]
        folds.append({"held_out": held, "train_cids": sorted({p[0] for p in tr}), "pca_fit_cids": sorted({p[0] for p in tr}),
                      "pca_fit_n_texts": len(tr_texts), "n_pairs_train": len(tr), "n_pairs_test": len(te),
                      "pca_components_sha256": hashlib.sha256(fz.comp.tobytes()).hexdigest()})
    loco = {str(C): acc_of([x for c in convs for x in oof[C][c]]) for C in CS}
    best_c = sorted(CS, key=lambda C: (-loco[str(C)], C))[0]
    all_texts = sorted({p[2] for p in pairs} | {p[3] for p in pairs})
    fz = Featurizer(emb_of, all_texts)
    D = fz(emb_of, [p[2] for p in pairs]) - fz(emb_of, [p[3] for p in pairs])
    w = fit_pairwise(D, best_c)
    vs = [float(x) for x in (fz(emb_of, [p[2] for p in vpairs]) - fz(emb_of, [p[3] for p in vpairs])) @ w] if vpairs else []
    corr = lambda s: 1.0 if s > 0 else (0.5 if s == 0 else 0.0)
    loco_by = {c: [corr(x) for x in oof[best_c][c]] for c in convs}
    val_by = {}
    for p, s in zip(vpairs, vs):
        val_by.setdefault(p[0], []).append(corr(s))
    loco_acc, val_acc = loco[str(best_c)], acc_of(vs)
    gate = {"loco_acc": loco_acc, "val_acc": val_acc, "thresholds": dict(GATE),
            "passed": bool(loco_acc is not None and val_acc is not None and loco_acc >= GATE["loco_acc"]
                           and val_acc >= GATE["val_acc"]),
            "loco_ci": cluster_ci(loco_by, a.n_boot), "val_ci": cluster_ci(val_by, a.n_boot),
            "note": "a gate, not an arm: when it fails the selector stays Borda (§6.3); the CIs are reported as they are"}
    out = {"version": "v18-reranker-1", "fold": a.fold, "embed_model": a.embed_model,
           "features": {"pca_dim": int(fz.comp.shape[0]), "style": list(SS.STYLE_FEATURES), "length": False,
                        "tfidf": False},
           "pca": {"mean": fz.mu.tolist(), "components": fz.comp.tolist()},
           "scaler": {"mean": fz.f_mean.tolist(), "std": fz.f_std.tolist()}, "w": w.tolist(), "C": best_c,
           "train_cids": convs, "pairs_train": st, "pairs_val": vst, "n_pairs": len(pairs), "n_states": st["n_states"],
           "n_conversations": len(convs),
           "data_sha256": {"cands_train": sha_file(a.cands_train), "cands_val": sha_file(a.cands_val),
                           "splits": sha_file(a.splits)},
           "loco": {"acc_by_C": loco, "chosen_C": best_c, "folds": folds,
                    "oof_scores": {c: oof[best_c][c] for c in convs}},
           "platt": platt([x for c in convs for x in oof[best_c][c]]),
           "validation": {"acc": val_acc, "n_pairs": len(vpairs), "n_conversations": len(val_by)},
           "gate": gate, "code_sha256": {c: sha_file(os.path.join(HERE, c)) for c in ("train_reranker.py", "style_select.py")},
           "opened_files": sorted(set(OPENED)), "time": time.time()}
    with open(a.out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(json.dumps({"out": a.out, "C": best_c, "loco_acc_by_C": loco, "val_acc": val_acc, "n_pairs": len(pairs),
                      "n_states": st["n_states"], "n_conversations": len(convs), "gate_passed": gate["passed"],
                      "loco_ci": gate["loco_ci"], "val_ci": gate["val_ci"]}), flush=True)
    return out


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--fold", type=int, required=True)
    g.add_argument("--split", choices=("train_all", "validation_all"), required=True)
    g.add_argument("--seeds", type=int, nargs="+", required=True)
    g.add_argument("--init-adapter", required=True)
    g.add_argument("--init-policy-sha", default=INIT_POLICY_SHA)
    g.add_argument("--planner-path", default=None)
    g.add_argument("--vllm-url", default="http://127.0.0.1:8031/v1")
    g.add_argument("--gpu", type=int, default=0)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--force-unload", action="store_true")
    g.add_argument("--splits", default=SPLITS)
    g.add_argument("--corpus", default=CORPUS)
    g.add_argument("--out", required=True)
    g.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("fit")
    f.add_argument("--fold", type=int, required=True)
    f.add_argument("--cands-train", required=True)
    f.add_argument("--cands-val", required=True)
    f.add_argument("--splits", default=SPLITS)
    f.add_argument("--embed-model", default=SS.SIMCSE)
    f.add_argument("--n-boot", type=int, default=N_BOOT)
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.abspath(a.out).startswith(BENCH):
        ap.error("never write into the benchmark tree")
    if a.cmd == "generate":
        if a.split == "train_all" and sorted(a.seeds) != [0, 1]:
            ap.error("spec §6.1: train_all with seeds 0 and 1")
        if a.split == "validation_all" and a.seeds != [0]:
            ap.error("spec §6.3: validation_all with seed 0")
        if not a.dry_run and not a.planner_path:
            ap.error("--planner-path is required")
    return a


def main(argv=None, embed=None):
    a = parse(argv)
    if a.cmd == "fit":
        sys.addaudithook(_audit)
        return fit(a, embed=embed)
    return generate(a)


if __name__ == "__main__":
    main()
