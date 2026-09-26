#!/usr/bin/env python3
"""Independent content-predictability check for human stop timing (Addendum D).

1) extract: frozen base Qwen2.5-7B (NO adapter, never trained on stop labels), last hidden state
   at the final prompt position for every PRISM train/validation stop position (same encoder).
2) probe: logistic regression on [turn one-hot] vs [turn one-hot + PCA(hidden)], fit on train,
   evaluated on validation: full AUC, and within-same-turn AUC of the content part
   (logit minus the turn fixed effect), with session bootstrap.
If the within-turn AUC of the content part stays ~0.5, stop timing is not linearly decodable
from content beyond position, independent of any SFT choice.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def extract(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from stop_prompt import encode_stop_prompt
    tok = AutoTokenizer.from_pretrained(args.base_model)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quant,
                                                 device_map={"": args.gpu}, low_cpu_mem_usage=True).eval()
    for side in ("train", "validation"):
        rows = [json.loads(l) for l in open(os.path.join(args.data, side + ".jsonl"))]
        H = np.zeros((len(rows), model.config.hidden_size), dtype=np.float16)
        with torch.no_grad():
            for i, r in enumerate(rows):
                ids, _ = encode_stop_prompt(tok, r["user"], 1536)
                out = model(input_ids=torch.tensor([ids], device="cuda:%d" % args.gpu),
                            use_cache=False, output_hidden_states=True)
                H[i] = out.hidden_states[-1][0, -1].float().cpu().numpy()
                if (i + 1) % 250 == 0:
                    print(side, i + 1, flush=True)
        meta = [{"record_id": r["record_id"], "turn_index": int(r["turn_index"]),
                 "y": int(r["should_stop"])} for r in rows]
        np.savez_compressed(os.path.join(args.out, side + ".npz"), H=H, meta=np.array(json.dumps(meta)))


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def within_turn(score, y, turn):
    num = den = 0.0
    for t in np.unique(turn):
        m = turn == t
        p, n = score[m & (y == 1)], score[m & (y == 0)]
        if len(p) and len(n):
            num += auc(p, n) * len(p) * len(n)
            den += len(p) * len(n)
    return num / den if den else float("nan"), int(den)


class PCA:
    """PCA via SVD (torch; sklearn is not installed in the lab envs)."""
    def __init__(self, n_components, random_state=0):
        self.k = n_components

    def fit(self, X):
        import torch
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.mean = X.mean(0)
        _, _, V = torch.linalg.svd(X - self.mean, full_matrices=False)
        self.V = V[: self.k].T
        return self

    def transform(self, X):
        import torch
        return ((torch.from_numpy(np.asarray(X, dtype=np.float32)) - self.mean) @ self.V).numpy()


class LogisticRegression:
    """L2 logistic regression (sklearn convention: penalty 1/(2C)·|w|², intercept unpenalized)."""
    def __init__(self, C=1.0, max_iter=5000):
        self.C, self.max_iter = C, max_iter

    def fit(self, X, y):
        import torch
        X = torch.from_numpy(np.asarray(X, dtype=np.float64))
        y = torch.from_numpy(np.asarray(y, dtype=np.float64))
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.zeros((), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=self.max_iter, tolerance_grad=1e-9,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y, reduction="sum") \
                + (w ** 2).sum() / (2 * self.C)
            loss.backward()
            return loss
        opt.step(closure)
        self.coef_ = w.detach().numpy()[None, :]
        self.intercept_ = b.detach().numpy()
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_[0] + float(self.intercept_)


def probe(args):
    d = {}
    for side in ("train", "validation"):
        z = np.load(os.path.join(args.out, side + ".npz"))
        meta = json.loads(str(z["meta"]))
        d[side] = (z["H"].astype(np.float32), np.array([m["y"] for m in meta]),
                   np.minimum(np.array([m["turn_index"] for m in meta]), 12), [m["record_id"] for m in meta])
    (Htr, ytr, ttr, _), (Hva, yva, tva, sva) = d["train"], d["validation"]
    oh = lambda t: np.eye(13)[t]
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-3
    res = {}
    base = LogisticRegression(C=1e6, max_iter=5000).fit(oh(ttr), ytr)
    s_turn = base.decision_function(oh(tva))
    res["turn_only"] = {"auc": auc(s_turn[yva == 1], s_turn[yva == 0])}
    for k in (16, 64, 256, "all"):
        if k == "all":   # no reduction: PCA can discard a low-variance signal direction
            Ztr, Zva = (Htr - mu) / sd, (Hva - mu) / sd
        else:
            pca = PCA(n_components=k, random_state=0).fit((Htr - mu) / sd)
            Ztr, Zva = pca.transform((Htr - mu) / sd), pca.transform((Hva - mu) / sd)
        for C in (0.01, 0.1, 1.0):
            clf = LogisticRegression(C=C, max_iter=5000).fit(np.hstack([oh(ttr), Ztr]), ytr)
            w = clf.coef_[0]
            content = Zva @ w[13:]
            full = clf.decision_function(np.hstack([oh(tva), Zva]))
            wt, pairs = within_turn(content, yva, tva)
            # session bootstrap of the within-turn AUC
            sess = defaultdict(list)
            for i, s in enumerate(sva):
                sess[s].append(i)
            ids = list(sess)
            rng = random.Random(20260924)
            draws = []
            for _ in range(500):
                idx = np.array([i for s in (rng.choice(ids) for _ in ids) for i in sess[s]])
                v, _ = within_turn(content[idx], yva[idx], tva[idx])
                if not np.isnan(v):
                    draws.append(v)
            draws.sort()
            res["pca%s_C%g" % (k, C)] = {"auc_full": auc(full[yva == 1], full[yva == 0]),
                                         "within_turn_auc_content": wt, "pairs": pairs,
                                         "within_turn_95": [draws[int(.025 * len(draws))],
                                                            draws[int(.975 * len(draws)) - 1]]}
            print(json.dumps({"cfg": "pca%s_C%g" % (k, C), **res["pca%s_C%g" % (k, C)]}), flush=True)
    json.dump(res, open(os.path.join(args.out, "probe_results.json"), "w"), indent=1)
    print(json.dumps(res["turn_only"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("extract", "probe", "both"))
    ap.add_argument("--base-model")
    ap.add_argument("--data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.mode in ("extract", "both"):
        extract(args)
    if args.mode in ("probe", "both"):
        probe(args)


if __name__ == "__main__":
    main()
