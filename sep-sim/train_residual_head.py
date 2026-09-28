#!/usr/bin/env python3
"""Stage D2: exact expected-return training of a residual stop head, with controllers.

Gate hazard h = sigmoid(z_B + w·x + b), x = standardized last hidden state (train stats),
z_B = Stage B YES log-odds, w = b = 0 at start (identical to Stage B).

Exact expectation over each logged no-gate episode (truncation property):
  P(stop at t) = h_t Π_{s<t}(1-h_s);   survive -> the episode's own outcome.
Loss (Addendum B/C of PREREG_STAGE_BC_20260924.md), inner-train only:
  w_len * (mean E[emitted] - mean K)^2 / var(K)
  - lam_cov * gap_cov - lam_comp * gap_comp      gap = mean E[x] - (mean x_nogate - delta)
  + NLL on gold-prefix rows  + beta * KL(h || h_B)  + 1e-4 * |w|^2
Controllers: stageB | fixed | dual | random | llm (see prereg). Selection on inner-val V.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch

BOUNDS = {"lr": (1e-4, 1e-1), "beta": (0.01, 1.0), "w_len": (0.1, 10.0)}
C0 = {"lr": 1e-2, "beta": 0.1, "w_len": 1.0}
DELTA = {"cov": 0.02, "comp": 0.02}
ROUNDS, STEPS, ETA = 8, 50, 0.5


def clip_config(new, old):
    out = {}
    for k, (lo, hi) in BOUNDS.items():
        v = float(new.get(k, old[k]))
        if not math.isfinite(v) or v <= 0:
            v = old[k]
        v = min(max(v, old[k] / 3, lo), old[k] * 3, hi)
        out[k] = v
    return out


def load(npz_path, outcomes_path, human_path):
    z = np.load(npz_path)
    meta = json.loads(str(z["meta"]))
    X, zb = z["hidden"].astype(np.float32), z["logit"].astype(np.float32)
    outcomes = {}
    for l in open(outcomes_path):
        e = json.loads(l)
        outcomes[(e["conversation_id"], e["seed"], e["replicate"], e["speaker"])] = e
    human = json.load(open(human_path))
    sides = {}
    for side in ("inner_train", "inner_validation"):
        gold = [i for i, m in enumerate(meta) if m["kind"] == "gold" and m["side"] == side]
        eps = {}
        for i, m in enumerate(meta):
            if m["kind"] == "episode" and m["side"] == side:
                eps.setdefault((m["conversation_id"], m["seed"], m["replicate"], m["speaker"]), []).append((m["t"], i))
        episodes = []
        for key, steps in sorted(eps.items()):
            steps.sort()
            o = outcomes[key]
            tr = {s["t"]: s for s in o["trace"]}
            idx = [i for _, i in steps]
            before = []   # outcome if the user stops before writing at step t
            cov, comp = 0.0, 0.0
            for t, _ in steps:
                before.append((t - 1, cov, comp))
                if tr[t]["decision"] == "continue":
                    cov, comp = tr[t]["coverage_after"], float(tr[t]["complete_after"])
            episodes.append({"key": key, "idx": idx, "before": np.array(before, dtype=np.float32),
                             "final": np.array([o["emitted_user_turns"], o["coverage"], float(o["complete"])],
                                               dtype=np.float32),
                             "K": float(human[key[0]])})
        sides[side] = {"gold": gold, "episodes": episodes}
    tgt = np.array([m.get("target_stop", False) for m in meta], dtype=np.float32)
    return X, zb, tgt, sides, meta


class Head(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(d))
        self.b = torch.nn.Parameter(torch.zeros(()))

    def forward(self, X, zb):
        return zb + X @ self.w + self.b


def expectations(h, side):
    em, cov, comp, abs_err = [], [], [], []
    for e in side["episodes"]:
        he = h[e["idx"]]
        surv = torch.cumprod(torch.cat([torch.ones(1), 1 - he]), 0)
        p_stop = he * surv[:-1]
        p_final = surv[-1]
        B = torch.from_numpy(e["before"])
        F = torch.from_numpy(e["final"])
        v = p_stop @ B + p_final * F
        em.append(v[0]); cov.append(v[1]); comp.append(v[2])
        abs_err.append(p_stop @ torch.abs(B[:, 0] - e["K"]) + p_final * abs(F[0].item() - e["K"]))
    return torch.stack(em), torch.stack(cov), torch.stack(comp), torch.stack(abs_err)


def objective(model, X, zb, tgt, side, cfg, lam, hB, varK, parts=False):
    z = model(X, zb)
    h = torch.sigmoid(z)
    em, cov, comp, abs_err = expectations(h, side)
    K = torch.tensor([e["K"] for e in side["episodes"]])
    nog = torch.tensor(np.array([e["final"] for e in side["episodes"]]))
    len_term = (em.mean() - K.mean()) ** 2 / varK
    gap_cov = cov.mean() - (nog[:, 1].mean() - DELTA["cov"])
    gap_comp = comp.mean() - (nog[:, 2].mean() - DELTA["comp"])
    g = side["gold"]
    nll = torch.nn.functional.binary_cross_entropy_with_logits(z[g], tgt[g]) if g else torch.zeros(())
    allidx = sorted({i for e in side["episodes"] for i in e["idx"]} | set(g))
    p, q = h[allidx].clamp(1e-6, 1 - 1e-6), hB[allidx].clamp(1e-6, 1 - 1e-6)
    kl = (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).mean()
    loss = (cfg["w_len"] * len_term - lam["cov"] * gap_cov - lam["comp"] * gap_comp
            + nll + cfg["beta"] * kl + 1e-4 * (model.w ** 2).sum())
    if not parts:
        return loss, gap_cov.detach(), gap_comp.detach()
    return {"loss": loss.item(), "len_term": len_term.item(), "mean_emitted": em.mean().item(),
            "mean_K": K.mean().item(), "mean_abs_err": abs_err.mean().item(),
            "coverage": cov.mean().item(), "coverage_nogate": nog[:, 1].mean().item(),
            "complete": comp.mean().item(), "complete_nogate": nog[:, 2].mean().item(),
            "gap_cov": gap_cov.item(), "gap_comp": gap_comp.item(), "gold_nll": nll.item(),
            "kl": kl.item(), "n_episodes": len(side["episodes"]), "n_gold": len(g),
            "w_norm": model.w.norm().item(), "b": model.b.item()}


def val_score(p):
    return p["len_term"] + 10 * max(0.0, -p["gap_cov"]) + 10 * max(0.0, -p["gap_comp"]) + p["gold_nll"]


def llm_propose(history, cfg, log_path):
    """gpt-5-mini proposes the next config from inner-train aggregates only."""
    import urllib.request
    payload_view = {"bounds": BOUNDS, "max_change_factor_per_round": 3, "current_config": cfg,
                    "history_inner_train_aggregates": history[-4:],
                    "goal": ("Minimize len_term (simulated mean dialogue length vs human mean, squared, "
                             "normalized) while keeping gap_cov and gap_comp >= 0 (constraints) and KL to the "
                             "reference gate small. You only see training aggregates.")}
    system = ("You tune three hyperparameters of an optimizer. Reply with ONLY a JSON object "
              "{\"lr\": float, \"beta\": float, \"w_len\": float, \"reason\": string}.")
    body = {"model": "gpt-5-mini", "reasoning_effort": "minimal",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(payload_view)}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
    raw, proposal = "", {}
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = json.loads(r.read())["choices"][0]["message"]["content"]
            s = raw[raw.find("{"): raw.rfind("}") + 1]
            proposal = json.loads(s)
            break
        except Exception as ex:  # logged, config kept
            raw = raw or "ERROR %s" % ex
            time.sleep(5)
    new = clip_config(proposal, cfg)
    rec = {"request_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
           "request_user": payload_view, "response": raw, "proposal": proposal, "applied": new}
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--controller", choices=("stageB", "fixed", "dual", "random", "llm"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    X, zb, tgt, sides, meta = load(args.features, args.outcomes, args.human)
    tr_idx = sorted({i for e in sides["inner_train"]["episodes"] for i in e["idx"]} | set(sides["inner_train"]["gold"]))
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-3
    X = torch.from_numpy((X - mu) / sd)
    zb, tgt = torch.from_numpy(zb), torch.from_numpy(tgt)
    hB = torch.sigmoid(zb)
    varK = float(np.var([e["K"] for e in sides["inner_train"]["episodes"]]) + 1e-6)
    model = Head(X.shape[1])
    cfg, lam = dict(C0), {"cov": 1.0, "comp": 1.0}
    log = []
    tr, va = sides["inner_train"], sides["inner_validation"]
    with torch.no_grad():
        p_tr = objective(model, X, zb, tgt, tr, cfg, lam, hB, varK, parts=True)
        p_va = objective(model, X, zb, tgt, va, cfg, lam, hB, varK, parts=True)
    best = {"round": 0, "V": val_score(p_va), "val": p_va, "train": p_tr, "cfg": dict(cfg),
            "state": {k: v.clone() for k, v in model.state_dict().items()}}
    log.append({"round": 0, "cfg": dict(cfg), "lam": dict(lam), "train": p_tr, "val": p_va, "V": best["V"]})
    rounds = 0 if args.controller == "stageB" else ROUNDS
    for rnd in range(1, rounds + 1):
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        for _ in range(STEPS):
            opt.zero_grad()
            loss, gc, gp = objective(model, X, zb, tgt, tr, cfg, lam, hB, varK)
            loss.backward()
            opt.step()
            if args.controller != "fixed":
                lam["cov"] = max(0.0, lam["cov"] - ETA * gc.item())
                lam["comp"] = max(0.0, lam["comp"] - ETA * gp.item())
        with torch.no_grad():
            p_tr = objective(model, X, zb, tgt, tr, cfg, lam, hB, varK, parts=True)
            p_va = objective(model, X, zb, tgt, va, cfg, lam, hB, varK, parts=True)
        V = val_score(p_va)
        log.append({"round": rnd, "cfg": dict(cfg), "lam": dict(lam), "train": p_tr, "val": p_va, "V": V})
        if V < best["V"]:
            best = {"round": rnd, "V": V, "val": p_va, "train": p_tr, "cfg": dict(cfg),
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        # the controller sees ONLY inner-train aggregates
        agg = {"round": rnd, "config": dict(cfg), **{k: round(v, 5) for k, v in p_tr.items()
                                                     if k not in ("n_gold",)}}
        if args.controller == "random":
            cfg = clip_config({k: math.exp(rng.uniform(math.log(lo), math.log(hi)))
                               for k, (lo, hi) in BOUNDS.items()}, cfg)
        elif args.controller == "llm":
            cfg = llm_propose([l_["train"] | {"config": l_["cfg"], "round": l_["round"]} for l_ in log],
                              cfg, os.path.join(args.out, "llm_controller.jsonl"))
    torch.save(best["state"], os.path.join(args.out, "best_head.pt"))
    np.savez(os.path.join(args.out, "standardizer.npz"), mu=mu, sd=sd)
    summary = {"controller": args.controller, "best_round": best["round"], "best_V": best["V"],
               "best_cfg": best["cfg"], "val": best["val"], "train": best["train"],
               "features": args.features, "seed": args.seed, "rounds": rounds, "steps_per_round": STEPS}
    json.dump({"summary": summary, "log": log}, open(os.path.join(args.out, "result.json"), "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
