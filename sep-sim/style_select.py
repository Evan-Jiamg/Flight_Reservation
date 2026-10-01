# -*- coding: utf-8 -*-
"""Length + style Selector (user decision, 2026-09-25).

Among the guard-passing candidates of a turn, rank by
  (a) length: |words(candidate) - the Planner's length target|  (smaller is better)
  (b) style:  cosine similarity of the candidate to the mean embedding of reference texts
              (larger is better). References: in Task 1 from turn 2 on, the person's OWN earlier real
              messages (part of the given history); on turn 1 and in Task 2, the same-style few-shot
              examples used this turn (other people's real messages from the allowed pool).
and pick the smallest rank sum (Borda). No weight: each signal has an equal say. Ties: better length
rank, then candidate order. A missing signal (no length target / no references) gives every candidate
the same rank on it, so the other signal decides.

Embeddings: princeton-nlp/sup-simcse-bert-base-uncased (the E1.6 StyleSelector's model), pooler output,
loaded with transformers directly (sentence-transformers is not installed in the pinned env), on CPU so it
never competes with generation for the GPU.
"""
from __future__ import annotations

import re
import threading

SIMCSE = "/tmp2/TREC_UserSim_MingZhi/models/princeton-nlp_sup-simcse-bert-base-uncased"


def n_words(text):
    return len(re.findall(r"\S+", text or ""))


def ranks(values, higher_better=False):
    """Competition ranks (0 = best); equal values share a rank."""
    order = sorted(set(values), reverse=higher_better)
    pos = {v: i for i, v in enumerate(order)}
    return [pos[v] for v in values]


def borda_pick(idxs, len_dist, style_sim):
    """idxs: candidate indices in order; len_dist/style_sim: per idx (None = signal missing).
    -> (chosen idx, info)."""
    if not idxs:
        raise ValueError("no candidates to select from")
    ld = [len_dist[i] if len_dist[i] is not None else 0 for i in idxs]
    ss = [style_sim[i] if style_sim[i] is not None else 0.0 for i in idxs]
    rl = ranks(ld, higher_better=False)
    rs = ranks(ss, higher_better=True)
    score = [a + b for a, b in zip(rl, rs)]
    best = min(range(len(idxs)), key=lambda k: (score[k], rl[k], idxs[k]))
    return idxs[best], {"candidates": list(idxs), "len_rank": rl, "style_rank": rs, "score": score,
                        "len_dist": [len_dist[i] for i in idxs], "style_sim": [style_sim[i] for i in idxs]}


class StyleScorer:
    def __init__(self, path=SIMCSE, device="cpu"):
        self.path, self.device = path, device
        self._tok = self._model = None
        self._lock = threading.Lock()

    def _load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.path)
        self._model = AutoModel.from_pretrained(self.path).to(self.device).eval()
        _ = torch

    CHUNK_TOKENS = 450         # < 512 BERT positions incl. [CLS]/[SEP], with room for decode/re-encode drift

    def embed(self, texts):
        """One normalised vector per text; a text longer than CHUNK_TOKENS wordpieces is split into chunks
        whose embeddings are averaged (no truncation anywhere)."""
        import torch
        with self._lock:
            if self._model is None:
                self._load()
            pieces, owner = [], []
            for i, t in enumerate(texts):
                ids = self._tok((t or ""), add_special_tokens=False)["input_ids"] or []
                for s in range(0, max(1, len(ids)), self.CHUNK_TOKENS):
                    pieces.append(self._tok.decode(ids[s:s + self.CHUNK_TOKENS]))
                    owner.append(i)
            enc = self._tok(pieces, padding=True, truncation=False, return_tensors="pt")
            if enc["input_ids"].shape[1] > 512:
                raise AssertionError("style chunk longer than 512 tokens")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                e = self._model(**enc).pooler_output.float()
            e = torch.nn.functional.normalize(e, dim=-1)
            out = torch.zeros(len(texts), e.shape[1])
            cnt = torch.zeros(len(texts), 1)
            for j, i in enumerate(owner):
                out[i] += e[j]
                cnt[i] += 1
            return torch.nn.functional.normalize(out / cnt, dim=-1)

    def similarities(self, cands, refs):
        """cosine(candidate, mean of reference embeddings) per candidate; None when no refs/empty cand."""
        refs = [r for r in (refs or []) if (r or "").strip()]
        if not refs:
            return [None] * len(cands)
        live = [i for i, c in enumerate(cands) if (c or "").strip()]
        out = [None] * len(cands)
        if not live:
            return out
        import torch
        E = self.embed([cands[i] for i in live])
        R = torch.nn.functional.normalize(self.embed(refs).mean(0, keepdim=True), dim=-1)
        sims = (E @ R.T)[:, 0].tolist()
        for i, s in zip(live, sims):
            out[i] = float(s)
        return out


def select(cands, eligible, target_words, refs, scorer):
    """Borda over the eligible candidates (all non-empty ones when none is eligible)."""
    idxs = list(eligible) if eligible else [i for i, c in enumerate(cands) if (c or "").strip()] or [0]
    len_dist = [abs(n_words(c) - target_words) if target_words else None for c in cands]
    style = scorer.similarities(cands, refs) if scorer is not None else [None] * len(cands)
    return borda_pick(idxs, len_dist, style)


# ================================================================== SPEC v18 §6 (human-likeness reranker)
STYLE_FEATURES = ("lowercase_start", "ends_with_punct", "n_question", "n_exclaim", "upper_ratio")


def style_features(text):
    """SPEC v18 §6.2: the 5 style features (no length): lowercase first letter, sentence-final punctuation, number of
    question marks, number of exclamation marks, share of uppercase among the letters."""
    t = (text or "").strip()
    letters = [ch for ch in t if ch.isalpha()]
    first = letters[0] if letters else ""
    return [1.0 if (first and first.islower()) else 0.0,
            1.0 if (t and t[-1] in ".!?") else 0.0,
            float(t.count("?")), float(t.count("!")),
            (sum(1 for ch in letters if ch.isupper()) / len(letters)) if letters else 0.0]


def borda_order(info):
    """The Borda ranking of borda_pick's info: candidate indices sorted by (score, len_rank, index) -- borda_pick's own
    tie-break, so order[0] is the Borda choice."""
    rows = sorted(zip(info["score"], info["len_rank"], info["candidates"]))
    return [c for _, _, c in rows]


def borda_top_k(info, k=2):
    """§6.4 step 2: the k best Borda candidates in Borda order, plus every candidate tied with the k-th on the Borda
    score. -> list of candidate indices."""
    score = dict(zip(info["candidates"], info["score"]))
    order = borda_order(info)
    top = order[:k]
    if len(order) > k:
        kth = score[top[-1]]
        top += [c for c in order[k:] if score[c] == kth]
    return top


def rerank_choice(info, rerank_scores, k=2):
    """§6.4 step 3: among borda_top_k, the highest human-likeness score; ties by the Borda score (lower), then the
    candidate index. rerank_scores: {candidate index: s(x)} (a candidate without a score -- a blank text -- ranks last).
    -> (chosen index, top-k list)."""
    score = dict(zip(info["candidates"], info["score"]))
    top = borda_top_k(info, k)
    best = min(top, key=lambda c: (-(rerank_scores.get(c) if rerank_scores.get(c) is not None else -1e300),
                                   score[c], c))
    return best, top


class HumanLikenessScorer:
    """SPEC v18 §6.2 runtime half of train_reranker.py: phi(x) = [PCA_32(SimCSE(x)), 5 style features], standardised;
    s(x) = w . phi(x). The SimCSE vector is StyleScorer.embed's (pooler output, normalised, chunked; CPU), the very
    function the reranker was fitted with. Only the ranking is used (no calibration)."""

    def __init__(self, path, embedder=None):
        import json
        self.path = path
        raw = open(path, "rb").read()
        import hashlib
        self.sha256 = hashlib.sha256(raw).hexdigest()
        d = json.loads(raw.decode("utf-8"))
        self.meta = d
        self.mean = d["pca"]["mean"]
        self.components = d["pca"]["components"]
        self.f_mean, self.f_std = d["scaler"]["mean"], d["scaler"]["std"]
        self.w = d["w"]
        if d.get("features", {}).get("style") != list(STYLE_FEATURES):
            raise ValueError("reranker %s: style features %r differ from this code's %r"
                             % (path, d.get("features", {}).get("style"), STYLE_FEATURES))
        self.gate_passed = bool((d.get("gate") or {}).get("passed"))
        self.embedder = embedder

    def features(self, texts, embeddings=None):
        import numpy as np
        if embeddings is None:
            embeddings = self.embedder.embed(texts)
        E = np.asarray(embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings, dtype=np.float64)
        P = (E - np.asarray(self.mean)) @ np.asarray(self.components).T
        S = np.asarray([style_features(t) for t in texts], dtype=np.float64)
        X = np.concatenate([P, S], axis=1)
        return (X - np.asarray(self.f_mean)) / np.asarray(self.f_std)

    def scores(self, texts):
        """s(x) per text; None for a blank text."""
        import numpy as np
        live = [i for i, t in enumerate(texts) if (t or "").strip()]
        out = [None] * len(texts)
        if not live:
            return out
        X = self.features([texts[i] for i in live])
        s = X @ np.asarray(self.w)
        for i, v in zip(live, s.tolist()):
            out[i] = float(v)
        return out

    def describe(self):
        g = self.meta.get("gate") or {}
        return {"path": self.path, "sha256": self.sha256, "C": self.meta.get("C"), "gate": g,
                "n_pairs": self.meta.get("n_pairs"), "embed_model": self.meta.get("embed_model")}


def select_rerank(cands, eligible, target_words, refs, scorer, hl_scorer, k=2):
    """SPEC v18 §6.4: Borda first (select(), unchanged), then the human-likeness reranker chooses among the Borda top-k
    (ties at the k-th included). -> (chosen index, info): borda_pick's info plus borda_index (the v17 choice), topk,
    rerank_scores (aligned with info["candidates"], None where not scored), rerank_changed."""
    b_idx, info = select(cands, eligible, target_words, refs, scorer)
    top = borda_top_k(info, k)
    sc = hl_scorer.scores([cands[i] for i in top])
    rs = dict(zip(top, sc))
    choice, top2 = rerank_choice(info, rs, k)
    assert top2 == top
    info = dict(info, borda_index=b_idx, topk=top, rerank_k=int(k),
                rerank_scores=[rs.get(c) for c in info["candidates"]], rerank_changed=choice != b_idx,
                reranker_sha256=hl_scorer.sha256)
    return choice, info
