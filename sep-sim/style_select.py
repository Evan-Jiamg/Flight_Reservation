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
