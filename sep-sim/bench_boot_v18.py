#!/usr/bin/env python3
"""SPEC v18 §8: paired conversation bootstrap of the benchmark's per-turn dumps (score_method.py --dump-per-turn) --
v18 (GRPO+R) against v17 SFT (u0) and v17 GRPO (u5) on fold 2's test side (9 conversations, 40 user turns), 10000
resamples of the conversations (seed 0). Per metric: the two point values, the difference, its 95% CI and P(v18 better).

What can be recomputed per turn from the dumps (rows keyed "record_id|turn"):
  every numeric / boolean per-turn field present in all three dumps -> its mean over the scored turns (rates and means);
  Length W1 -> W1 between the non-empty selected utterances' whitespace lengths and the people's message lengths of the
  same resampled conversations (the corpus slice), i.e. the scorer's "utterance_length ... wasserstein_1" on a resample.
What cannot (no per-turn value in the dump): the 2AFC Judge Human-Spotting accuracy, Lexical Separability, MMD /
Frechet, the act-rate gaps -- their point values come from the results files only (reported, never bootstrapped here);
Stop-Point AUC has its own paired bootstrap in score_stop.py --baseline-decisions. 2AFC reads the first non-empty
SAMPLE, i.e. a candidate we did not output, and the reranker decides which one that is (§6.5): never read it as "the
output is more human-like". 9 conversations: directions only.

  python bench_boot_v18.py --dump-v18 D18.json --dump-u0 D0.json --dump-u5 D5.json --corpus SLICE.jsonl --out boot.json
"""
from __future__ import annotations

import argparse
import json
import random

LOWER_BETTER = {"length_w1": True, "utterance_repeating_own_earlier_utterance_in_session_at_unigram_jaccard_0_6": True,
                "turn_with_empty_output_without_end_decision": True,
                "turn_with_non_user_utterance_rule_hit_or_empty_output_without_end_decision": True,
                "tfidf_logistic_regression_out_of_fold_simulated_probability_mean_over_row_orders": None}


def load_dump(p):
    d = json.load(open(p, encoding="utf-8"))
    by = {}
    for k, r in d["rows"].items():
        by.setdefault(r["record_id"], {})[int(r["turn_index"])] = r
    return d.get("method_id"), by


def human_lengths(corpus):
    out = {}
    for l in open(corpus, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            out[r["record_id"]] = [len(m["text"].split()) for m in r["chat_messages"]
                                   if str(m.get("participant_name", "")).lower() == "user"]
    return out


def w1(a, b):
    if not a or not b:
        return None
    hi = max(max(a), max(b))
    w, ca, cb = 0.0, 0.0, 0.0
    for k in range(0, hi + 1):
        ca += sum(1 for x in a if x == k) / len(a)
        cb += sum(1 for x in b if x == k) / len(b)
        w += abs(ca - cb)
    return w


def numeric_fields(dumps):
    keys = None
    for _, by in dumps:
        ks = set()
        for turns in by.values():
            for r in turns.values():
                ks |= {k for k, v in r.items() if isinstance(v, (bool, int, float)) and k not in ("turn_index",)}
        keys = ks if keys is None else keys & ks
    return sorted(keys or [])


def metric(by, hum, recs, field):
    if field == "length_w1":
        sim = [int(by[r][t]["utterance_length_in_whitespace_tokens"]) for r in recs for t in by.get(r, {})
               if not by[r][t].get("turn_with_empty_output")]
        h = [x for r in recs for x in hum.get(r, [])]
        return w1(sim, h)
    xs = [float(by[r][t][field]) for r in recs for t in by.get(r, {}) if by[r][t].get(field) is not None]
    return (sum(xs) / len(xs)) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump-v18", required=True)
    ap.add_argument("--dump-u0", required=True)
    ap.add_argument("--dump-u5", required=True)
    ap.add_argument("--corpus", required=True, help="the fold-2 test-side corpus slice (bench_score_f2*.sh)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d18, d0, d5 = load_dump(a.dump_v18), load_dump(a.dump_u0), load_dump(a.dump_u5)
    hum = human_lengths(a.corpus)
    recs = sorted(set(d18[1]) & set(d0[1]) & set(d5[1]))
    if sorted(set(d18[1])) != recs or not recs:
        raise SystemExit("the dumps do not cover the same conversations")
    fields = ["length_w1"] + numeric_fields([d18, d0, d5])
    out = {"conversations": recs, "n_boot": a.n_boot, "methods": {"v18 (GRPO+R)": d18[0], "v17 SFT (u0)": d0[0],
                                                                  "v17 GRPO (u5)": d5[0]}, "comparisons": {}}
    for name, base in (("v17 SFT (u0)", d0), ("v17 GRPO (u5)", d5)):
        rng = random.Random(0)
        res = {}
        boots = {f: [] for f in fields}
        for _ in range(a.n_boot):
            rs = [rng.choice(recs) for _ in recs]
            for f in fields:
                x, y = metric(base[1], hum, rs, f), metric(d18[1], hum, rs, f)
                if x is not None and y is not None:
                    boots[f].append(y - x)
        for f in fields:
            pa, pb = metric(base[1], hum, recs, f), metric(d18[1], hum, recs, f)
            ds = sorted(boots[f])
            lb = LOWER_BETTER.get(f, None)
            res[f] = {"base": pa, "v18": pb, "diff": None if None in (pa, pb) else pb - pa,
                      "ci95": [ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds)) - 1]] if ds else None,
                      "p_v18_better": (sum(1 for x in ds if (x < 0 if lb else x > 0)) / len(ds)) if (ds and lb is not None) else None}
            print("%-16s %-90s %s -> %s  diff %s  CI %s" % (name, f[:90], _f(pa), _f(pb), _f(res[f]["diff"]), res[f]["ci95"]))
        out["comparisons"]["v18 (GRPO+R) vs " + name] = res
    out["not_bootstrapped"] = {
        "non_first_turn_first_non_empty_sample_context_free_llm_judge_two_alternative_forced_choice_human_identification_"
        "accuracy_vs_same_turn_gold_utterance": "Judge Human-Spotting (2AFC): POINT VALUE ONLY from the results files -- "
        "the per-turn dump has no 2AFC field, so no bootstrap CI; it reads the first non-empty SAMPLE (a candidate we did "
        "not output, decided by the reranker's pick, §6.5)",
        "lexical_separability / mmd / frechet / act-rate gaps": "point values only (no per-turn value)",
        "stop_point_auc": "score_stop.py --baseline-decisions (its own paired conversation bootstrap)"}
    print("NOT BOOTSTRAPPED: Judge Human-Spotting (2AFC) is a point value only (no per-turn dump field; it reads a "
          "candidate we did not output, SPEC v18 §6.5); lexical separability / MMD / Frechet / act gaps likewise; "
          "Stop AUC: score_stop.py's own paired bootstrap")
    out["note"] = ("per-turn dump fields only; 2AFC / lexical separability / MMD / act gaps: point values in the results "
                   "files; Stop AUC: score_stop.py's paired bootstrap; 9 conversations: directions only; the 2AFC reads "
                   "a candidate we did not output (SPEC v18 §6.5)")
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    return out


def _f(x):
    return "None" if x is None else "%.4f" % x


if __name__ == "__main__":
    main()
