#!/usr/bin/env python3
"""Summarize Task 2 episodes and paired differences with the SCENARIO as the unit.

Turns are non-empty emitted user utterances. decision_steps additionally counts
a gate stop or an empty draw. end_kind is reported as separate rates
(stop_gate / empty / speaker_end / planner_end / t_max) instead of one merged
"ended_by_token".

Paired inference: for every scenario, average the arm difference over the seeds
present in both arms; the scenario means are the independent units for the SE
(n_scenarios - 1 denominator) and for the cluster bootstrap. If R0/judge are
stochastic and not replicated, the interval covers scenario sampling for one
realisation of the environment only (say so in any report).

Accepts both the corrected runner (rollout_stop_sft.py, has emitted_user_turns)
and the legacy runner (turns = decision steps; recomputed from the trace).
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from math import sqrt

END_KINDS = ("stop_gate", "empty", "speaker_end", "planner_end", "t_max")
LEGACY_KIND = {"end_token": "speaker_end", "planner": "planner_end"}


def read(path):
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    out = {}
    for row in rows:
        if "trace" not in row:
            raise ValueError("episode trace required to count emitted user turns")
        emitted = sum(bool((t.get("user") or "").strip()) for t in row["trace"])
        if "emitted_user_turns" in row:
            if row["emitted_user_turns"] != emitted:
                raise ValueError("emitted_user_turns disagrees with trace")
            row["corrected_runner"] = True
        else:
            row["decision_steps"] = int(row["turns"])
            kind = row.get("stop_kind", "unknown")
            row["end_kind"] = LEGACY_KIND.get(kind, kind)
            row["corrected_runner"] = False
        row["emitted_user_turns"] = emitted
        key = (row["conversation_id"], int(row["seed"]))
        if key in out:
            raise ValueError("duplicate episode %s" % (key,))
        out[key] = row
    return out


def value(row, name):
    if name.startswith("end_"):
        return float(row["end_kind"] == name[4:])
    if name == "complete":
        return float(bool(row["complete"]))
    return float(row[name])


METRICS = ("emitted_user_turns", "decision_steps", "coverage", "complete") + \
          tuple("end_" + k for k in END_KINDS)


def summary(rows):
    values = list(rows.values())
    n = len(values)
    kinds = Counter(r["end_kind"] for r in values)
    return {"episodes": n, "scenarios": len({k[0] for k in rows}),
            "corrected_runner": all(r["corrected_runner"] for r in values),
            "turns_definition": "non-empty emitted user utterances",
            "emitted_user_turns_mean": sum(r["emitted_user_turns"] for r in values) / n,
            "decision_steps_mean": sum(r["decision_steps"] for r in values) / n,
            "end_kind_rate": {k: kinds.get(k, 0) / n for k in END_KINDS},
            "end_kind_count": dict(kinds),
            "coverage_mean": sum(r["coverage"] for r in values) / n,
            "complete_rate": sum(bool(r["complete"]) for r in values) / n,
            "complete_count": sum(bool(r["complete"]) for r in values)}


def paired(a, b, n_boot=4000, seed=20260924):
    common = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise ValueError("paired episode ids differ: %d vs %d (common %d)"
                         % (len(a), len(b), len(common)))
    by_scenario = defaultdict(list)
    for key in common:
        by_scenario[key[0]].append(key)
    scen = sorted(by_scenario)
    m = len(scen)
    result = {"n_scenarios": m, "n_episodes": len(common), "unit": "scenario (seed-averaged)"}
    rng = random.Random(seed)
    boot_idx = [[rng.randrange(m) for _ in range(m)] for _ in range(n_boot)]
    for name in METRICS:
        d = [sum(value(b[k], name) - value(a[k], name) for k in by_scenario[s]) /
             len(by_scenario[s]) for s in scen]
        mean = sum(d) / m
        se = sqrt(sum((x - mean) ** 2 for x in d) / (m - 1) / m) if m > 1 else float("nan")
        draws = sorted(sum(d[i] for i in idx) / m for idx in boot_idx)
        result[name] = {"mean_difference_new_minus_base": mean, "scenario_se": se,
                        "scenario_bootstrap_95": [draws[int(.025 * n_boot)],
                                                  draws[int(.975 * n_boot) - 1]]}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new")
    args = ap.parse_args()
    a = read(args.base)
    result = {"base": summary(a)}
    if args.new:
        b = read(args.new)
        result.update(new=summary(b), paired=paired(a, b))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
