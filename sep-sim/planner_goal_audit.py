#!/usr/bin/env python3
"""Where does the Planner's stop decision go wrong, and does it see the whole dialogue?

Per logged no-gate Ditto episode:
  * Planner prompt length (tokens) at every step vs the 12000-token truncation of
    models.Planner.raw_with (tokenizer truncation is right-sided: it would drop the NEWEST text).
  * At the Planner's first end act t_p: requirement shards revealed / satisfied so far
    (ledger revealed_at / satisfied_at <= t_p-1), and whether coverage/satisfaction still
    increased after t_p in the no-gate continuation (= premature end, measurable here only
    because PLANNER_END was off).
  * Episodes with no end act: final satisfaction.
"""
import glob
import json
import sys

from transformers import AutoTokenizer

R = "/tmp2/mzjiang_usersim/grpo_planner/stageC_v1"
PLANNER = "/tmp2/TREC_UserSim_MingZhi/UserLM/00-models-v4GRPO-deps/Qwen2.5-32B-Instruct"
sys.path.insert(0, "/home/mzjiang/Sep-Simulator")
from sepsim import planner_prompt as PP  # noqa: E402


def main(pattern):
    tok = AutoTokenizer.from_pretrained(PLANNER)
    SYS = PP.system_prompt()
    eps = [json.loads(l) for p in sorted(glob.glob(pattern)) for l in open(p)]
    lens, over = [], 0
    prem, ontime, sat_frac, rev_frac, no_end = 0, 0, [], [], []
    for e in eps:
        for s in e["trace"]:
            text = tok.apply_chat_template([{"role": "system", "content": SYS},
                                            {"role": "user", "content": s["gate_prompt"]}],
                                           tokenize=False, add_generation_prompt=True)
            n = len(tok(text)["input_ids"])
            lens.append(n)
            over += n > 12000
        led = e["ledger"] or {}
        rev, sat, nreq = led.get("revealed_at", {}), led.get("satisfied_at", {}), e["n_req"]
        tp = next((s["t"] for s in e["trace"] if s.get("ended_planner")), None)
        if tp is None:
            no_end.append(len(sat) / nreq)
            continue
        r_before = sum(1 for v in rev.values() if v <= tp - 1)
        s_before = sum(1 for v in sat.values() if v <= tp - 1)
        rev_frac.append(r_before / nreq)
        sat_frac.append(s_before / nreq)
        later = sum(1 for v in rev.values() if v >= tp) + sum(1 for v in sat.values() if v >= tp)
        if later:
            prem += 1
        else:
            ontime += 1
    lens.sort()
    print(json.dumps({
        "episodes": len(eps), "planner_prompt_tokens": {"median": lens[len(lens) // 2], "p95": lens[int(.95 * len(lens))],
                                                         "max": lens[-1], "over_12000": over, "steps": len(lens)},
        "with_end_act": len(sat_frac), "premature_end_more_progress_later": prem, "no_progress_after_end": ontime,
        "at_end_revealed_frac_mean": sum(rev_frac) / len(rev_frac),
        "at_end_satisfied_frac_mean": sum(sat_frac) / len(sat_frac),
        "no_end_act_episodes": len(no_end)}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
