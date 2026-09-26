"""Server-side test of the D5 truncation fix with the REAL tokenizers (no model weights loaded).

1. Regression: for every logged Planner prompt of Ditto rep0 (680) whose chat text is <= 12000
   tokens, TokProxy(...)(text, truncation=True, max_length=12000) returns exactly the ids of the
   legacy call; for longer ones the proxy returns the FULL prompt (legacy dropped the end).
2. fit_planner_user leaves every logged prompt unchanged (all fit the 32104 budget) and, on a
   synthetic 60k-token history, returns a prompt within budget that keeps the static prefix,
   ends with the assistant generation header, and keeps the newest user message.
3. FitDittoSpeaker.build_prompt (tokenizer only): unchanged prompt when it fits; on a synthetic
   over-long history it drops the oldest exchanges, keeps the newest, stays within budget and
   ends with the generation header. Legacy-cap regression for the 7000 cap as in (1).
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/mzjiang/Sep-Simulator")
sys.modules.setdefault("torchvision", None)
sys.modules.setdefault("torchaudio", None)

from transformers import AutoTokenizer  # noqa: E402
import fit_prompts as F  # noqa: E402
from sepsim import models, planner_prompt as PP  # noqa: E402

PLANNER = "/tmp2/TREC_UserSim_MingZhi/UserLM/00-models-v4GRPO-deps/Qwen2.5-32B-Instruct"
DITTO = "/tmp2/mzjiang_usersim/models/Ditto-8B"
R = "/tmp2/mzjiang_usersim/grpo_planner/stageC_v1"


def main():
    ptok = AutoTokenizer.from_pretrained(PLANNER)
    SYS = PP.system_prompt()
    proxy = F.TokProxy(ptok, F.PLANNER_BUDGET)
    eps = [json.loads(l) for p in sorted(glob.glob(R + "/rep0_ditto_shard*/nogate.jsonl")) for l in open(p)]
    same = longer = 0
    for e in eps:
        for s in e["trace"]:
            up = s["gate_prompt"]
            fitted, info = F.fit_planner_user(ptok, SYS, up)
            assert fitted == up and not info["compacted"]
            text = ptok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": up}],
                                            tokenize=False, add_generation_prompt=True)
            legacy = ptok(text, return_tensors="pt", truncation=True, max_length=12000)["input_ids"][0].tolist()
            new = proxy(text, return_tensors="pt", truncation=True, max_length=12000)["input_ids"][0].tolist()
            full = ptok(text, return_tensors="pt")["input_ids"][0].tolist()
            assert new == full
            if len(full) <= 12000:
                assert new == legacy
                same += 1
            else:
                assert legacy == full[:12000] and new != legacy
                longer += 1
    print("planner regression: %d identical to legacy, %d previously truncated now full" % (same, longer))

    # synthetic over-long Planner prompt
    up = eps[0]["trace"][3]["gate_prompt"]
    static, hist = up.split(F.MARKER, 1)
    newest = "USER: THE NEWEST QUESTION about licensing"
    blocks = ["USER: old question %d\n\nASSISTANT: %s" % (i, "dataset listing " * 400) for i in range(40)]
    long_up = static + F.MARKER + "\n\n".join(blocks) + "\n\n" + newest
    fitted, info = F.fit_planner_user(ptok, SYS, long_up)
    text = ptok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": fitted}],
                                    tokenize=False, add_generation_prompt=True)
    n = len(ptok(text)["input_ids"])
    assert info["compacted"] and n <= F.PLANNER_BUDGET, (info, n)
    assert fitted.startswith(static + F.MARKER + F.OMITTED)
    assert fitted.endswith(newest) and text.endswith("<|im_start|>assistant\n")
    print("planner synthetic fit ok:", info, "final", n)

    # Ditto speaker (tokenizer only)
    FitDitto = F.make_fit_ditto_speaker(models.DittoSpeaker)
    sp = FitDitto(path=DITTO, gpu=None, position="system")
    sp._tok = AutoTokenizer.from_pretrained(DITTO)       # load() would also load weights; tokenizer suffices
    sp._raw_tok = sp._tok
    sp._tok = F.TokProxy(sp._raw_tok, sp.budget)
    e = eps[0]
    hu = [s["user"] for s in e["trace"] if s["decision"] == "continue"]
    ha = [s["agent"] for s in e["trace"] if s["decision"] == "continue"]
    sc_text = "persona and goal text"
    block = "- act: Inquire / ask_more"
    p = sp.build_prompt(sc_text, block, hu, ha, len(hu) + 1)
    assert not sp.last_fit["compacted"]
    legacy = sp._raw_tok(p, return_tensors="pt", truncation=True, max_length=7000)["input_ids"][0].tolist()
    new = sp._tok(p, return_tensors="pt", truncation=True, max_length=7000)["input_ids"][0].tolist()
    assert new == sp._raw_tok(p, return_tensors="pt")["input_ids"][0].tolist()
    print("ditto prompt tokens", len(new), "legacy identical:", new == legacy)
    big_u = ["old user turn %d" % i for i in range(60)] + ["NEWEST USER TURN"]
    big_a = [("assistant reply %d " % i) + "dataset listing " * 350 for i in range(60)] + ["latest reply"]
    p = sp.build_prompt(sc_text, block, big_u, big_a, 62)
    m = len(sp._raw_tok(p)["input_ids"])
    assert sp.last_fit["compacted"], sp.last_fit
    assert m <= sp.budget, m
    assert "NEWEST USER TURN" in p and "latest reply" in p
    import re
    assert not re.search(r"old user turn 0\b", p), "oldest exchange should have been dropped"
    assert F.SPEAKER_OMISSION_NOTE.strip() in p
    # generation tail must be identical to that of an unfitted reference prompt
    ref = models.DittoSpeaker.build_prompt(sp, sc_text, block, ["xyzzy-u"], ["xyzzy-a"], 2)
    gen_tail = ref[ref.rfind("xyzzy-a") + len("xyzzy-a"):]
    assert p.endswith(gen_tail), (repr(p[-80:]), repr(gen_tail))
    print("ditto synthetic fit ok:", sp.last_fit, "gen tail:", repr(gen_tail))
    print("ALL FIT TESTS PASSED")


if __name__ == "__main__":
    main()
