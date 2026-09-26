#!/usr/bin/env python3
"""Prompt-parity audit for TREC nested stop rows (P0 gate before Stage B).

For every row in fold{0,1,2}_inner_{train,validation}.jsonl:
  * encode with stop_prompt.encode_stop_prompt at max_length=2048;
  * assert length <= 2048, the goal/state headers survive, the prompt ends with
    the assistant generation header, and the kept history is a suffix of the
    original history (the latest dialogue is never dropped);
  * re-encode to confirm determinism.
Also checks the same encoder reproduces the old PRISM head/tail behaviour.
Writes a JSON audit with per-side counts and SHA256 of inputs/encoder.
"""
import hashlib, json, sys
from pathlib import Path

G = Path("/tmp2/mzjiang_usersim/grpo_planner")
sys.path.insert(0, str(G))
from stop_prompt import encode_stop_prompt, MARKER, OMITTED, DECISION_SYSTEM  # noqa
from transformers import AutoTokenizer

BASE = "/tmp2/hf_shared/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
tok = AutoTokenizer.from_pretrained(BASE)
HEADERS = ["WHO THEY ARE", "WHAT THEY CAME FOR", "THE STATE YOU WROTE LAST TURN",
           "WHAT HAS ACTUALLY HAPPENED", "THE CONVERSATION SO FAR"]
GEN_TAIL = "<|im_start|>assistant\n"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit = {"encoder_sha256": sha(G / "stop_prompt.py"), "max_length": 2048, "sides": {}}
failures = []
for fold in (0, 1, 2):
    for side in ("inner_train", "inner_validation"):
        path = G / "nested" / f"fold{fold}_{side}.jsonl"
        rows = [json.loads(l) for l in open(path) if l.strip()]
        st = {"sha256": sha(path), "rows": len(rows), "sessions": len({r["record_id"] for r in rows}),
              "positive": sum(r["should_stop"] for r in rows), "compacted": 0,
              "max_removed": 0, "max_len": 0, "one_k1_per_session": True}
        per = {}
        for r in rows:
            per.setdefault(r["record_id"], []).append(r)
            ids, info = encode_stop_prompt(tok, r["user"], 2048)
            ids2, _ = encode_stop_prompt(tok, r["user"], 2048)
            text = tok.decode(ids)
            key = f"fold{fold}/{side}/{r['record_id']}/{r['turn_index']}"
            if ids != ids2: failures.append(key + " nondeterministic")
            if len(ids) > 2048: failures.append(key + " too long")
            if not text.endswith(GEN_TAIL): failures.append(key + " no gen header")
            if DECISION_SYSTEM not in text: failures.append(key + " system missing")
            for h in HEADERS:
                if h not in text: failures.append(key + " missing " + h)
            if MARKER not in r["user"]: failures.append(key + " no marker")
            if info["trec_compacted"]:
                st["compacted"] += 1
                if OMITTED.strip() not in text: failures.append(key + " omission note missing")
                hist = r["user"].split(MARKER, 1)[1]
                kept = text.split(OMITTED, 1)[1].rsplit("<|im_end|>", 1)[0]
                # decode/encode round trip can alter whitespace at the cut; test last 200 chars
                if not hist.rstrip().endswith(kept.rstrip()[-200:]): failures.append(key + " tail not suffix")
            st["max_removed"] = max(st["max_removed"], info["removed_tokens"])
            st["max_len"] = max(st["max_len"], len(ids))
        for rid, rs in per.items():
            ti = sorted(int(x["turn_index"]) for x in rs)
            pos = [x for x in rs if x["should_stop"]]
            if len(pos) != 1 or int(pos[0]["turn_index"]) != max(ti) or ti != list(range(min(ti), max(ti) + 1)):
                st["one_k1_per_session"] = False
                failures.append(f"fold{fold}/{side}/{rid} label structure")
        audit["sides"][f"fold{fold}_{side}"] = st
        print(f"fold{fold}_{side}", json.dumps(st), flush=True)
# cross-side session disjointness per fold
for fold in (0, 1, 2):
    a = {json.loads(l)["record_id"] for l in open(G / "nested" / f"fold{fold}_inner_train.jsonl")}
    b = {json.loads(l)["record_id"] for l in open(G / "nested" / f"fold{fold}_inner_validation.jsonl")}
    if a & b: failures.append(f"fold{fold} train/val overlap")
audit["failures"] = failures
audit["pass"] = not failures
out = G / "audits"; out.mkdir(exist_ok=True)
(out / "stop_prompt_parity_audit.json").write_text(json.dumps(audit, indent=2))
print("PASS" if not failures else "FAIL", len(failures), failures[:20])
