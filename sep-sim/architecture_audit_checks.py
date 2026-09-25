#!/usr/bin/env python3
"""Read-only data checks for the architecture audit (no model calls).

A. Planner end acts in Ditto rep0: how many coincide with a named non-weak stop_rule (so the
   read_plan override is the mechanism), how many are Complete acts with stop_rule none
   (the sampler's own choice); what Ditto actually says at the end step.
B. Task-2 ledger lines shown to the Planner: are useful/unhelpful/best-quality lines constant?
C. Corpus: scenario/goal fields, finished status of union scenarios, the human's last message.
D. Planner parse failures in the Ditto runs.
"""
import glob
import json
import random
import re
from collections import Counter

G = "/tmp2/mzjiang_usersim/grpo_planner"
R = G + "/stageC_v1"
WEAK = {"difference_threshold"}
RULES = {"disgust", "cost_exceeds_value", "satiation", "mental_list", "rate_of_gain", "difference_threshold"}

eps = [json.loads(l) for p in sorted(glob.glob(R + "/rep0_ditto_shard*/nogate.jsonl")) for l in open(p)]
print("== A. end-act mechanism (Ditto rep0, first end step per episode)")
mech = Counter()
ends_all = Counter()
examples = []
for e in eps:
    for s in e["trace"]:
        if s.get("ended_planner"):
            r = s.get("stop_rule", "none")
            ends_all["override-capable rule" if r in RULES - WEAK else ("weak rule" if r in WEAK else "no rule (sampler)")] += 1
    s = next((x for x in e["trace"] if x.get("ended_planner")), None)
    if s is None:
        continue
    r = s.get("stop_rule", "none")
    mech["named non-weak rule -> override" if r in RULES - WEAK else
         ("weak rule" if r in WEAK else "stop_rule none -> sampler chose Complete")] += 1
    examples.append((e["conversation_id"][:8], e["seed"], s["t"], s.get("move"), s.get("act"), r, (s.get("user") or "")[:160]))
print(dict(mech)); print("all end steps:", dict(ends_all))
random.Random(0).shuffle(examples)
for ex in examples[:8]:
    print("  ", ex)

print("== B. ledger lines in Task-2 planner prompts (Ditto rep0)")
vals = Counter()
for e in eps:
    for s in e["trace"]:
        p = s["gate_prompt"]
        m = re.search(r"- useful replies: (\d+), unhelpful: (\d+)", p)
        b = re.search(r"- best offered so far: (.*)", p)
        c = re.search(r"- (stopping conditions available: .*|no stopping condition is met)", p)
        vals[("useful=%s unhelpful=%s" % m.groups()) if m else "no line"] += 1
        vals["best=" + (b.group(1) if b else "none")] += 1
        vals[(c.group(1)[:60]) if c else "no cond line"] += 1
for k, v in vals.most_common(12):
    print("  ", v, k)

print("== C. corpus")
recs = [json.loads(l) for l in open("/home/mzjiang/v5-latency/data.jsonl")]
print("keys:", sorted(recs[0].keys()), "| scenario keys:", sorted(recs[0]["scenario"].keys()))
print("goal keys:", sorted((recs[0]["scenario"].get("goal") or {}).keys()))
msg_keys = Counter(k for r in recs for m in r["chat_messages"] for k in m)
print("message keys:", dict(msg_keys))
union = {s["conversation_id"] for s in json.load(open(R + "/scenarios_union.json"))["scenarios"]}
by = {r["conversation_id"]: r for r in recs}
fin = Counter()
last_user = []
for cid in union:
    r = by[cid]
    msgs = r["chat_messages"]
    fin[str([m.get("is_final") for m in msgs][-1:])] += 1
    us = [m["text"] for m in msgs if m["participant_name"].lower() == "user"]
    last_user.append((len(us), us[-1][:140] if us else ""))
print("union scenarios:", len(union), "| last message is_final:", dict(fin))
closing = sum(1 for _, t in last_user if re.search(r"\b(thank|thanks|great|perfect|that's all|bye)\b", t.lower()))
print("human last user message looks like a closing (thanks/great/perfect/bye):", closing, "/", len(last_user))
for k, t in random.Random(1).sample(last_user, 6):
    print("   K=%d last: %s" % (k, t))
print("scenario goal example:", json.dumps(by[sorted(union)[0]]["scenario"].get("goal"))[:600])

print("== D. planner parse failures")
for p in sorted(glob.glob(R + "/rep0_ditto_shard*/runstats_rep0.json")):
    print("  ", p.split("/")[-2], open(p).read().replace("\n", " ")[:200])
unp = sum(1 for e in eps for s in e["trace"] if s.get("planner_unparsed"))
print("  planner_unparsed steps (trace):", unp, "/", sum(len(e["trace"]) for e in eps))
