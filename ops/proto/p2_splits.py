# Phase 0 (read-only): do the 3 outer folds' test_all partition the finished corpus? How many one-message
# conversations per split (under M2 they are almost always a false negative on termination_f1)?
import json
import sys
from collections import Counter

G = "/tmp2/mzjiang_usersim/grpo_planner"
sys.path.insert(0, G + "/trees/e1r_cf19400")
from sepsim import pipeline  # noqa: E402

recs = {}
for l in open("/home/mzjiang/v5-latency/data.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        recs[r["conversation_id"]] = r
finished = {c for c, r in recs.items() if any(m.get("is_final") is True for m in r.get("chat_messages", []))}
nuser = {c: len(pipeline.split_messages(recs[c])[0]) for c in finished}
sp = json.load(open(G + "/splits_v1.json", encoding="utf-8"))
union = []
for f in sp["folds"]:
    ta = f["test_all"]
    union += ta
    line = {"fold": f["fold"], "sizes": f.get("sizes")}
    for k in ("train", "train_all", "validation", "validation_all", "test", "test_all"):
        ids = f[k]
        line["%s_one_message" % k] = sum(1 for c in ids if nuser.get(c) == 1)
        line["%s_turns_hist" % k] = dict(sorted(Counter(min(nuser.get(c, 0), 10) for c in ids).items()))
    print(json.dumps(line))
cnt = Counter(union)
print(json.dumps({"finished_sessions": len(finished), "test_all_union": len(set(union)),
                  "sessions_in_two_test_folds": sum(1 for v in cnt.values() if v > 1),
                  "finished_not_in_any_test": len(finished - set(union)),
                  "test_ids_not_finished": len(set(union) - finished),
                  "partition": set(union) == finished and all(v == 1 for v in cnt.values()),
                  "corpus_one_message": sum(1 for c in finished if nuser[c] == 1),
                  "corpus_user_turns_total": sum(nuser.values())}))
