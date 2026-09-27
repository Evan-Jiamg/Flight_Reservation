#!/bin/bash
# User 2026-09-27: "u5 評估完先告訴我結果，然後因為組員要用，所以 u25 先暫停".
# Waits for the u5 re-selection summary, then stops the re-selection before/while u25 starts, releases the training
# GPU (the placeholder target goes to 0, so it does not take the card back) and writes the u5 report.
# Resume later: bash run_v15_reselect.sh (u5's summary is reused; only u25 runs; partial u25 episodes are reused).
G=/tmp2/mzjiang_usersim/grpo_planner; RUN=$G/runs/pend_f2_v11; H=$G/hold
settarget() { echo $2 > $H/target_$1.tmp && mv -f $H/target_$1.tmp $H/target_$1; }
echo "waiting for the u5 summary $(date)"
until python3 -c "
import json, sys
sys.exit(0 if any(json.loads(l).get('kind') == 'summary' and json.loads(l)['update'] == 5 for l in open('$RUN/reselect.jsonl') if l.strip()) else 1)"; do
  pgrep -u mzjiang -f run_v15_reselect.sh > /dev/null || { echo "re-selection ended before the u5 summary"; break; }
  sleep 20
done
echo "u5 summary present $(date); pausing u25"
pkill -u mzjiang -f run_v15_reselect.sh
sleep 2
RP=$(pgrep -u mzjiang -f "train_planner_rl.py --fold 2.*--reselect-seeds")
[ -n "$RP" ] && kill $RP
for i in $(seq 1 60); do pgrep -u mzjiang -f "train_planner_rl.py --fold 2" > /dev/null || break; sleep 2; done
TG=$(cat $H/role_train); settarget $TG 0
sleep 15
echo "stopped: $(pgrep -u mzjiang -af 'run_v15_reselect|train_planner_rl.py --fold 2' | wc -l) processes left; placeholder GPU$TG holds $(cat $H/status_$TG) GiB (target 0)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
python3 - <<'PYEOF' > $RUN/reselect_u5_report.txt 2>&1
import json, random, statistics
R = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
rows = [json.loads(l) for l in open(R + "reselect.jsonl") if l.strip()]
s = [r for r in rows if r.get("kind") == "summary" and r["update"] == 5][-1]
old = [json.loads(l) for l in open(R + "validation.jsonl") if l.strip()]
o = [r for r in old if r.get("kind") == "summary" and r["update"] == 5][-1]
ts, ots = s["turn_stats"] or {}, o["turn_stats"] or {}
print("u5 re-selection (seeds %s): n=%d unclean=%d selection=%s" % (s["val_seeds"], s["n_episodes"], s["n_unclean_episodes"], s["selection_score"]))
print("   sim turns %.2f  human %.2f  |diff| %.2f  W1 %.3f  coverage %.3f  task1 term_f1 %s  end %s" % (
    ts["sim_turns_mean"], ts["human_turns_mean"], ts["abs_diff_mean"], ts["turn_w1"], ts["coverage_mean"],
    (s.get("task1") or {}).get("term_f1"), ts.get("end_kinds")))
print("u5 in training (seeds %s): n=%d selection=%s sim %.2f W1 %.3f coverage %.3f" % (o["val_seeds"], o["n_episodes"],
    o["selection_score"], ots["sim_turns_mean"], ots["turn_w1"], ots["coverage_mean"]))
eps = {}
for r in rows:
    if r.get("kind") == "episode" and r["update"] == 5:
        eps[(r["conversation_id"], r["seed"])] = r["episode"]
eps = {k: e for k, e in eps.items() if e.get("clean")}
by = {}
for (c, sd), e in eps.items():
    by.setdefault(c, []).append(e)
print("per conversation (human turns | sim turns over 8 seeds | coverage mean):")
for c in sorted(by):
    es = by[c]
    print("   %s human %d | sim %s mean %.2f | cov %.3f" % (c[:8], es[0]["human_turns"], sorted(e["emitted_user_turns"] for e in es),
          statistics.mean(e["emitted_user_turns"] for e in es), statistics.mean(float(e["coverage"]) for e in es)))
convs = sorted(by)
rng = random.Random(0)
sims, errs = [], []
for _ in range(10000):
    es = [e for c in (rng.choice(convs) for _ in convs) for e in by[c]]
    sims.append(statistics.mean(e["emitted_user_turns"] for e in es))
    errs.append(statistics.mean(e["emitted_user_turns"] - e["human_turns"] for e in es))
for name, xs in (("sim turns", sims), ("sim - human", errs)):
    xs.sort()
    print("   %-11s 95%% CI over conversations [%.2f, %.2f]" % (name, xs[250], xs[9749]))
PYEOF
cat $RUN/reselect_u5_report.txt
echo "PAUSE DONE $(date)"
