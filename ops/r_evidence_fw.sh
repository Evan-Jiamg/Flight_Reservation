G=/tmp2/mzjiang_usersim/grpo_planner; R=$G/runs/pend_f2_v16
echo "##### test_boot.txt (fold 2 test)"; cat $R/test_boot.txt
echo "##### reselect_boot.txt (fold 2 8-seed re-selection)"; cat $R/reselect_boot.txt
echo "##### validation.jsonl summaries (2-seed selection scores)"
python3 - <<'PYEOF'
import json
for fn in ("validation.jsonl", "reselect.jsonl"):
    for l in open("/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/" + fn):
        r = json.loads(l)
        if r.get("kind") == "summary":
            print(fn, "u%d" % r["update"], "selection_score", r["selection_score"], "seeds", r.get("val_seeds"))
print("##### updates.jsonl: mean emitted user turns of TRAIN rollouts per update")
import glob
for l in open("/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/updates.jsonl"):
    r = json.loads(l)
    tr = r.get("train") or r.get("train_stats") or {}
    cm = (tr.get("components_mean") or {}) if isinstance(tr, dict) else {}
    print("u%s" % r.get("update"), "turns_mean", cm.get("turns"), "keys", sorted(r.keys())[:12] if r.get("update") == 1 else "")
PYEOF
echo "##### v11 archive re-selection (u5 2-seed vs 8-seed)"
A=$(ls -d $G/archive/pend_f2_v11_u25*/ 2>/dev/null | head -1); echo "archive: $A"
python3 - "$A" <<'PYEOF'
import json, sys, os, glob
a = sys.argv[1]
for fn in glob.glob(os.path.join(a, "**", "validation.jsonl"), recursive=True) + glob.glob(os.path.join(a, "**", "reselect.jsonl"), recursive=True):
    for l in open(fn):
        r = json.loads(l)
        if r.get("kind") == "summary" and r["update"] in (0, 5, 10, 15, 20, 25):
            print(os.path.relpath(fn, a), "u%d" % r["update"], "sel", r["selection_score"], "seeds", r.get("val_seeds"))
PYEOF
echo "##### v16 smoke: grad norms"
grep -rhoE "\"(rl_grad_norm|aux_grad_norm)\": [0-9.e-]+" $G/runs/v16_prelim 2>/dev/null | sort | uniq -c | head -10
