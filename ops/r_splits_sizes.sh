python3 - <<'PYEOF'
import json
d = json.load(open("/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json"))
for f in d["folds"]:
    print("fold", f["fold"], {k: len(f[k]) for k in ("train", "train_all", "validation", "validation_all", "test", "test_all") if k in f})
PYEOF
