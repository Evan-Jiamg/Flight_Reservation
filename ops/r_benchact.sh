B=/tmp2/hchsu/trec2026-usersim-benchmark; cd $B
echo "--- leaderboard.md (head)"; sed -n 1,60p leaderboard/leaderboard.md
echo "--- act-related keys in results/ditto_8b/main_dataset_search.json"
PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python - <<'EOF'
import json, os
B="/tmp2/hchsu/trec2026-usersim-benchmark/results"
def flat(d, p=""):
    for k, v in d.items():
        key = p + "." + k if p else k
        if isinstance(v, dict): yield from flat(v, key)
        elif isinstance(v, (int, float)) and any(s in key.lower() for s in ("act", "intent", "adher", "follow", "tvd")):
            yield key, v
for m in ("ditto_8b", "humanlm", "sep1st_v2fix", "sep1st_v2fix_noann", "ours_t1", "userlm_r1_repro", "ditto_8b_main_gptoss", "humanlm_main_gptoss"):
    f = f"{B}/{m}/main_dataset_search.json"
    if not os.path.exists(f): print(m, "missing"); continue
    d = json.load(open(f))
    items = list(flat(d))[:14]
    print("==", m, "|", "; ".join(f"{k}={v:.4g}" for k, v in items))
EOF
