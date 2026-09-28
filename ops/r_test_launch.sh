G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v16; RUN=$G/runs/pend_f2_v16
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
T=$G/.test_deploy; rm -rf $T; mkdir -p $T
echo "$PAYLOAD_B64" | base64 -d | tar xzf - -C $T || { echo "UNPACK FAILED"; exit 1; }
cp $T/eval_test_rl.py $T/eval_test_boot.py $T/local_sha_v16_test.txt $C/ && cp $T/run_v16_test.sh $G/ && chmod +x $G/run_v16_test.sh
cd $C && sha256sum -c local_sha_v16_test.txt && sha256sum -c --quiet local_sha_v16.txt && echo "snapshot sha OK"
ls $RUN/test.jsonl $RUN/test_meta.jsonl 2>&1 | cut -c1-90
echo "=== pre-flight: the fold-2 test ids"
cd $C && PYTHONNOUSERSITE=1 $PY - <<'PYEOF' || { echo "PREFLIGHT FAILED - not launching"; exit 1; }
import json, sys
d = {int(f["fold"]): f for f in json.load(open("/tmp2/mzjiang_usersim/grpo_planner/splits_v1.json"))["folds"]}[2]
test, test_all = sorted(d["test"]), sorted(d["test_all"])
recs = {}
for l in open("/home/mzjiang/v5-latency/data.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l); recs[r["conversation_id"]] = r
F = json.load(open("/tmp2/hchsu/trec2026-usersim-benchmark/domains/main_dataset_search/folds3_goal_persona_v1.json"))
import task2_env as TE
reqs = json.load(open(TE.BENCH + "/data/req_shards_v1.json"))
bad = []
for c in test_all:
    miss = [k for k, ok in (("corpus", c in recs), ("goal_of", c in F["goal_of"]), ("persona_of", c in F["persona_of"])) if not ok]
    if c in test and c not in reqs:
        miss.append("req_shards")
    print(c[:16], "Task2" if c in test else "Task1-only", "OK" if not miss else "MISSING %s" % miss)
    bad += miss
print("test %d, test_all %d" % (len(test), len(test_all)))
sys.exit(1 if bad else 0)
PYEOF
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
setsid nohup bash $G/run_v16_test.sh > $G/run_v16_test.log 2>&1 < /dev/null &
sleep 30
echo "=== launcher log"; cat $G/run_v16_test.log
pgrep -u mzjiang -af "run_v16_test|gpu_holder2|vllm serve|eval_test_rl" | grep -v pgrep | cut -c1-90
