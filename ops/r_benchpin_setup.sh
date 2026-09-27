G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v16; B=/tmp2/hchsu/trec2026-usersim-benchmark
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
P=$G/bench_pin/ca13b33
pgrep -u mzjiang -af "run_v16_night|run_v16_prelim|run_v16_launch|train_planner_rl" | grep -v pgrep && { echo "v16 running - not touching"; exit 1; }
# 1. pinned copy of the benchmark tools at ca13b33 (read-only export from the shared repo)
if [ ! -d $P/tools ]; then
  mkdir -p $P && git -c safe.directory=$B -C $B archive ca13b33 tools | tar -x -C $P || { echo "archive failed"; exit 1; }
  ( cd $P && find tools -type f | sort | xargs sha256sum > MANIFEST_tools.sha256 )
  sha256sum $B/data/req_shards_v1.json $B/domains/main_dataset_search/folds3_goal_persona_v1.json > $P/MANIFEST_data_live.sha256
  git -c safe.directory=$B -C $B log --format="%H %ad %s" --date=iso -1 ca13b33 > $P/COMMIT.txt
  chmod -R a-w $P/tools $P/MANIFEST_tools.sha256 $P/MANIFEST_data_live.sha256 $P/COMMIT.txt
fi
cat $P/COMMIT.txt; echo "pinned files: $(wc -l < $P/MANIFEST_tools.sha256)"
( cd $P && sha256sum -c --quiet MANIFEST_tools.sha256 ) && echo "pin manifest OK"
grep -c "^class R0Client\|^class Ledger" $P/tools/r0_client.py
# 2. the code update into pend_v16
echo "$PAYLOAD_B64" | base64 -d > /tmp2/mzjiang_usersim/v16pin.tgz && X=/tmp2/mzjiang_usersim/v16pin && mkdir -p $X && rm -rf $X/* && tar xzf /tmp2/mzjiang_usersim/v16pin.tgz -C $X
chmod u+w $C/* && cp -f $X/code/* $C/
cd $C && sha256sum -c --quiet local_sha_v16.txt && echo "pend_v16 sha OK ($(wc -l < local_sha_v16.txt) files)" || { echo "SHA MISMATCH"; exit 1; }
# 3. the real import path: pinned modules, the API our code uses
cd $C && PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" $PY - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import task2_env as TE
TE.setup_environment("pend")
import r0_client, metrics.judge
TE.check_bench_module(r0_client); TE.check_bench_module(metrics.judge)
print("r0_client from", r0_client.__file__)
print("judge from", metrics.judge.__file__)
print("API:", r0_client.R0Client.__name__, r0_client.Ledger.__name__, metrics.judge.Judge.__name__)
live = os.path.join(TE.BENCH, "tools")
print("live tools on sys.path:", live in sys.path)
PYEOF
CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 nice -n 19 $PY -m pytest -q -p no:cacheprovider test_v16.py test_v16b.py test_v16c.py \
    test_pend.py test_audit4.py test_pend_generate.py 2>&1 | tail -1
