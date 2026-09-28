G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v16
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
pgrep -u mzjiang -af "run_v16_night|run_v16_prelim|run_v16_launch|train_planner_rl" | grep -v pgrep && { echo "v16 running - not touching"; exit 1; }
echo "$PAYLOAD_B64" | base64 -d > /tmp2/mzjiang_usersim/v16pin2.tgz && X=/tmp2/mzjiang_usersim/v16pin2 && mkdir -p $X && rm -rf $X/* && tar xzf /tmp2/mzjiang_usersim/v16pin2.tgz -C $X
chmod u+w $C/* && cp -f $X/code/* $C/
cd $C && sha256sum -c --quiet local_sha_v16.txt && echo "pend_v16 sha OK ($(wc -l < local_sha_v16.txt) files)" || { echo "SHA MISMATCH"; exit 1; }
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" $PY - <<'PYEOF' || { echo "IMPORT / DATA CHECK FAILED - not restarting"; exit 1; }
import sys, os
sys.path.insert(0, os.getcwd())
import task2_env as TE
TE.setup_environment("pend")
import r0_client, metrics.judge
TE.check_bench_module(r0_client); TE.check_bench_module(metrics.judge)
print("pinned tools OK:", r0_client.R0Client.__name__, r0_client.Ledger.__name__, metrics.judge.Judge.__name__)
print("data OK:", {k.split("/")[-1]: v[:12] for k, v in TE.check_bench_data().items()})
PYEOF
CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 nice -n 19 $PY -m pytest -q -p no:cacheprovider test_v16.py test_v16b.py test_v16c.py test_pend.py test_pend_generate.py 2>&1 | tail -1
rm -rf $G/runs/v16_prelim/smoke_run
setsid nohup bash $G/run_v16_night.sh > $G/run_v16_night.log 2>&1 < /dev/null &
sleep 8
echo "--- night log"; cat $G/run_v16_night.log | cut -c1-200
bash $G/watch_v16.sh | head -1
