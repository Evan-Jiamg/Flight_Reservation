G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v16
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
echo "$PAYLOAD_B64" | base64 -d > /tmp2/mzjiang_usersim/v16night.tgz || exit 1
X=/tmp2/mzjiang_usersim/v16n; mkdir -p $X && rm -rf $X/* && tar xzf /tmp2/mzjiang_usersim/v16night.tgz -C $X
pgrep -u mzjiang -af "run_v16_night|run_v16_prelim|run_v16_launch|train_planner_rl" | grep -v pgrep && { echo "something of v16 already running - not touching"; exit 1; }
# 1. the audit-R fixes into the snapshot (verify_pipeline.py, step0_coverage.py, test_v16c.py, sha list), then sha + tests
chmod u+w $C/*
cp -f $X/code/* $C/
cd $C && sha256sum -c --quiet local_sha_v16.txt && echo "pend_v16 sha OK ($(wc -l < local_sha_v16.txt) files)" || { echo "SHA MISMATCH - not starting"; exit 1; }
CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 nice -n 19 $PY -m pytest -q -p no:cacheprovider test_v16.py test_v16b.py test_v16c.py \
    test_verify_pipeline.py test_rl_advantages.py test_pend.py 2>&1 | tail -1 | tee /tmp2/mzjiang_usersim/v16n/tests.txt
grep -q " failed" /tmp2/mzjiang_usersim/v16n/tests.txt && { echo "TESTS FAILED - not starting"; exit 1; }
# 2. the night scripts
for f in $X/ops/*.sh; do bash -n $f || { echo "SYNTAX ERROR $f - not starting"; exit 1; }; done
cp -f $X/ops/*.sh $G/ && chmod +x $G/run_v16_*.sh $G/watch_v16.sh
# 3. start the night runner (it waits for a completely free GPU, then prelim, then - only if the smoke passes - formal)
setsid nohup bash $G/run_v16_night.sh > $G/run_v16_night.log 2>&1 < /dev/null &
sleep 8
echo "--- night log"; cat $G/run_v16_night.log | cut -c1-200
bash $G/watch_v16.sh
