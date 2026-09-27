G=/tmp2/mzjiang_usersim/grpo_planner; S=$G/code_snapshots
echo "$PAYLOAD_B64" | base64 -d > /tmp2/mzjiang_usersim/v14deploy.tgz || exit 1
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
# nothing is stopped: the re-selection waits until the formal run has ended by itself
[ -d $S/pend_v14 ] && { echo "pend_v14 already exists"; exit 1; }
pgrep -u mzjiang -f run_v14_reselect > /dev/null && { echo "v14 already queued"; exit 1; }
cp -a $S/pend_v13 $S/pend_v14 && chmod -R u+w $S/pend_v14 && tar xzf /tmp2/mzjiang_usersim/v14deploy.tgz -C $S/pend_v14 code
mv -f $S/pend_v14/code/* $S/pend_v14/ && rmdir $S/pend_v14/code
cd $S/pend_v14 && sha256sum -c --quiet local_sha_v14.txt && echo "pend_v14 sha OK ($(wc -l < local_sha_v14.txt) files)" || { echo "SHA MISMATCH"; exit 1; }
CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 nice -n 19 $PY -m pytest -q -p no:cacheprovider test_reselect.py test_intervention.py \
    test_r0_attribution.py test_verify_pipeline.py test_rl_advantages.py test_llm4_controller.py 2>&1 | tail -1
tar xzf /tmp2/mzjiang_usersim/v14deploy.tgz -C $G --strip-components=1 ops/run_v14_reselect.sh ops/reselect_boot.py ops/watch.sh
chmod +x $G/run_v14_reselect.sh
setsid nohup bash $G/run_v14_reselect.sh > $G/run_v14_reselect.log 2>&1 < /dev/null &
sleep 10
echo "--- v14 log"; cat $G/run_v14_reselect.log | cut -c1-200
bash $G/watch.sh
