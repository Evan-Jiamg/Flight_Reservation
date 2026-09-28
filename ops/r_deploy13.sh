G=/tmp2/mzjiang_usersim/grpo_planner; S=$G/code_snapshots
echo "$PAYLOAD_B64" | base64 -d > /tmp2/mzjiang_usersim/v13deploy.tgz || exit 1
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
# nothing is stopped here: the v13 continuation waits until the v12 continuation has ended by itself
[ -d $S/pend_v13 ] && { echo "pend_v13 already exists"; exit 1; }
pgrep -u mzjiang -f "run_v13_continue|run_v13_guard" > /dev/null && { echo "v13 already queued"; exit 1; }
cp -a $S/pend_v12 $S/pend_v13 && tar xzf /tmp2/mzjiang_usersim/v13deploy.tgz -C $S/pend_v13 code
mv $S/pend_v13/code/* $S/pend_v13/ && rmdir $S/pend_v13/code
cd $S/pend_v13 && sha256sum -c --quiet local_sha_v13.txt && echo "pend_v13 sha OK ($(wc -l < local_sha_v13.txt) files)" || { echo "SHA MISMATCH"; exit 1; }
CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 nice -n 19 $PY -m pytest -q -p no:cacheprovider test_intervention.py test_r0_attribution.py \
    test_verify_pipeline.py test_llm4_controller.py test_rl_controllers.py test_rl_advantages.py 2>&1 | tail -1
tar xzf /tmp2/mzjiang_usersim/v13deploy.tgz -C $G --strip-components=1 ops/run_v13_continue.sh ops/run_v13_guard.sh ops/watch.sh \
    ops/intervention_w_dist_v13.json
chmod +x $G/run_v13_continue.sh $G/run_v13_guard.sh
setsid nohup bash $G/run_v13_continue.sh > $G/run_v13_continue.log 2>&1 < /dev/null &
sleep 5
setsid nohup bash $G/run_v13_guard.sh > $G/run_v13_guard.log 2>&1 < /dev/null &
sleep 10
echo "--- v13 continuation log"; cat $G/run_v13_continue.log | cut -c1-200
bash $G/watch.sh
