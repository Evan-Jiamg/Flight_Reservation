G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold; S=$G/code_snapshots
echo "$PAYLOAD_B64" | base64 -d > /tmp2/mzjiang_usersim/v12deploy.tgz || exit 1
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
if pgrep -u mzjiang -f "train_planner_rl.py|run_v11_continue|run_v12_continue|run_v11_guard|run_v12_guard" > /dev/null; then
  echo "something is still running; not deploying"; pgrep -u mzjiang -af "train_planner_rl.py|run_v1" | cut -c1-120; exit 1; fi
[ -d $S/pend_v12 ] && { echo "pend_v12 already exists"; exit 1; }
cp -a $S/pend_v11 $S/pend_v12 && tar xzf /tmp2/mzjiang_usersim/v12deploy.tgz -C $S/pend_v12 code
mv $S/pend_v12/code/* $S/pend_v12/ && rmdir $S/pend_v12/code
cd $S/pend_v12 && sha256sum -c --quiet local_sha_v12.txt && echo "pend_v12 sha OK ($(wc -l < local_sha_v12.txt) files)" || { echo "SHA MISMATCH"; exit 1; }
PYTHONNOUSERSITE=1 $PY -m pytest -q -p no:cacheprovider test_r0_attribution.py test_verify_pipeline.py 2>&1 | tail -1
tar xzf /tmp2/mzjiang_usersim/v12deploy.tgz -C $G --strip-components=1 ops/run_v12_continue.sh ops/run_v12_guard.sh ops/watch.sh
chmod +x $G/run_v12_continue.sh $G/run_v12_guard.sh
# placeholder: keep the roles (server GPU keeps the running servers; training GPU taken all at once at 45 GiB)
pkill -u mzjiang -f gpu_holder2.py; sleep 3; rm -f $H/stop
PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder2.py $H >> $G/gpu_holder2.log 2>&1 < /dev/null &
sleep 10
echo 45 > $H/target_$(cat $H/role_train).tmp && mv -f $H/target_$(cat $H/role_train).tmp $H/target_$(cat $H/role_train)
setsid nohup bash $G/run_v12_continue.sh > $G/run_v12_continue.log 2>&1 < /dev/null &
sleep 5
setsid nohup bash $G/run_v12_guard.sh > $G/run_v12_guard.log 2>&1 < /dev/null &
sleep 60
echo "--- continuation log"; cat $G/run_v12_continue.log | cut -c1-250
echo "--- holder"; tail -3 $G/gpu_holder2.log | cut -c1-200
bash $G/watch.sh
