G=/tmp2/mzjiang_usersim/grpo_planner; S=$G/code_snapshots/pend_v13; M=/tmp2/MingZhi_HcWang/MingZhi_Code
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
ls -l $M/code/train_planner_rl.py $M/code/rl_controllers.py | cut -c1-60
chmod u+w $M/code/train_planner_rl.py $M/code/rl_controllers.py
cp -p $S/train_planner_rl.py $S/rl_controllers.py $M/code/
cd $M/code && sha256sum -c --quiet local_sha_v13.txt && echo "MingZhi_Code/code = pend_v13: sha OK ($(wc -l < local_sha_v13.txt) files)"
diff -rq $S $M/code 2>&1 | grep -v "__pycache__\|Only in $M/code" | head -5
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=$M/_pycache E1R_TREE=$G/trees/e1r_cf19400 V2FIX_TREE=/home/mzjiang/Sep-Simulator
nice -n 19 $PY -m pytest -q -p no:cacheprovider --ignore=test_batching_server.py --ignore=test_rl_algos_server.py 2>&1 | tail -1
echo "live trainer: $(pgrep -u mzjiang -f 'train_planner_rl.py --fold 2' | head -1)"
