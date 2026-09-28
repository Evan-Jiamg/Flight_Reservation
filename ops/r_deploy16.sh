G=/tmp2/mzjiang_usersim/grpo_planner; S=$G/code_snapshots
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
T=/tmp2/mzjiang_usersim/v16deploy.tgz
[ -f $T ] || { echo "payload missing"; exit 1; }
[ -d $S/pend_v16 ] && { echo "pend_v16 already exists"; exit 1; }
# deploy only: snapshot + sha + CPU tests + scripts in place; NO experiment is started here
cp -a $S/pend_v15 $S/pend_v16 && chmod -R u+w $S/pend_v16
mkdir -p /tmp2/mzjiang_usersim/v16x && rm -rf /tmp2/mzjiang_usersim/v16x/* && tar xzf $T -C /tmp2/mzjiang_usersim/v16x
cmp -s /tmp2/mzjiang_usersim/v16x/code/goal_judge.py $S/pend_v15/goal_judge.py && echo "goal_judge.py identical to pend_v15" || echo "NOTE: goal_judge.py differs from pend_v15"
cp -f /tmp2/mzjiang_usersim/v16x/code/* $S/pend_v16/
cd $S/pend_v16 && sha256sum -c --quiet local_sha_v16.txt && echo "pend_v16 sha OK ($(wc -l < local_sha_v16.txt) files)" || { echo "SHA MISMATCH"; exit 1; }
CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 nice -n 19 $PY -m pytest -q -p no:cacheprovider test_v16.py test_v16b.py test_v16c.py \
    test_rl_advantages.py test_pend.py test_verify_pipeline.py test_intervention.py test_reselect.py test_r0_attribution.py \
    test_llm4_controller.py test_rl_controllers.py test_audit4.py 2>&1 | tail -1
cp -f /tmp2/mzjiang_usersim/v16x/ops/*.sh $G/ && chmod +x $G/run_v16_prelim.sh $G/run_v16_formal.sh $G/run_v16_guard.sh $G/run_v16_launch.sh
ls -la $G/run_v16_*.sh | awk '{print $5, $9}'
echo "GPU:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
pgrep -u mzjiang -af "vllm serve|train_planner_rl|gpu_holder2" | cut -c1-100; echo "(our processes above; empty = none)"
