G=/tmp2/mzjiang_usersim/grpo_planner; D=$G/dev_v3
rm -rf $D && mkdir -p $D && cd $D && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
export PYTHONNOUSERSITE=1
for t in test_task2_episode.py test_planner_prompt_v3.py test_goal_judge.py test_derive_gate_arms.py test_analyzers.py; do echo "== $t"; $PY $t 2>&1 | tail -2; done
setsid nohup bash $D/run_smoke_v3.sh > $G/smoke_v3.log 2>&1 < /dev/null &
sleep 60; cat $G/smoke_v3.log | grep -v -i warn | tail -8
