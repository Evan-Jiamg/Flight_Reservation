D=/tmp2/mzjiang_usersim/grpo_planner/dev_v3
rm -rf $D && mkdir -p $D && cd $D && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
for t in test_task2_episode.py test_planner_prompt_v3.py test_goal_judge.py test_derive_gate_arms.py test_analyzers.py test_fit_prompts.py test_judge_tokenization.py; do
  echo "=== $t"; $PY $t 2>&1 | grep -v -i -E "warn|pynvml" | tail -4
done
$PY -m py_compile rollout_ditto_v3.py train_goal_judge.py label_goal_status.py build_judge_samples.py make_crossfit_groups.py && echo "compile ok"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
