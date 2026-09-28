D=/tmp2/mzjiang_usersim/grpo_planner/dev_v3
rm -rf $D && mkdir -p $D && cd $D && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
TESTS="test_task2_episode.py test_planner_prompt_v3.py test_fit_prompts.py"
for t in $TESTS; do echo "=== $t"; $PY $t 2>&1 | grep -v -i -E "warn|pynvml" | tail -12; done
