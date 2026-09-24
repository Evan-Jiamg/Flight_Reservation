G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/analysis_v4
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.py > SHA256SUMS && chmod -w *.py
PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python; export PYTHONNOUSERSITE=1
for t in test_task2_episode.py test_derive_gate_arms.py test_analyzers.py; do $PY $t 2>&1 | tail -1 | cut -c1-70; done
$PY -m py_compile stageC_analysis.py && grep -c "rep%d_shard" stageC_analysis.py && echo "analysis_v4 staged at $C"
