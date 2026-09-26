G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/ditto_compare_v1
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.py > SHA256SUMS
PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python ditto_compare.py 2>&1 | tail -20
