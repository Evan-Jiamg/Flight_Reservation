G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/arch_audit_v1
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python architecture_audit_checks.py 2>&1 | tail -70
