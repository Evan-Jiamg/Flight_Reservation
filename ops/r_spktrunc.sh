G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/trunc_audit_v1
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/consistent-test/bin/python speaker_trunc_audit.py "$G/stageC_v1/rep0_ditto_shard*/nogate.jsonl" 2>&1 | grep -v -i warn | tail -60
