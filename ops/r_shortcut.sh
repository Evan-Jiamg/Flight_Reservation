G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/audit_v1
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python
PYTHONNOUSERSITE=1 $PY shortcut_audit.py $G/stageA_eval_v1/prism_val_ep1.jsonl $G/stageA_eval_v1/prism_val_ep2.jsonl \
  $G/stageB_eval_v1/fold0_ep0_innerval.jsonl $G/stageB_eval_v1/fold0_ep2_innerval.jsonl \
  $G/stageB_eval_v1/fold1_ep0_innerval.jsonl $G/stageB_eval_v1/fold1_ep2_innerval.jsonl \
  $G/stageB_eval_v1/fold2_ep0_innerval.jsonl $G/stageB_eval_v1/fold2_ep2_innerval.jsonl
echo "--- which prompt lines carry turn count / rule outputs (TREC example)"
head -1 $G/nested/fold0_inner_train.jsonl | PYTHONNOUSERSITE=1 $PY -c "import json,sys; u=json.loads(sys.stdin.read())['user']; [print('  ', l) for l in u.split('\n') if any(k in l for k in ('turns so far','SENT','unhelpful','stopping condition','gives up','patience','frustrat'))]"
echo "--- PRISM prompt example lines"
head -1 $G/prism_pretrain/validation.jsonl | PYTHONNOUSERSITE=1 $PY -c "import json,sys; u=json.loads(sys.stdin.read())['user']; print(u[:700])"
