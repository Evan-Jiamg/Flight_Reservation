G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/mask_ablation_v1
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.py > SHA256SUMS
PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python
BASE=/tmp2/hf_shared/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared
O=$G/mask_ablation_v1; mkdir -p $O
for f in 0 1 2; do for m in nomask mask; do
  X=""; [ $m = mask ] && X="--mask-rules"
  $PY train_stop_sft_cox.py --train $G/nested/fold${f}_inner_train.jsonl --validation $G/nested/fold${f}_inner_validation.jsonl \
    --base-model $BASE --init-adapter $G/trec_inner_fold${f}_7b_v1/epoch2 --gpu 1 --compute-dtype bfloat16 \
    --micro-batch 1 --max-length 2048 --epochs 0 --no-offset $X --out $O/fold${f}_$m > $O/fold${f}_$m.log 2>&1
  grep "^epoch 0" $O/fold${f}_$m.log | cut -c1-230 | sed "s/^/fold$f $m: /"
done; done
$PY - <<'EOF'
import json
O="/tmp2/mzjiang_usersim/grpo_planner/mask_ablation_v1"
for f in (0,1,2):
    a=[json.loads(l) for l in open(f"{O}/fold{f}_nomask/epoch0_val_predictions.jsonl")]
    b=[json.loads(l) for l in open(f"{O}/fold{f}_mask/epoch0_val_predictions.jsonl")]
    d=[abs(x["p_stop"]-y["p_stop"]) for x,y in zip(a,b)]
    print(f"fold{f}: mean |dp| {sum(d)/len(d):.4f} max {max(d):.4f}; mean p nomask {sum(x['p_stop'] for x in a)/len(a):.4f} mask {sum(y['p_stop'] for y in b)/len(b):.4f}")
EOF
