G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v16
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
cd $G && for f in run_v16_fold.sh run_v16_folds01.sh; do [ -e $f ] && echo "EXISTS $f (overwriting)"; done
echo "$PAYLOAD_B64" | base64 -d | tar xzf - -C $G && chmod +x run_v16_fold.sh run_v16_folds01.sh
sha256sum run_v16_fold.sh run_v16_folds01.sh reselect_boot.py | cut -c1-16
ls -d runs/pend_f0_v16 runs/pend_f1_v16 2>&1 | cut -c1-80
cd $C && PYTHONNOUSERSITE=1 $PY -c "
import train_planner_rl as T
for f in (0,1,2):
    s=T.load_split('$G/splits_v1.json', f)
    print('fold', f, {k: len(v) for k, v in s.items() if isinstance(v,(list,set))})
" 2>&1 | grep -v Warning | tail -5
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
pgrep -u mzjiang -af "train_planner_rl|vllm serve|gpu_holder2|run_v16" | grep -v pgrep | cut -c1-80; echo "(no our processes above = idle)"
