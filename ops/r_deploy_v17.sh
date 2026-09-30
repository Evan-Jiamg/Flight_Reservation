G=/tmp2/mzjiang_usersim/grpo_planner; D=$G/.deploy_v17; C=$G/code_snapshots/pend_v17
ls -la $D || { echo "NO DEPLOY DIR"; exit 1; }
[ -e $C ] && { echo "STOP: $C already exists - not overwriting"; exit 1; }
mkdir -p $C && tar xzf $D/pend_v17.tgz -C $C || { echo "EXTRACT FAILED"; exit 1; }
cd $C && sha256sum -c --quiet local_sha_v17.txt && echo "SNAPSHOT SHA OK ($(wc -l < local_sha_v17.txt) files)" || { echo "SNAPSHOT SHA MISMATCH"; exit 1; }
for f in run_v17_fold.sh run_v17_test.sh run_v17_smoke.sh; do
  [ -e $G/$f ] && echo "note: $G/$f exists - replacing"
  cp $D/$f $G/$f && chmod +x $G/$f
  file $G/$f | grep -q CRLF && echo "WARNING: $f has CRLF"
  bash -n $G/$f && echo "$f syntax ok"
done
for f in gpu_holder2.py start_servers5.sh splits_v1.json; do [ -e $G/$f ] && echo "have $f" || echo "MISSING $f"; done
ls $G/bench_pin/ca13b33 >/dev/null 2>&1 && echo "have bench_pin" || echo "MISSING bench_pin"
ls -d $G/runs/pend_f2_v17 2>/dev/null && echo "WARNING: runs/pend_f2_v17 exists" || echo "runs/pend_f2_v17 not present (fresh)"
/home/mzjiang/miniconda3/envs/consistent-test/bin/python -c "import peft, torch, transformers; print('peft', peft.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
