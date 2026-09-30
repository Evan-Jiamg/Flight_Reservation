G=/tmp2/mzjiang_usersim/grpo_planner
if pgrep -u mzjiang -f "run_v17_|train_planner_rl.py|eval_test_rl.py|smoke_v17.py|gpu_holder3\.py|cutover_holder3" > /dev/null; then echo "STOP: something v17 / holder3 already running"; pgrep -u mzjiang -af "run_v17_|train_planner_rl|gpu_holder3|cutover_holder3" | cut -c1-100; exit 1; fi
mkdir -p $G/v17_scripts_r2 && cp -p $G/run_v17_fold.sh $G/run_v17_test.sh $G/run_v17_smoke.sh $G/v17_scripts_r2/
mkdir -p $G/.deploy_h3 && tar xzf /tmp2/mzjiang_usersim/h3.tgz -C $G/.deploy_h3 || { echo "EXTRACT FAILED"; exit 1; }
for f in gpu_holder3.py start_servers6.sh cutover_holder3.sh run_v17_fold.sh run_v17_test.sh run_v17_smoke.sh; do
  cp $G/.deploy_h3/$f $G/$f || { echo "COPY FAILED $f"; exit 1; }
  case $f in *.sh) chmod +x $G/$f; bash -n $G/$f || { echo "SYNTAX $f"; exit 1; }; grep -c $'\r' $G/$f | grep -q '^0$' || { echo "CRLF $f"; exit 1; };; esac
done
sha256sum $G/gpu_holder3.py $G/start_servers6.sh $G/cutover_holder3.sh $G/run_v17_fold.sh | cut -c1-16
# the failed 23:36 attempt left an empty run dir only (no training started)
ls -A $G/runs/pend_f2_v17 2>/dev/null | head
cd $G && setsid nohup bash $G/cutover_holder3.sh > $G/cutover_holder3.log 2>&1 < /dev/null &
# chain: when the cut-over is done -> fold 2 -> test 2 (HOLD_DIR=hold3); a cut-over ABORT ends the chain (nothing started)
cat > $G/chain_f2_h3.sh <<'EOF'
G=/tmp2/mzjiang_usersim/grpo_planner
until grep -q "cut-over done" $G/cutover_holder3.log 2>/dev/null; do
  grep -q "ABORT" $G/cutover_holder3.log 2>/dev/null && { echo "cut-over aborted -> chain not started $(date)"; exit 1; }
  sleep 5
done
export HOLD_DIR=$G/hold3
echo "chain: cut-over done, starting fold 2 $(date)"
bash $G/run_v17_fold.sh 2 > $G/run_v17_fold2.log 2>&1 && bash $G/run_v17_test.sh 2 > $G/run_v17_test2.log 2>&1
echo "chain end rc=$? $(date)"
EOF
setsid nohup bash $G/chain_f2_h3.sh > $G/chain_f2_h3.log 2>&1 < /dev/null &
sleep 60
echo "--- cutover"; tail -15 $G/cutover_holder3.log
echo "--- holder3"; tail -5 $G/gpu_holder3.log 2>/dev/null; cat $G/hold3/plan 2>/dev/null; echo
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
