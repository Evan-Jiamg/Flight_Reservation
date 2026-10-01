G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v18; D=$G/.deploy_v18b
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
if pgrep -u mzjiang -f "run_v1[78]_|train_planner_rl.py|eval_test_rl.py|label_acts.py|train_reranker.py" > /dev/null; then echo "STOP: a job of ours is running"; exit 1; fi
[ -e ${C}_r1 ] && { echo "STOP: ${C}_r1 exists"; exit 1; }
rm -rf $D; mkdir -p $D && tar xzf /tmp2/mzjiang_usersim/deploy18b.tgz -C $D || { echo "EXTRACT FAILED"; exit 1; }
mkdir -p $D/snap && tar xzf $D/pend_v18.tgz -C $D/snap && (cd $D/snap && sha256sum -c --quiet local_sha_v18.txt) || { echo "NEW SNAPSHOT SHA FAIL"; exit 1; }
mv $C ${C}_r1 && mv $D/snap $C && (cd $C && sha256sum -c --quiet local_sha_v18.txt && echo "SNAPSHOT v18 SHA OK ($(wc -l < local_sha_v18.txt) files)") || exit 1
for f in run_v18_label.sh run_v18_reranker.sh run_v18_smoke.sh run_v18_fold.sh run_v18_test.sh run_v18_bench.sh run_v18_chain.sh start_servers6.sh; do
  cp $D/$f $G/$f && chmod +x $G/$f && bash -n $G/$f || { echo "SYNTAX/COPY FAIL $f"; exit 1; }
  grep -q $'\r' $G/$f && { echo "CRLF $f"; exit 1; }
done
grep -c "foreign()" $G/start_servers6.sh; grep -c "ours_oss" $G/run_v18_label.sh
rm -rf $G/labels_v18
export HOLD_DIR=$G/hold3
setsid nohup bash $G/run_v18_label.sh 2 > $G/run_v18_label.log 2>&1 < /dev/null &
sleep 90; tail -12 $G/run_v18_label.log; tail -4 $G/gpu_holder3.log; tail -3 $G/servers_check_label_v18.log 2>/dev/null
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
