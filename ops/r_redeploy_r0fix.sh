G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v17; RUN=$G/runs/pend_f2_v17
if pgrep -u mzjiang -f "run_v17_|train_planner_rl.py|eval_test_rl.py|smoke_v17.py|chain_f2" > /dev/null; then echo "STOP: v17 process running"; exit 1; fi
[ -e ${C}_r2 ] && { echo "STOP: ${C}_r2 exists"; exit 1; }
[ -e ${RUN}_abort1 ] && { echo "STOP: ${RUN}_abort1 exists"; exit 1; }
mkdir -p $G/.deploy_r0fix && rm -rf $G/.deploy_r0fix/* && tar xzf /tmp2/mzjiang_usersim/r0fix.tgz -C $G/.deploy_r0fix || { echo "EXTRACT FAILED"; exit 1; }
mkdir -p $G/.deploy_r0fix/snap && tar xzf $G/.deploy_r0fix/pend_v17.tgz -C $G/.deploy_r0fix/snap && (cd $G/.deploy_r0fix/snap && sha256sum -c --quiet local_sha_v17.txt) || { echo "NEW SNAPSHOT SHA FAIL"; exit 1; }
mv $C ${C}_r2 && mv $G/.deploy_r0fix/snap $C && echo "snapshot replaced (old -> pend_v17_r2)"
(cd $C && sha256sum -c --quiet local_sha_v17.txt && echo "SNAPSHOT SHA OK ($(wc -l < local_sha_v17.txt) files)") || exit 1
grep -c "R0_EMPTY_REDRAWS" $C/task2_env.py
mv $RUN ${RUN}_abort1 && echo "old run -> pend_f2_v17_abort1"
cat $G/hold3/plan; echo
curl -s -m 5 http://127.0.0.1:8029/v1/models | head -c 60; echo; curl -s -m 5 http://127.0.0.1:8031/v1/models | head -c 60; echo
cat > $G/chain_f2_h3b.sh <<'EOF'
G=/tmp2/mzjiang_usersim/grpo_planner
export HOLD_DIR=$G/hold3
echo "chain b: starting fold 2 $(date)"
bash $G/run_v17_fold.sh 2 > $G/run_v17_fold2.log 2>&1 && bash $G/run_v17_test.sh 2 > $G/run_v17_test2.log 2>&1
echo "chain end rc=$? $(date)"
EOF
setsid nohup bash $G/chain_f2_h3b.sh > $G/chain_f2_h3b.log 2>&1 < /dev/null &
sleep 60
tail -5 $G/run_v17_fold2.log; tail -3 $G/runs/pend_f2_v17/train.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
