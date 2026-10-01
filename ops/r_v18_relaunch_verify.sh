G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v18; D=$G/.deploy_v18c
if pgrep -u mzjiang -f "run_v18_|train_planner_rl.py|eval_test_rl.py|v18_pipeline" > /dev/null; then echo "STOP: a v18 job is running"; exit 1; fi
[ -e ${C}_r2 ] && { echo "STOP: ${C}_r2 exists"; exit 1; }
rm -rf $D; mkdir -p $D && tar xzf /tmp2/mzjiang_usersim/deploy18c.tgz -C $D || { echo "EXTRACT FAILED"; exit 1; }
mkdir -p $D/snap && tar xzf $D/pend_v18.tgz -C $D/snap && (cd $D/snap && sha256sum -c --quiet local_sha_v18.txt) || { echo "NEW SNAPSHOT SHA FAIL"; exit 1; }
mv $C ${C}_r2 && mv $D/snap $C && (cd $C && sha256sum -c --quiet local_sha_v18.txt && echo "SNAPSHOT v18 SHA OK ($(wc -l < local_sha_v18.txt) files)") || exit 1
for f in run_v18_test.sh run_v18_chain.sh; do cp $D/$f $G/$f && chmod +x $G/$f && bash -n $G/$f || { echo "FAIL $f"; exit 1; }; done
grep -c "ALLOW_CODE_CHANGE" $G/run_v18_test.sh; grep -c "max-model-len \${OSS_MAX_MODEL_LEN" $G/start_servers6.sh
cat > $G/v18_pipeline_c.sh <<'EOF'
G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold3
export HOLD_DIR=$H CUDA_DEVICE_ORDER=PCI_BUS_ID ALLOW_CODE_CHANGE=1
log() { echo "$(date +%H:%M:%S) $*"; }
if ! pgrep -u mzjiang -f "python.* .*gpu_holder3\.py" > /dev/null; then
  rm -rf $H; mkdir -p $H
  CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 setsid nohup /home/mzjiang/miniconda3/envs/consistent-test/bin/python $G/gpu_holder3.py $H >> $G/gpu_holder3.log 2>&1 < /dev/null &
  sleep 10; log "placeholder started by the pipeline"
fi
log "verify (fold, training finished) -> test (ALLOW_CODE_CHANGE=1) -> bench"
bash $G/run_v18_chain.sh > $G/chain_f2_v18c.log 2>&1 || { log "PIPELINE STOP: chain"; tail -8 $G/chain_f2_v18c.log; exit 1; }
log "PIPELINE DONE"
EOF
setsid nohup bash $G/v18_pipeline_c.sh > $G/v18_pipeline_c.log 2>&1 < /dev/null &
sleep 2
sed -i 's#v18_pipeline_b.sh#v18_pipeline_c.sh#g; s#v18_pipeline_b.log#v18_pipeline_c.log#g' $G/release_after_v18_pipeline.sh
grep -c "v18_pipeline_c" $G/release_after_v18_pipeline.sh
pgrep -u mzjiang -f "release_after_v18_pipeline" > /dev/null || setsid nohup bash $G/release_after_v18_pipeline.sh > $G/release_after_v18_pipeline.log 2>&1 < /dev/null &
sleep 90; cat $G/v18_pipeline_c.log; grep -vE "Loading|it/s\]" $G/chain_f2_v18c.log | tail -15 | cut -c1-200; tail -3 $G/gpu_holder3.log
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
