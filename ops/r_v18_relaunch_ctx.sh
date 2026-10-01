G=/tmp2/mzjiang_usersim/grpo_planner
echo "$PAYLOAD_B64" | base64 -d | tar xz -C $G/.deploy_ctx 2>/dev/null || { mkdir -p $G/.deploy_ctx && echo "$PAYLOAD_B64" | base64 -d | tar xz -C $G/.deploy_ctx; }
if pgrep -u mzjiang -f "run_v18_|train_planner_rl.py|eval_test_rl.py|v18_pipeline.sh" > /dev/null; then echo "STOP: a v18 job is running"; exit 1; fi
mkdir -p $G/v18_scripts_pre_ctx && cp -p $G/start_servers6.sh $G/run_v18_*.sh $G/v18_scripts_pre_ctx/
for f in start_servers6.sh run_v18_fold.sh run_v18_test.sh run_v18_bench.sh run_v18_smoke.sh run_v18_reranker.sh; do
  cp $G/.deploy_ctx/$f $G/$f && chmod +x $G/$f && bash -n $G/$f || { echo "FAIL $f"; exit 1; }
  grep -q $'\r' $G/$f && { echo "CRLF $f"; exit 1; }
done
grep -n "max-model-len" $G/start_servers6.sh | head -2; grep -c "R0_CONTEXT" $G/run_v18_fold.sh
# restart OUR gpt-oss at the new context (start_servers6 re-plans and relaunches it inside the fold run)
pkill -u mzjiang -f "vllm serve .*--port 8029"
for i in $(seq 1 60); do pgrep -u mzjiang -f "vllm serve .*--port 8029" > /dev/null || break; sleep 2; done
pkill -9 -u mzjiang -f "vllm serve .*--port 8029"
cat > $G/v18_pipeline_b.sh <<'EOF'
G=/tmp2/mzjiang_usersim/grpo_planner
export HOLD_DIR=$G/hold3 CUDA_DEVICE_ORDER=PCI_BUS_ID
log() { echo "$(date +%H:%M:%S) $*"; }
log "relaunch after the R0 context fix: chain (fold resume -> test -> bench)"
bash $G/run_v18_chain.sh > $G/chain_f2_v18b.log 2>&1 || { log "PIPELINE STOP: chain"; tail -8 $G/chain_f2_v18b.log; exit 1; }
log "PIPELINE DONE"
EOF
setsid nohup bash $G/v18_pipeline_b.sh > $G/v18_pipeline_b.log 2>&1 < /dev/null &
sleep 2
sed -i 's#pgrep -u mzjiang -f "v18_pipeline.sh"#pgrep -u mzjiang -f "v18_pipeline_b.sh"#; s#tail -4 \$G/v18_pipeline.log#tail -4 $G/v18_pipeline_b.log#' $G/release_after_v18_pipeline.sh
grep -c "v18_pipeline_b" $G/release_after_v18_pipeline.sh
setsid nohup bash $G/release_after_v18_pipeline.sh > $G/release_after_v18_pipeline.log 2>&1 < /dev/null &
sleep 120; cat $G/v18_pipeline_b.log; tail -6 $G/chain_f2_v18b.log; tail -4 $G/gpu_holder3.log
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
