G=/tmp2/mzjiang_usersim/grpo_planner
cd $G && echo "$PAYLOAD_B64" | base64 -d | tar xzf - -C $G && chmod +x run_v16_fold.sh run_v16_folds01.sh
sha256sum run_v16_fold.sh run_v16_folds01.sh | cut -c1-16
ls gpu_holder2.py start_servers5.sh reselect_boot.py splits_v1.json || { echo "MISSING FILE - not launching"; exit 1; }
setsid nohup bash $G/run_v16_folds01.sh 1 > $G/run_v16_folds.log 2>&1 < /dev/null &
sleep 40
echo "=== sequencer log"; cat $G/run_v16_folds.log
echo "=== fold 1 log"; tail -5 $G/run_v16_fold1.log 2>/dev/null
pgrep -u mzjiang -af "run_v16|gpu_holder2|vllm serve|train_planner_rl" | grep -v pgrep | cut -c1-90
cat $G/hold/role_* 2>/dev/null | tr '\n' ' '; echo; ls $G/hold 2>/dev/null | tr '\n' ' '; echo
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
