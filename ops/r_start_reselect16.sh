G=/tmp2/mzjiang_usersim/grpo_planner
X=/tmp2/mzjiang_usersim/v16rs; mkdir -p $X && rm -rf $X/* && echo "$PAYLOAD_B64" | base64 -d | tar xz -C $X || { echo "payload error"; exit 1; }
bash -n $X/run_v16_reselect.sh || { echo "script error"; exit 1; }
pgrep -u mzjiang -af "run_v16_reselect|train_planner_rl|gpu_holder2|vllm serve" | grep -v pgrep && { echo "our processes still running - not starting"; exit 1; }
cp -f $X/run_v16_reselect.sh $X/reselect_boot.py $G/ && chmod +x $G/run_v16_reselect.sh
( cd $G/code_snapshots/pend_v16 && sha256sum -c --quiet local_sha_v16.txt ) && echo "pend_v16 sha OK" || { echo "snapshot sha mismatch - not starting"; exit 1; }
setsid nohup bash $G/run_v16_reselect.sh > $G/run_v16_reselect.log 2>&1 < /dev/null &
sleep 40
echo "--- reselect log"; cat $G/run_v16_reselect.log | cut -c1-200
echo "--- servers"; tail -3 $G/servers_check.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "roles srv=$(cat $G/hold/role_server 2>/dev/null) trn=$(cat $G/hold/role_train 2>/dev/null)"
