G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/stageC_v1
if [ -d $C ]; then echo "snapshot exists, refusing"; exit 1; fi
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf -
cmp stop_prompt.py $G/stop_prompt.py && echo "stop_prompt identical to Stage B"
PY=/home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python
export PYTHONNOUSERSITE=1
for t in test_task2_episode.py test_derive_gate_arms.py test_analyzers.py; do $PY $t 2>&1 | tail -1 | cut -c1-80; done
sha256sum *.py *.sh *.md > SHA256SUMS; chmod -w *.py *.sh
echo "--- Stage B status"; tail -1 $G/run_stageB_v1.log
for f in 0 1 2; do grep -E "^epoch 2 train_nll|complete|FOLD" $G/trec_inner_fold${f}_7b_v1.log | cut -c1-200; done
setsid nohup bash $C/run_stageC_v1.sh > $G/run_stageC_v1.log 2>&1 < /dev/null &
sleep 45; cat $G/run_stageC_v1.log; tail -3 $G/stageC_v1/smoke_crn.log 2>/dev/null | cut -c1-300
