#!/bin/bash
# SPEC v18 fold-2 chain after the user approved the labels and the smoke passed: training (run_v18_fold.sh) -> test
# evaluation (run_v18_test.sh, the servers kept) -> benchmark suite (run_v18_bench.sh, releases every GPU at the end).
# Each step must succeed before the next starts; a STOP of any step stops the chain (its log says why). The steps before
# this chain, in order: run_v18_label.sh (labels + the review sample) -> the USER writes the .APPROVED marker ->
# run_v18_reranker.sh (candidates, fit, gate) -> run_v18_smoke.sh.
# Usage: setsid nohup bash run_v18_chain.sh > /tmp2/mzjiang_usersim/grpo_planner/chain_f2_v18.log 2>&1 < /dev/null &
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner
D=$(cd "$(dirname "$0")" && pwd)
echo "=== CHAIN v18 start $(date)"
bash $D/run_v18_fold.sh 2 || { echo "CHAIN STOP: training $(date)"; exit 1; }
# ALLOW_CODE_CHANGE (env) is passed through to the test step
KEEP_SERVERS=1 bash $D/run_v18_test.sh 2 || { echo "CHAIN STOP: test $(date)"; exit 1; }
bash $D/run_v18_bench.sh 2 || { echo "CHAIN STOP: benchmark $(date)"; exit 1; }
echo "=== CHAIN v18 DONE $(date)"
