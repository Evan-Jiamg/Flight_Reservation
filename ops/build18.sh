#!/bin/bash
# SPEC v18 code snapshot (code_snapshots/pend_v18): (1) regenerate ops/local_sha_v18.txt = the sha256 of the WORKING-COPY
# bytes of the v17 snapshot's files + the v18 files (RESUME.md: a snapshot is a tar of the working copy, never a fresh
# checkout elsewhere); (2) tar them, unpack, check the list (sha256sum -c), and pack the deploy bundle with the v18 run
# scripts (ops/v11ops/run_v18_*.sh). Local only: nothing is uploaded here.
#   bash ops/build18.sh            -> ops/dep18/deploy18.tgz (+ ops/local_sha_v18.txt)
#   SHA_ONLY=1 bash ops/build18.sh -> only ops/local_sha_v18.txt
set -e
W=$(cd "$(dirname "$0")/.." && pwd)
T=$W/ops/dep18
V18_FILES="v18_rules.py verify_v18.py label_acts.py train_reranker.py smoke_v18.py test_v18.py v18_fixtures.py
bench_tf_generate.py test_bench_tf_generate.py bench_score_f2_v18.sh bench_boot_v18.py"
cd $W/sep-sim
{ sed 's/^[0-9a-f]* \*//' $W/ops/local_sha_v17.txt | tr -d '\r'; for f in $V18_FILES; do echo $f; done; } | awk '!seen[$0]++' > $W/ops/local_sha_v18.list
for f in $(cat $W/ops/local_sha_v18.list); do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
sha256sum -b $(cat $W/ops/local_sha_v18.list) > $W/ops/local_sha_v18.txt
rm -f $W/ops/local_sha_v18.list
echo "local_sha_v18.txt: $(wc -l < $W/ops/local_sha_v18.txt) files"
[ "${SHA_ONLY:-0}" = "1" ] && exit 0
rm -rf $T; mkdir -p $T/snap $T/outer
sed 's/^[0-9a-f]* \*//' $W/ops/local_sha_v18.txt | tr -d '\r' > $T/list.txt
tar cf $T/snap.tar -T $T/list.txt
tar xf $T/snap.tar -C $T/snap
cp $W/ops/local_sha_v18.txt $T/snap/
(cd $T/snap && sha256sum -c --quiet local_sha_v18.txt && echo "LOCAL SHA OK")
tar czf $T/outer/pend_v18.tgz -C $T/snap .
cp $W/ops/v11ops/run_v18_label.sh $W/ops/v11ops/run_v18_reranker.sh $W/ops/v11ops/run_v18_smoke.sh \
   $W/ops/v11ops/run_v18_fold.sh $W/ops/v11ops/run_v18_test.sh $W/ops/v11ops/run_v18_bench.sh \
   $W/ops/v11ops/run_v18_chain.sh $W/ops/v11ops/start_servers6.sh $T/outer/   # start_servers6.sh: + OSS_ONLY (audit A)
tar czf $T/deploy18.tgz -C $T/outer .
ls -la $T/deploy18.tgz
file $T/outer/*.sh
