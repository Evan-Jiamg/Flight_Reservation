G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v16; OUT=$G/runs/v16_prelim
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
echo "$PAYLOAD_B64" | base64 -d > /tmp2/mzjiang_usersim/v16s0.tgz && X=/tmp2/mzjiang_usersim/v16s0 && mkdir -p $X && rm -rf $X/* && tar xzf /tmp2/mzjiang_usersim/v16s0.tgz -C $X
chmod u+w $C/step0_coverage.py $C/local_sha_v16.txt && cp -f $X/code/* $C/
cd $C && sha256sum -c --quiet local_sha_v16.txt && echo "pend_v16 sha OK" || { echo "SHA MISMATCH"; exit 1; }
curl -s -m 5 http://127.0.0.1:8029/v1/models | grep -q gpt-oss-120b || { echo "gpt-oss not up"; exit 1; }
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
nice -n 10 timeout 1h $PY step0_coverage.py --fold 2 --splits $G/splits_v1.json --out $OUT/step0_f2.jsonl --workers 2 > $OUT/step0.log 2>&1
echo "step0 rc=$?"
grep -v "^\s*$" $OUT/step0.log | grep -vE "^  [0-9a-f]{10} n=" | head -80 | cut -c1-200
