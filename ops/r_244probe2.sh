M=/tmp2/mzjiang_usersim; G=$M/grpo_planner; C=$M/code_snapshots/prism_probe_v2
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.py > SHA256SUMS
PY=/home/mzjiang/miniconda3/envs/consistent/bin/python
PYTHONNOUSERSITE=1 $PY test_probe.py 2>&1 | tail -3
cat > $C/run_probe_after_extract.sh <<'EOF'
G=/tmp2/mzjiang_usersim/grpo_planner; C=/tmp2/mzjiang_usersim/code_snapshots/prism_probe_v2
until [ -f $G/prism_probe_v1/validation.npz ]; do sleep 60; done
sleep 30
cd $C && PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/consistent/bin/python prism_content_probe.py probe --out $G/prism_probe_v1 > $G/prism_probe_v1_probe.log 2>&1
echo "probe rc=$? $(date)" >> $G/prism_probe_v1_probe.log
EOF
setsid nohup bash $C/run_probe_after_extract.sh > /dev/null 2>&1 < /dev/null &
tail -2 $G/prism_probe_v1.log | cut -c1-120
