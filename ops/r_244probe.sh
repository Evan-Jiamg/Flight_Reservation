M=/tmp2/mzjiang_usersim; G=$M/grpo_planner; C=$M/code_snapshots/prism_probe_v1
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.py > SHA256SUMS
PY=/home/mzjiang/miniconda3/envs/consistent/bin/python
PYTHONNOUSERSITE=1 $PY -c "import sklearn; print('sklearn', sklearn.__version__)" 2>&1 | tail -1
FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$1==2{print $2}')
echo "GPU2 free $FREE MiB"
BASE=$M/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
PYTHONNOUSERSITE=1 HF_HOME=$M/hf setsid nohup $PY prism_content_probe.py both --base-model $BASE \
  --data $G/prism_pretrain --out $G/prism_probe_v1 --gpu 2 > $G/prism_probe_v1.log 2>&1 < /dev/null &
sleep 90; tail -3 $G/prism_probe_v1.log | cut -c1-200; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
