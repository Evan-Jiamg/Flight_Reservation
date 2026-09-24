M=/tmp2/mzjiang_usersim; mkdir -p $M/hf $M/grpo_planner $M/code_snapshots
export HF_HOME=$M/hf HF_HUB_CACHE=$M/hf/hub XDG_CACHE_HOME=$M/cache
cd $M
cat > dl_qwen7b.py <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28",
                      allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("DONE", p)
EOF
PYTHONNOUSERSITE=1 setsid nohup /home/mzjiang/miniconda3/envs/consistent/bin/python dl_qwen7b.py > dl_qwen7b.log 2>&1 < /dev/null &
PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/consistent/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])" 2>&1 | tail -1
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
sleep 20; tail -c 200 dl_qwen7b.log
