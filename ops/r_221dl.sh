export HF_HOME=/tmp2/mzjiang_usersim/hf HF_HUB_CACHE=/tmp2/mzjiang_usersim/hf/hub XDG_CACHE_HOME=/tmp2/mzjiang_usersim/cache PIP_CACHE_DIR=/tmp2/mzjiang_usersim/cache/pip
cd /tmp2/mzjiang_usersim
cat > dl_qwen7b.py <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28",
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"])
print("DONE", p)
EOF
PYTHONNOUSERSITE=1 setsid nohup /home/mzjiang/miniconda3/envs/consistent/bin/python dl_qwen7b.py > dl_qwen7b.log 2>&1 < /dev/null &
sleep 30; tail -c 300 dl_qwen7b.log; du -sh /tmp2/mzjiang_usersim/hf 2>/dev/null
