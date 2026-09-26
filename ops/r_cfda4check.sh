echo "host $(hostname)"
echo "--- ssh cfda4 -> cfda5 (BatchMode, no password)"
timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new 140.109.21.245 hostname 2>&1 | tail -1
echo "--- data on cfda4"
for p in /tmp2/hchsu/trec2026-usersim-benchmark /tmp2/mzjiang_usersim/grpo_planner/trees/e1r_cf19400 \
         /tmp2/mzjiang_usersim/grpo_planner/splits_v1.json /home/mzjiang/v5-latency/data.jsonl /tmp2/hf_shared \
         /tmp2/TREC_UserSim_MingZhi/models/princeton-nlp_sup-simcse-bert-base-uncased /home/mzjiang/Sep-Simulator; do
  [ -e $p ] && echo "ok      $p" || echo "MISSING $p"
done
ls /tmp2/mzjiang_usersim/grpo_planner 2>/dev/null | head -20 | tr '\n' ' '; echo
echo "--- python env"
/home/mzjiang/miniconda3/envs/consistent-test/bin/python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'tf', transformers.__version__, 'peft', peft.__version__)" 2>&1 | tail -1
/home/mzjiang/miniconda3/envs/consistent-test/bin/python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -1
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
