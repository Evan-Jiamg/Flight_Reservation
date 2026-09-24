V='import torch,transformers,sys
try:
  import peft; pv=peft.__version__
except Exception as e: pv="ERR "+str(e)[:40]
try:
  import bitsandbytes as b; bv=b.__version__
except Exception as e: bv="ERR "+str(e)[:40]
from transformers import models
q=hasattr(models,"qwen3_vl")
print("torch",torch.__version__,torch.__file__.split("/site-packages")[0][-40:],"| tf",transformers.__version__,"| peft",pv,"| bnb",bv,"| qwen3_vl",q, "| cuda", torch.cuda.is_available())'
echo "== blackwell-kto-test, NOUSERSITE=1"; PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python -c "$V" 2>&1 | grep -v -i warn | tail -1
echo "== blackwell-kto-test, user site ON"; /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python -c "$V" 2>&1 | grep -v -i warn | tail -1
echo "== consistent-test, NOUSERSITE=1"; PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/consistent-test/bin/python -c "$V" 2>&1 | grep -v -i warn | tail -1
head -3 /tmp2/mzjiang_usersim/logs/ditto_swap_full.log | cut -c1-200
