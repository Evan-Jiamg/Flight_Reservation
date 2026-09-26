V='import torch,transformers
import peft, bitsandbytes as b
print("torch",torch.__version__,"| tf",transformers.__version__,"| peft",peft.__version__,"| bnb",b.__version__,"| cuda",torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
for e in consistent blackwell-kto-test consistent-test; do echo "== $e"; PYTHONNOUSERSITE=1 /home/mzjiang/miniconda3/envs/$e/bin/python -c "$V" 2>&1 | grep -v -i warn | tail -1; done
mkdir -p /tmp2/mzjiang_usersim/hf && ls /tmp2/mzjiang_usersim
timeout 20 curl -sI https://huggingface.co | head -1
