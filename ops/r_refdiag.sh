G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v17
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared CUDA_DEVICE_ORDER=PCI_BUS_ID
export Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
U1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
if [ "$U1" -gt 2000 ]; then echo "GPU1 busy ($U1 MiB), abort"; exit 0; fi
O=$(ls -td $G/runs/smoke_v17_f2_* | head -1)
export U0=$(dirname $(find $O/ckpt -path "*u00000*/adapter/adapter_model.safetensors" | head -1))
echo "U0=$U0"
mkdir -p $G/diag; cd $C
cat > $G/diag/refdiag.py <<'EOF'
import os, sys, torch, inspect
sys.path.insert(0, os.getcwd())
import peft, rl_algos as RA
from task2_env import PlannerLM
print("peft", peft.__version__, "torch", torch.__version__)
torch.cuda.set_device(0)
planner = PlannerLM(os.environ["Q4"], gpu=0, nf4=False, dtype="bfloat16", adapter=None, trainable=False)
model = RA.setup_policy(planner, 12345)
lr = RA.TorchLearner(model, "grpo", {}, lr=1e-5, seed=0)
u0 = os.environ["U0"]
lr.load_policy(os.path.dirname(u0))
lr.load_ref(u0)
dt = {}
for n, p in model.named_parameters():
    if ".ref." in n or ".default." in n:
        k = "ref" if ".ref." in n else "default"
        dt.setdefault(k, set()).add(str(p.dtype))
print("dtypes", {k: sorted(v) for k, v in dt.items()})
P = dict(model.named_parameters())
mx = 0.0; nz = 0; cnt = 0
for n, p in P.items():
    if ".default." in n:
        r = P.get(n.replace(".default.", ".ref."))
        if r is None:
            continue
        cnt += 1
        d = float((p.detach().float() - r.detach().float()).abs().max())
        mx = max(mx, d)
        nz += int(float(r.detach().float().abs().max()) > 0)
print("paired tensors", cnt, "max |default-ref| weight", mx, "ref tensors nonzero", nz)
tok = planner.tok if hasattr(planner, "tok") else planner.tokenizer
text = "The user wants to plan a trip to Kyoto in spring and asks about cherry blossom timing, hotels near Gion, and a day trip to Nara. " * 30
ids = tok(text)["input_ids"]
s = {"prompt_ids": ids[:300], "gen_ids": ids[300:420], "temperature": 1.0}
RA._set_mode(model, "train_nodropout")
a1 = lr._logp(dict(s), grad=False)[0].float().cpu()
a2 = lr._logp(dict(s), grad=False)[0].float().cpu()
r1 = lr._logp(dict(s), grad=False, reference=True)[0].float().cpu()
model.eval()
e1 = lr._logp(dict(s), grad=False)[0].float().cpu()
er = lr._logp(dict(s), grad=False, reference=True)[0].float().cpu()
def f(x, y): return "max %.3g mean %.3g" % (float((x - y).abs().max()), float((x - y).abs().mean()))
print("train: default vs default", f(a1, a2))
print("train: default vs ref    ", f(a1, r1))
print("eval : default vs ref    ", f(e1, er))
print("train vs eval default    ", f(a1, e1))
for n, p in model.named_parameters():
    if ".ref." in n:
        p.data = p.data.float()
RA._set_mode(model, "train_nodropout")
r2 = lr._logp(dict(s), grad=False, reference=True)[0].float().cpu()
print("train: default vs ref(fp32 cast)", f(a1, r2))
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
sd = load_file(os.path.join(u0, "adapter_model.safetensors"))
print("file dtypes", sorted({str(v.dtype) for v in sd.values()}))
res = set_peft_model_state_dict(model, sd, adapter_name="ref")
P = dict(model.named_parameters())
mx = max(float((P[n].detach() - P[n.replace(".default.", ".ref.")].detach()).abs().max()) for n in P if ".default." in n)
print("after exact reload: max |default-ref| weight", mx)
r3 = lr._logp(dict(s), grad=False, reference=True)[0].float().cpu()
print("train: default vs ref(exact reload)", f(a1, r3))
EOF
CUDA_VISIBLE_DEVICES=1 timeout 900 $PY -c "import sys; sys.path.insert(0, '$C'); exec(open('$G/diag/refdiag.py').read())" 2>&1 | grep -vE "Loading weights|it/s\]|Warning|warn" | tail -30
