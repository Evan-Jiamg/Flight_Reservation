# Incremental GPU grabber (user 2026-09-30 23:5x: "fill GPU1 now so nobody else squeezes in; when their job exits we
# take the freed memory at once"). Run with CUDA_VISIBLE_DEVICES=<one GPU>; holds every free GiB above MARGIN in
# 1 GiB blocks, keeps grabbing as memory frees up, never touches other processes. Files in $D: status (GiB held),
# target (optional max GiB; lower it to release down to it), stop (release everything and exit).
import os
import sys
import time

import torch

D = sys.argv[1]
MARGIN = float(os.environ.get("GRAB_MARGIN_GIB", "1"))
GIB = 1 << 30
os.makedirs(D, exist_ok=True)
blocks = []


def rd(name):
    try:
        return open(os.path.join(D, name)).read().strip()
    except OSError:
        return None


def wr(name, value):
    tmp = os.path.join(D, name + ".tmp")
    with open(tmp, "w") as f:
        f.write(str(value))
    os.replace(tmp, os.path.join(D, name))


def log(msg):
    print("%s %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


log("grabber up on visible GPU %s, margin %.1f GiB" % (os.environ.get("CUDA_VISIBLE_DEVICES"), MARGIN))
last = -1
while not os.path.exists(os.path.join(D, "stop")):
    t = rd("target")
    cap = int(float(t)) if t not in (None, "") else 10 ** 6
    if len(blocks) > cap:
        del blocks[cap:]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    else:
        while len(blocks) < cap and torch.cuda.mem_get_info()[0] / GIB - MARGIN >= 1.0:
            try:
                blocks.append(torch.empty(GIB, dtype=torch.uint8, device="cuda"))
            except RuntimeError:
                break
    if len(blocks) != last:
        log("holds %d GiB (free now %.1f GiB)" % (len(blocks), torch.cuda.mem_get_info()[0] / GIB))
        last = len(blocks)
    wr("status", len(blocks))
    time.sleep(1)
blocks.clear()
torch.cuda.synchronize()
torch.cuda.empty_cache()
wr("status", 0)
log("grabber stopped, all released")
