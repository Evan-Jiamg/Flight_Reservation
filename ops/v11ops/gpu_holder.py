# GPU memory placeholder on cfda5 (user request 2026-09-26): grab FREE memory as soon as it appears, so the pipeline's
# GPUs cannot be taken again while it waits or between training chunks. Never touches other processes.
#
# Control (files in $G/hold/):  target_<g>  = GiB to hold on GPU g (the holder grows/shrinks to it)
#                               status_<g>  = GiB currently held (written by the holder)
#                               stop        = free everything and exit
# Grows in 1 GiB blocks while (free - MARGIN) allows; shrinks by freeing blocks + empty_cache.
import os
import sys
import time

import torch

HOLD = sys.argv[1]
MARGIN = float(os.environ.get("HOLD_MARGIN_GIB", "3"))
GIB = 1 << 30
os.makedirs(HOLD, exist_ok=True)
n = torch.cuda.device_count()
blocks = {g: [] for g in range(n)}


def read_target(g, default):
    try:
        return int(float(open(os.path.join(HOLD, "target_%d" % g)).read().strip()))
    except (OSError, ValueError):
        return default


def write_status(g):
    tmp = os.path.join(HOLD, "status_%d.tmp" % g)
    with open(tmp, "w") as f:
        f.write(str(len(blocks[g])))
    os.replace(tmp, os.path.join(HOLD, "status_%d" % g))


for g in range(n):
    if not os.path.exists(os.path.join(HOLD, "target_%d" % g)):
        with open(os.path.join(HOLD, "target_%d" % g), "w") as f:
            f.write("91")
    write_status(g)
print("holder up: %d GPUs, margin %.1f GiB (%s)" % (n, MARGIN, time.strftime("%H:%M:%S")), flush=True)
last = {}
while not os.path.exists(os.path.join(HOLD, "stop")):
    for g in range(n):
        tgt = read_target(g, 91)
        changed = False
        while len(blocks[g]) > tgt:
            blocks[g].pop()
            changed = True
        if changed:
            torch.cuda.synchronize(g)
            with torch.cuda.device(g):
                torch.cuda.empty_cache()
        while len(blocks[g]) < tgt:
            free, _ = torch.cuda.mem_get_info(g)
            if free / GIB - 1.0 < MARGIN:
                break
            try:
                blocks[g].append(torch.empty(GIB, dtype=torch.uint8, device="cuda:%d" % g))
            except RuntimeError:          # someone else took it first: try again next round
                break
        write_status(g)
        if last.get(g) != len(blocks[g]):
            print("%s g%d holds %d GiB (target %d)" % (time.strftime("%H:%M:%S"), g, len(blocks[g]), tgt), flush=True)
            last[g] = len(blocks[g])
    time.sleep(2)
for g in range(n):
    blocks[g].clear()
    with torch.cuda.device(g):
        torch.cuda.empty_cache()
    write_status(g)
print("holder stopped, all released (%s)" % time.strftime("%H:%M:%S"), flush=True)
