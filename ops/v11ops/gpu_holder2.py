# GPU placeholder on cfda5, ALL-AT-ONCE (user 2026-09-26): nothing is held until a GPU's free memory covers a whole
# requirement; then that requirement is taken in one go. Never touches other processes; never holds a partial amount.
#
# Roles (decided here, one GPU each):  server  = the first GPU with free >= SERVER_NEED + MARGIN  (gpt-oss + Planner vLLM)
#                                     train   = another GPU with free >= TRAIN_NEED + MARGIN     (learner + Ditto)
# A train GPU that later has room for the whole server requirement becomes the server GPU (servers are needed first).
# Files in $H:  role_server, role_train (GPU index)   status_<g> (GiB held)   target_<g> (GiB wanted; scripts lower
#               it to hand memory over and raise it to take it back -- a raise is again taken all at once)   stop
import os
import sys
import time

import torch

H = sys.argv[1]
SERVER_NEED = int(os.environ.get("SERVER_NEED_GIB", "91"))
TRAIN_NEED = int(os.environ.get("TRAIN_NEED_GIB", "45"))
MARGIN = float(os.environ.get("HOLD_MARGIN_GIB", "1"))
GIB = 1 << 30
os.makedirs(H, exist_ok=True)
n = torch.cuda.device_count()
blocks = {g: [] for g in range(n)}


def rd(name, default=None):
    try:
        return open(os.path.join(H, name)).read().strip()
    except OSError:
        return default


def wr(name, value):
    tmp = os.path.join(H, name + ".tmp")
    with open(tmp, "w") as f:
        f.write(str(value))
    os.replace(tmp, os.path.join(H, name))


def free_gib(g):
    return torch.cuda.mem_get_info(g)[0] / GIB


def grab(g, amount):
    """Take `amount` GiB on GPU g in one go, or nothing at all."""
    if amount <= 0:
        return True
    if free_gib(g) - MARGIN < amount:
        return False
    new = []
    try:
        for _ in range(amount):
            new.append(torch.empty(GIB, dtype=torch.uint8, device="cuda:%d" % g))
    except RuntimeError:                       # someone took part of it meanwhile: give everything back
        new.clear()
        with torch.cuda.device(g):
            torch.cuda.empty_cache()
        return False
    blocks[g].extend(new)
    return True


def shrink(g, to):
    if len(blocks[g]) > to:
        del blocks[g][to:]
        torch.cuda.synchronize(g)
        with torch.cuda.device(g):
            torch.cuda.empty_cache()


def log(msg):
    print("%s %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


log("holder up: %d GPUs, all-at-once, server %d GiB, train %d GiB, margin %.1f" % (n, SERVER_NEED, TRAIN_NEED, MARGIN))
while not os.path.exists(os.path.join(H, "stop")):
    rs, rt = rd("role_server"), rd("role_train")
    # --- role assignment (only while a role is open)
    if rs is None:
        if rt is not None:
            g = int(rt)                        # the train GPU grew enough for the servers: promote it
            if grab(g, SERVER_NEED - len(blocks[g])):
                wr("role_server", g); wr("target_%d" % g, SERVER_NEED); os.remove(os.path.join(H, "role_train"))
                log("g%d promoted to SERVER GPU, holds %d GiB" % (g, len(blocks[g])))
                continue
        for g in sorted(range(n), key=free_gib, reverse=True):
            if rt is not None and g == int(rt):
                continue
            if grab(g, SERVER_NEED - len(blocks[g])):
                wr("role_server", g); wr("target_%d" % g, SERVER_NEED)
                log("g%d is the SERVER GPU, holds %d GiB" % (g, len(blocks[g])))
                break
    rs, rt = rd("role_server"), rd("role_train")
    if rt is None:
        for g in sorted(range(n), key=free_gib, reverse=True):
            if rs is not None and g == int(rs):
                continue
            if grab(g, TRAIN_NEED - len(blocks[g])):
                wr("role_train", g); wr("target_%d" % g, TRAIN_NEED)
                log("g%d is the TRAIN GPU, holds %d GiB" % (g, len(blocks[g])))
                break
    # --- targets set by the pipeline (hand-over down, take back up -- again all at once)
    for g in range(n):
        t = rd("target_%d" % g)
        if t is None:
            continue
        t = int(float(t))
        if len(blocks[g]) > t:
            shrink(g, t)
            log("g%d handed over, now holds %d GiB" % (g, len(blocks[g])))
        elif len(blocks[g]) < t and grab(g, t - len(blocks[g])):
            log("g%d taken back, holds %d GiB" % (g, len(blocks[g])))
    for g in range(n):
        wr("status_%d" % g, len(blocks[g]))
    time.sleep(1)
for g in range(n):
    shrink(g, 0)
    wr("status_%d" % g, 0)
log("holder stopped, all released")
