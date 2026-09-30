"""End-to-end simulation of cutover_holder3.sh: pgrep / nvidia-smi / kill / date / stat / sleep stubbed in bash; every
`sleep` advances a simulated world by one tick (old holders obey their targets; holder3 = gpu_holder3.HolderState over
the real plan logic, reading hold3/extra_<g>). Needs a POSIX bash (Git Bash). Run: python -m pytest test_cutover_holder3.py"""
import os
import subprocess
import sys

import pytest

from test_start_servers6 import BASH

HERE = os.path.dirname(os.path.abspath(__file__))
pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")

SIM = r'''
import os, pickle, sys
sys.path.insert(0, %(here)r)
import gpu_holder3 as GH
G = sys.argv[1]; H3, OLD, GRAB = G + "/hold3", G + "/hold", G + "/grab1"
TOTAL = 97887 / 1024.0


def rd(p, d=0):
    try:
        return int(float(open(p).read().strip()))
    except (OSError, ValueError):
        return d


def wr(p, v):
    open(p, "w").write(str(v))


stopped = os.path.exists(OLD + "/stop")
before = rd(OLD + "/status_1") + rd(GRAB + "/status")
LAG = os.environ.get("SIM_LAG") == "1"


def obey(status_f, target_f, stop):
    """An old holder shrinks to its target; with SIM_LAG its status shows the target of the PREVIOUS tick."""
    t = rd(target_f, 10 ** 6)
    if LAG:
        t, _ = rd(target_f + ".prev", t), wr(target_f + ".prev", t)
    wr(status_f, 0 if stop else min(rd(status_f), t))


for g in (0, 1):
    obey(OLD + "/status_%%d" %% g, OLD + "/target_%%d" %% g, stopped)
obey(GRAB + "/status", GRAB + "/target", os.path.exists(GRAB + "/stop"))
old = {0: rd(OLD + "/status_0"), 1: rd(OLD + "/status_1") + rd(GRAB + "/status")}
others = {0: 30.0, 1: 5.0 + rd(G + "/stolen_1")}
if os.environ.get("SIM_STEAL") == "1" and old[1] < before:          # another user takes what the old holders free
    wr(G + "/stolen_1", rd(G + "/stolen_1") + before - old[1]); others[1] += before - old[1]
sp = H3 + "/sim_state.pkl"
st = pickle.load(open(sp, "rb")) if os.path.exists(sp) else GH.HolderState(2, GH.sizes_from_env({}))
held = {g: rd(H3 + "/status_%%d" %% g) for g in (0, 1)}
free = {g: TOTAL - others[g] - old[g] - held[g] for g in (0, 1)}
extra = None if st.committed else {g: rd(H3 + "/extra_%%d" %% g) for g in (0, 1)}
res = st.tick(held, free, extra=extra)
for g, k in GH.grab_counts(res["targets"], held, free, 1.0).items():
    held[g] += k
for g in (0, 1):
    wr(H3 + "/status_%%d" %% g, held[g]); wr(H3 + "/target_%%d" %% g, res["targets"][g])
if res["commit"]:
    r = res["commit"]
    wr(H3 + "/role_oss", r["oss"]); wr(H3 + "/role_planner", r["planner"]); wr(H3 + "/role_train", r["train"])
pickle.dump(st, open(sp, "wb"))
wr(H3 + "/plan", "sim %%s held %%s old %%s" %% (GH.describe(st.plan) if st.plan else "-", held, old))
'''

STUBS = r'''
CLK=$CUTOVER_G/clock; echo 1000000 > $CLK
date() { if [ "${1:-}" = "+%s" ]; then cat $CLK; else command date "$@"; fi; }
stat() { cat $CLK; }
sleep() { echo $(( $(cat $CLK) + 1 )) > $CLK; "$SIMPY" "$CUTOVER_G/sim.py" "$CUTOVER_G"; }
kill() { return 0; }
pgrep() {
  case "$*" in
    *"gpu_holder2|gpu_grab"*) [ -f $CUTOVER_G/hold/stop ] && return 1; echo 111 ;;
    *gpu_holder2*) echo 111 ;;
    *gpu_grab*) echo 222 ;;
    *gpu_holder3*) echo 333 ;;
    *) return 1 ;;
  esac; }
nvidia-smi() {
  case "$*" in
    *"--query-gpu=index,uuid"*) printf "0, GPU-aaa\n1, GPU-bbb\n" ;;
    *"--query-compute-apps"*) echo "222, ${GRAB_UUID:-GPU-bbb}" ;;
  esac; }
'''


def run_cutover(tmp_path, env_extra=None, roles=True, layout=None):
    g = tmp_path / "G"
    for d in ("hold", "grab1", "hold3"):
        (g / d).mkdir(parents=True)
    files = {"hold/status_0": "0", "hold/target_0": "45", "hold/status_1": "16", "hold/target_1": "16",
             "grab1/status": "26", "hold3/plan": "sim start"}
    files.update(layout or {})
    if roles:
        files.update({"hold/role_server": "1", "hold/role_train": "0"})
    for name, text in files.items():
        (g / name).write_text(text, newline="\n")
    (g / "sim.py").write_text(SIM % {"here": HERE}, newline="\n")
    script = tmp_path / "t.sh"
    script.write_text(STUBS + 'source "$CUT"\n', newline="\n")
    env = dict(os.environ, CUTOVER_G=str(g).replace("\\", "/"), SIMPY=sys.executable.replace("\\", "/"),
               CUT=os.path.join(HERE, "cutover_holder3.sh").replace("\\", "/"))
    env.update(env_extra or {})
    out = subprocess.run([BASH, str(script).replace("\\", "/")], capture_output=True, text=True, env=env, timeout=600)
    rd = lambda n: (g / n).read_text().strip() if (g / n).exists() else None
    return out.returncode, out.stdout, rd


def test_cutover_hands_old_memory_to_the_plan_that_needs_it(tmp_path):
    """Others 30 GiB on GPU 0 / 5 on GPU 1; old holders 16 + 26 = 42 GiB on GPU 1. holder3 must pick P2 with gpt-oss
    on GPU 1 (possible only with the old memory), absorb it step by step, commit, and the old surplus is released."""
    rc, out, rd = run_cutover(tmp_path)
    assert rc == 0, out
    assert "cut-over done" in out and "froze grab1 at 26 GiB" in out
    assert out.index("froze grab1") < out.index("froze hold2 g0")                       # grab1 frozen first
    assert (rd("hold3/role_oss"), rd("hold3/role_planner"), rd("hold3/role_train")) == ("1", "0", "0")
    assert (rd("hold3/status_0"), rd("hold3/status_1")) == ("61", "77")
    assert rd("hold/status_1") == "0" and rd("grab1/status") == "0"
    assert rd("hold/stop") is not None and rd("grab1/stop") is not None
    assert rd("hold3/extra_0") is None and rd("hold3/extra_1") is None
    assert "surplus" in out and "ABORT" not in out and "WARN" not in out


def test_cutover_prechecks_touch_nothing(tmp_path):
    rc, out, rd = run_cutover(tmp_path, roles=False)
    assert rc == 1 and "has no hold/role_server" in out
    assert rd("grab1/target") is None and rd("hold/target_1") == "16"                  # nothing frozen / changed
    rc, out, rd = run_cutover(tmp_path / "b", {"GRAB_UUID": "GPU-aaa"})
    assert rc == 1 and "not only on GPU 1" in out and rd("grab1/target") is None


def test_cutover_step_not_absorbed_freezes_then_aborts(tmp_path):
    rc, out, rd = run_cutover(tmp_path, {"SIM_STEAL": "1"})
    assert rc == 1
    assert "WARN: step of" in out and "not absorbed by holder3 within 120 s" in out
    steps = [l for l in out.splitlines() if " -> " in l and "GiB (holder3" in l]
    assert len(steps) == 1                                                            # frozen after the first step
    assert rd("hold/stop") is None and rd("grab1/stop") is None                       # old holders left running


def test_cutover_live_layout_with_status_lag(tmp_path):
    """The layout on cfda5 (hold2: 45 GiB on GPU 0, 16 on GPU 1; grab1: 32 on GPU 1; others 30 / 5) with the old
    holders' status lagging one tick behind their target: holder3 (P2, gpt-oss on GPU 1) takes 42 + 36 GiB from them,
    the rest is released as surplus after the commit, nothing is double-counted or aborted."""
    rc, out, rd = run_cutover(tmp_path, {"SIM_LAG": "1"},
                              layout={"hold/status_0": "45", "hold/target_0": "45", "grab1/status": "32"})
    assert rc == 0, out
    assert "froze grab1 at 32 GiB" in out and "froze hold2 g0 at 45 GiB" in out
    assert (rd("hold3/role_oss"), rd("hold3/role_planner"), rd("hold3/role_train")) == ("1", "0", "0")
    assert (rd("hold3/status_0"), rd("hold3/status_1")) == ("61", "77")
    assert rd("hold/status_0") == "0" and rd("hold/status_1") == "0" and rd("grab1/status") == "0"
    assert "ABORT" not in out and "WARN" not in out and "surplus" in out
    steps = [l for l in out.splitlines() if " -> " in l and "GiB (holder3" in l]
    assert all(int(l.split(" -> ")[0].split()[-1]) - int(l.split(" -> ")[1].split()[0]) <= 2 for l in steps)
