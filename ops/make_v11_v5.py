"""Generate the v5 scripts (all-at-once placeholder gpu_holder2.py; roles decided by the holder). Run from ops/."""
OLD_TAKE = """take_train_gpu() {   # wait until the holder holds 45 GiB on the training GPU, then hand it over (release to 0)
  TG=$(cat $H/train_gpu); n=0"""
NEW_TAKE = """take_train_gpu() {   # wait for the holder's training GPU (taken all at once, 45 GiB), then hand it over
  n=0; until [ -f $H/role_train ]; do [ $((n % 600)) -eq 0 ] && echo "WAITING_GPU: no training GPU with room yet ($(date +%H:%M))"; n=$((n + 1)); sleep 1; done
  TG=$(cat $H/role_train); n=0"""
for src, dst, extra in (("v11ops/run_v11_formal4.sh", "v11ops/run_v11_formal5.sh", []),
                        ("v11ops/run_v11_continue4.sh", "v11ops/run_v11_continue5.sh",
                         [("run_v11_formal4.sh", "run_v11_formal5.sh"), ("run_v11_formal4.log", "run_v11_formal5.log")])):
    s = open(src, encoding="utf-8").read()
    for a, b in [(OLD_TAKE, NEW_TAKE), ("bash $G/start_servers4.sh", "bash $G/start_servers5.sh"),
                 ('sleep 2; done\n  settarget $TG 0', 'sleep 1; done\n  settarget $TG 0')] + extra:
        assert a in s, (src, a[:60])
        s = s.replace(a, b)
    open(dst, "w", encoding="utf-8", newline="\n").write(s)

launcher = """#!/bin/bash
# v11 launcher (5): the all-at-once GPU placeholder takes a whole requirement the moment a cfda5 GPU has room for it
# (servers 91 GiB, training 45 GiB) and hands it over; gate then formal continuation; placeholder released at the end.
G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
while pgrep -u mzjiang -f "run_v11_launch4.sh|run_v11_formal4.sh|run_v11_continue4.sh|start_servers4.sh|gpu_holder.py" > /dev/null; do sleep 2; done
echo "older chain gone $(date)"
rm -rf $H; mkdir -p $H
PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder2.py $H > $G/gpu_holder2.log 2>&1 < /dev/null &
sleep 15; tail -2 $G/gpu_holder2.log
bash $G/run_v11_formal5.sh > $G/run_v11_formal5.log 2>&1
bash $G/run_v11_continue5.sh > $G/run_v11_continue5.log 2>&1
touch $H/stop
echo "LAUNCHER5 DONE $(date)"
"""
open("v11ops/run_v11_launch5.sh", "w", encoding="utf-8", newline="\n").write(launcher)

w = open("v11ops/watch.sh", encoding="utf-8").read()
for a, b in [
    ("for s in run_v11_launch4 run_v11_formal4 run_v11_continue4 gpu_holder run_v11_launch3 run_v11_formal3; do",
     "for s in run_v11_launch5 run_v11_formal5 run_v11_continue5 gpu_holder2; do"),
    ("stage=$(cat $G/run_v11_launch4.log $G/run_v11_formal4.log $G/run_v11_continue4.log 2>/dev/null",
     "stage=$(cat $G/run_v11_launch5.log $G/run_v11_formal5.log $G/run_v11_continue5.log 2>/dev/null"),
    ("for f in $G/run_v11_launch4.log $G/run_v11_formal4.log $G/run_v11_continue4.log $G/gpu_holder.log; do",
     "for f in $G/run_v11_launch5.log $G/run_v11_formal5.log $G/run_v11_continue5.log $G/gpu_holder2.log; do"),
    ('hold=$(for g in 0 1; do printf "g%s=%s/%s " $g "$(cat $G/hold/status_$g 2>/dev/null)" "$(cat $G/hold/target_$g 2>/dev/null)"; done)',
     'hold="$(for g in 0 1; do printf "g%s=%s/%s " $g "$(cat $G/hold/status_$g 2>/dev/null)" "$(cat $G/hold/target_$g 2>/dev/null)"; done)srv=$(cat $G/hold/role_server 2>/dev/null) trn=$(cat $G/hold/role_train 2>/dev/null) "'),
]:
    assert w.count(a) == 1, ("watch", a[:60])
    w = w.replace(a, b)
open("v11ops/watch.sh", "w", encoding="utf-8", newline="\n").write(w)
print("ok")
