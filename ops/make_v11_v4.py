"""Generate the v4 scripts: GPU placeholder hand-over for the servers and the training GPU. Run from ops/."""
HOLD_FUNCS = """H=$G/hold
held() { cat $H/status_$1 2>/dev/null || echo 0; }
settarget() { echo $2 > $H/target_$1.tmp && mv -f $H/target_$1.tmp $H/target_$1; }
take_train_gpu() {   # wait until the holder holds 45 GiB on the training GPU, then hand it over (release to 0)
  TG=$(cat $H/train_gpu); n=0
  until [ "$(held $TG)" -ge 45 ]; do [ $((n % 300)) -eq 0 ] && echo "WAITING_GPU: training GPU $TG holds $(held $TG)/45 GiB ($(date +%H:%M))"; n=$((n + 1)); sleep 2; done
  settarget $TG 0; until [ "$(held $TG)" -le 0 ]; do sleep 1; done
  GPU=$TG
}
"""

s = open("v11ops/run_v11_formal3.sh", encoding="utf-8").read()
reps = [
    ("bash $G/start_servers3.sh", "bash $G/start_servers4.sh"),
    ('echo "=== 3 waiting for a free training GPU (not the servers\' GPU) $(date)"\n'
     'GPU=""; while [ -z "$GPU" ]; do GPU=$(pickgpu); [ -z "$GPU" ] && sleep 60; done\n',
     'echo "=== 3 taking the training GPU from the placeholder $(date)"\ntake_train_gpu\n'),
    ('echo "train rc=$? $(date)";', 'rc=$?; settarget $TG 45; echo "train rc=$rc $(date)";'),
    ("cd $C\n", HOLD_FUNCS + "cd $C\n"),
]
for a, b in reps:
    assert s.count(a) == 1, ("formal", a[:70])
    s = s.replace(a, b)
open("v11ops/run_v11_formal4.sh", "w", encoding="utf-8", newline="\n").write(s)

s = open("v11ops/run_v11_continue3.sh", encoding="utf-8").read()
reps = [
    ("bash $G/start_servers3.sh", "bash $G/start_servers4.sh"),
    ("run_v11_formal3.sh", "run_v11_formal4.sh"),
    ("run_v11_formal3.log", "run_v11_formal4.log"),
    ('  GPU=""; while [ -z "$GPU" ]; do GPU=$(pickgpu); [ -z "$GPU" ] && sleep 60; done\n', '  take_train_gpu\n'),
    ('  rc=$?; echo "train rc=$rc $(date)"', '  rc=$?; settarget $TG 45; echo "train rc=$rc $(date)"'),
    ("cd $C\n", HOLD_FUNCS + "cd $C\n"),
]
for a, b in reps:
    assert s.count(a) >= 1, ("continue", a[:70])
    s = s.replace(a, b)
open("v11ops/run_v11_continue4.sh", "w", encoding="utf-8", newline="\n").write(s)

launcher = """#!/bin/bash
# v11 launcher (4): the GPU placeholder grabs free memory on every cfda5 GPU the moment it appears and hands it over
# to the servers and the trainer; gate then formal continuation; the placeholder is stopped (memory released) at the end.
G=/tmp2/mzjiang_usersim/grpo_planner; H=$G/hold
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
while pgrep -u mzjiang -f "run_v11_launch3.sh|run_v11_formal3.sh|run_v11_continue3.sh|start_servers3.sh" > /dev/null; do sleep 5; done
echo "older chain gone $(date)"
rm -rf $H; mkdir -p $H
PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder.py $H > $G/gpu_holder.log 2>&1 < /dev/null &
sleep 20; tail -2 $G/gpu_holder.log
bash $G/run_v11_formal4.sh > $G/run_v11_formal4.log 2>&1
bash $G/run_v11_continue4.sh > $G/run_v11_continue4.log 2>&1
touch $H/stop
echo "LAUNCHER4 DONE $(date)"
"""
open("v11ops/run_v11_launch4.sh", "w", encoding="utf-8", newline="\n").write(launcher)

w = open("v11ops/watch.sh", encoding="utf-8").read()
reps = [
    ("for s in run_v11_launch3 run_v11_formal3 run_v11_continue3 run_v11_launch run_v11_formal2 run_v11_continue2; do",
     "for s in run_v11_launch4 run_v11_formal4 run_v11_continue4 gpu_holder run_v11_launch3 run_v11_formal3; do"),
    ("stage=$(cat $G/run_v11_launch3.log $G/run_v11_formal3.log $G/run_v11_continue3.log 2>/dev/null",
     "stage=$(cat $G/run_v11_launch4.log $G/run_v11_formal4.log $G/run_v11_continue4.log 2>/dev/null"),
    ("for f in $G/run_v11_launch3.log $G/run_v11_formal3.log $G/run_v11_continue3.log; do",
     "for f in $G/run_v11_launch4.log $G/run_v11_formal4.log $G/run_v11_continue4.log $G/gpu_holder.log; do"),
    ('echo "$(date +%H:%M) running:[${st# }] | ${stage} | ${gpu}| ${srv} | rollouts=${nro} ${upd}"',
     'hold=$(for g in 0 1; do printf "g%s=%s/%s " $g "$(cat $G/hold/status_$g 2>/dev/null)" "$(cat $G/hold/target_$g 2>/dev/null)"; done)\n'
     'echo "$(date +%H:%M) running:[${st# }] | ${stage} | ${gpu}| held:${hold}| ${srv} | rollouts=${nro} ${upd}"'),
]
for a, b in reps:
    assert w.count(a) == 1, ("watch", a[:70])
    w = w.replace(a, b)
w = w.replace('pgrep -u mzjiang -f "$s.sh"', 'pgrep -u mzjiang -f "$s"')
open("v11ops/watch.sh", "w", encoding="utf-8", newline="\n").write(w)
print("ok")
