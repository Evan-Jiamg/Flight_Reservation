"""Generate the v3 gate / continuation / launcher scripts (servers on whichever cfda5 GPU frees first) and update
watch.sh. Run from ops/."""
OLD_PICK = """pickgpu() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \\
            awk -F', ' '$1!=0 {f=int(($2-$3)/1024); if (f>=42) print f" "$1}' | sort -nr | head -1 | cut -d" " -f2; }"""
NEW_PICK = """pickgpu() { SG=$(cat $G/server_gpu.txt 2>/dev/null || echo -1); nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \\
            awk -F', ' -v sg=$SG '$1!=sg {f=int(($2-$3)/1024); if (f>=42) print f" "$1}' | sort -nr | head -1 | cut -d" " -f2; }"""
for src, dst, reps in (
    ("v11ops/run_v11_formal2.sh", "v11ops/run_v11_formal3.sh",
     [("bash $G/start_servers2.sh", "bash $G/start_servers3.sh"), ("(not GPU0: the servers)", "(not the servers' GPU)")]),
    ("v11ops/run_v11_continue2.sh", "v11ops/run_v11_continue3.sh",
     [("bash $G/start_servers2.sh", "bash $G/start_servers3.sh"), ("run_v11_formal2.sh", "run_v11_formal3.sh"),
      ("run_v11_formal2.log", "run_v11_formal3.log")]),
):
    s = open(src, encoding="utf-8").read()
    assert s.count(OLD_PICK) == 1, src
    s = s.replace(OLD_PICK, NEW_PICK)
    for a, b in reps:
        assert a in s, (src, a)
        s = s.replace(a, b)
    open(dst, "w", encoding="utf-8", newline="\n").write(s)

launcher = """#!/bin/bash
# v11 launcher (3): servers on whichever cfda5 GPU frees first, the trainer on any other GPU with room; gate then
# formal continuation. Waits for any older launcher chain of ours to be gone first.
G=/tmp2/mzjiang_usersim/grpo_planner
while pgrep -u mzjiang -f "run_v11_launch.sh|run_v11_formal2.sh|run_v11_continue2.sh" > /dev/null; do sleep 30; done
echo "older chain gone $(date)"
bash $G/run_v11_formal3.sh > $G/run_v11_formal3.log 2>&1
bash $G/run_v11_continue3.sh > $G/run_v11_continue3.log 2>&1
echo "LAUNCHER3 DONE $(date)"
"""
open("v11ops/run_v11_launch3.sh", "w", encoding="utf-8", newline="\n").write(launcher)

w = open("v11ops/watch.sh", encoding="utf-8").read()
reps = [
    ("for s in run_v11_launch run_v11_formal2 run_v11_continue2 run_v11_formal run_v11_continue; do",
     "for s in run_v11_launch3 run_v11_formal3 run_v11_continue3 run_v11_launch run_v11_formal2 run_v11_continue2; do"),
    ('stage=$(cat $G/run_v11_launch.log $G/run_v11_formal2.log $G/run_v11_continue2.log 2>/dev/null | grep -E "^(===|WAITING_GPU0|GPU0 has|',
     'stage=$(cat $G/run_v11_launch3.log $G/run_v11_formal3.log $G/run_v11_continue3.log 2>/dev/null | grep -E "^(===|WAITING_GPU|server GPU|'),
    ("for f in $G/run_v11_launch.log $G/run_v11_formal2.log $G/run_v11_continue2.log; do",
     "for f in $G/run_v11_launch3.log $G/run_v11_formal3.log $G/run_v11_continue3.log; do"),
]
for a, b in reps:
    assert w.count(a) == 1, a[:60]
    w = w.replace(a, b)
open("v11ops/watch.sh", "w", encoding="utf-8", newline="\n").write(w)
print("ok")
