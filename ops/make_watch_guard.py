"""watch.sh: include the guard and the retry logs of the continuation. Run from ops/."""
w = open("v11ops/watch.sh", encoding="utf-8").read()
reps = [
    ("for s in run_v11_launch5 run_v11_formal5 run_v11_continue5 gpu_holder2; do",
     "for s in run_v11_launch5 run_v11_formal5 run_v11_continue5 gpu_holder2 run_v11_guard; do"),
    ("stage=$(cat $G/run_v11_launch5.log $G/run_v11_formal5.log $G/run_v11_continue5.log 2>/dev/null",
     "LAST=$(ls -t $G/run_v11_continue5_retry*.log 2>/dev/null | head -1)\n"
     "stage=$(cat $G/run_v11_launch5.log $G/run_v11_formal5.log $G/run_v11_continue5.log $LAST $G/run_v11_guard.log 2>/dev/null"),
    ("for f in $G/run_v11_launch5.log $G/run_v11_formal5.log $G/run_v11_continue5.log $G/gpu_holder2.log; do",
     "for f in $G/run_v11_launch5.log $G/run_v11_formal5.log $([ -n \"$LAST\" ] && echo $LAST || echo $G/run_v11_continue5.log) $G/gpu_holder2.log $G/run_v11_guard.log; do"),
    ('grep -E "^(===|WAITING_GPU|server GPU|gpt-oss|planner vLLM|train rc|verify|stop rule|early stop|LAUNCHER DONE|V11)"',
     'grep -E "^(===|WAITING_GPU|server GPU|gpt-oss|planner vLLM|train rc|verify|stop rule|early stop|LAUNCHER|V11|GUARD)"'),
    ('grep -nE "ABORT|STOP:|Traceback|FAILED|Killed|OutOfMemory|CUDA out of memory|MismatchAbort|did not pass" $f',
     'grep -nE "ABORT|STOP:|Traceback|FAILED|Killed|OutOfMemory|CUDA out of memory|MismatchAbort|did not pass|needs attention|giving up" $f'),
]
for a, b in reps:
    assert w.count(a) == 1, a[:60]
    w = w.replace(a, b)
open("v11ops/watch.sh", "w", encoding="utf-8", newline="\n").write(w)
print("ok")
