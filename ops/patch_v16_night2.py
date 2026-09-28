"""Unattended-scripts review fixes (1-5)."""
import sys

root = sys.argv[1]


def patch(fn, pairs):
    p = root + "/" + fn
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (fn, s.count(old), old[:80])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


patch("ops/v11ops/run_v16_prelim.sh", [
    ('''our_gpu() {     # our compute processes on any GPU
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "mzjiang" ] && echo $p
  done
}''', '''our_gpu() {     # OUR processes of this prelim on any GPU (vLLM engines, the smoke, Step 0) - never other GPU work
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "mzjiang" ] || continue
    ps -o args= -p $p 2>/dev/null | grep -qE "vllm|VLLM|EngineCore|smoke_v16|step0_coverage" && echo $p
  done
}'''),
])
patch("ops/v11ops/run_v16_night.sh", [
    ('''[ -d $G/runs/pend_f2_v16 ] && { echo "NIGHT: runs/pend_f2_v16 already exists - not starting a second formal run"; exit 1; }''',
     '''exec 9>$G/.v16_night.lock
flock -n 9 || { echo "NIGHT: another night runner holds the lock"; exit 1; }
[ -d $G/runs/pend_f2_v16 ] && { echo "NIGHT: runs/pend_f2_v16 already exists - not starting a second formal run"; exit 1; }'''),
    ('''until [ -n "$(free_gpu)" ] && ! pgrep -u mzjiang -f "vllm serve|train_planner_rl.py|gpu_holder2|run_v16_prelim" > /dev/null; do''',
     '''until [ -n "$(free_gpu)" ] && ! pgrep -u mzjiang -f "vllm serve|train_planner_rl.py|gpu_holder2|run_v16_prelim\\.sh" > /dev/null; do'''),
    ('''checks = [''', '''aux_only = g.get("n_informative_groups") == 0          # every group zero-std: an aux-only update (no RL gradient)
checks = ['''),
    ('''    ("vLLM / learner mismatch <= 0.1", u.get("behav_mismatch_mean") is not None and u["behav_mismatch_mean"] <= 0.1),''',
     '''    ("vLLM / learner mismatch <= 0.1", aux_only or (u.get("behav_mismatch_mean") is not None and u["behav_mismatch_mean"] <= 0.1)),'''),
    ('''    ("finite RL gradient", u.get("rl_grad_norm") is not None and math.isfinite(u["rl_grad_norm"]) and u["rl_grad_norm"] > 0),
]''', '''    ("finite RL gradient", aux_only or (u.get("rl_grad_norm") is not None and math.isfinite(u["rl_grad_norm"]) and u["rl_grad_norm"] > 0)),
]
if aux_only:
    print("WARN  every Task 1 group had identical rewards: aux-only update, RL gradient / mismatch not measured")'''),
])
for fn in ("ops/v11ops/run_v16_launch.sh", "ops/v11ops/run_v16_guard.sh"):
    tag = "LAUNCHER" if "launch" in fn else "GUARD"
    patch(fn, [
        ('''    pkill -9 -u mzjiang -f "vllm serve .*--port $port"
  done
  echo "%s: GPUs released $(date)"''' % tag,
         '''    pkill -9 -u mzjiang -f "vllm serve .*--port $port"
  done
  sleep 5
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do     # orphaned engines of our servers
    [ "$(ps -o user= -p $p 2>/dev/null)" = "mzjiang" ] || continue
    ps -o args= -p $p 2>/dev/null | grep -qE "vllm|VLLM|EngineCore" && kill -9 $p 2>/dev/null
  done
  echo "%s: GPUs released $(date)"''' % tag),
    ])
print("patched")
