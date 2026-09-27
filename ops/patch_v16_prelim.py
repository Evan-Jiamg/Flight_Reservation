"""Audit Q fixes: run_v16_prelim.sh (trap cleanup, kill -9 escalation, numeric GPU checks, exec grace, timeouts,
final GPU0 check) and smoke_v16.py (MismatchAbort keeps the measured parts, adapter check)."""
import sys

root = sys.argv[1]


def patch(fn, pairs):
    p = root + "/" + fn
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (fn, s.count(old), old[:80])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


patch("ops/v11ops/run_v16_prelim.sh", [
    ('''up() { curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q "$2"; }
wait_up() {
  for i in $(seq 1 120); do
    up $1 $2 && return 0
    pgrep -u mzjiang -f "port $1" > /dev/null || { echo "server on $1 died:"; grep -E "Error|error" $3 | tail -3 | cut -c1-250; return 1; }
    sleep 10
  done
  return 1
}
stop_port() {   # our own server on this port only
  pkill -u mzjiang -f "vllm serve .*--port $1"
  for i in $(seq 1 60); do pgrep -u mzjiang -f "vllm serve .*--port $1" > /dev/null || break; sleep 2; done
  sleep 10
}''', '''up() { curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q "$2"; }
wait_up() {
  sleep 5                        # let the launcher exec into vllm before checking that it lives
  for i in $(seq 1 120); do
    up $1 $2 && return 0
    pgrep -u mzjiang -f "vllm serve .*--port $1" > /dev/null || { echo "server on $1 died:"; grep -E "Error|error" $3 | tail -3 | cut -c1-250; return 1; }
    sleep 10
  done
  return 1
}
stop_port() {   # our own server on this port only; escalates to SIGKILL, then waits for its GPU workers to go
  pkill -u mzjiang -f "vllm serve .*--port $1"
  for i in $(seq 1 60); do pgrep -u mzjiang -f "vllm serve .*--port $1" > /dev/null || break; sleep 2; done
  pkill -9 -u mzjiang -f "vllm serve .*--port $1"
  sleep 10
}
our_gpu0() {    # our compute processes on GPU0
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "mzjiang" ] && echo $p
  done
}
gpu0_used() {
  v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [[ "$v" =~ ^[0-9]+$ ]] || { echo "STOP: cannot read GPU0 memory ($v)" >&2; echo 999999; return; }
  echo $v
}
cleanup() {     # on any exit: our two servers down; our leftover GPU0 processes killed
  stop_port 8031; stop_port 8029
  for p in $(our_gpu0); do kill -9 $p 2>/dev/null; done
}
trap cleanup EXIT'''),
    ('''used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$used" -gt 2048 ]; then echo "STOP: GPU0 is in use ($used MiB) - not taking it"; exit 1; fi''',
     '''used=$(gpu0_used)
if [ "$used" -gt 2048 ]; then echo "STOP: GPU0 is in use ($used MiB) - not taking it"; trap - EXIT; exit 1; fi'''),
    ('''if pgrep -u mzjiang -f "vllm serve|train_planner_rl.py|gpu_holder2" > /dev/null; then echo "STOP: our servers / training already running"; exit 1; fi''',
     '''if pgrep -u mzjiang -f "vllm serve|train_planner_rl.py|gpu_holder2" > /dev/null; then echo "STOP: our servers / training already running"; trap - EXIT; exit 1; fi'''),
    ('''CUDA_VISIBLE_DEVICES=0 $PY smoke_v16.py''', '''CUDA_VISIBLE_DEVICES=0 timeout 3h $PY smoke_v16.py'''),
    ('''used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$used" -gt 2048 ]; then echo "STOP: GPU0 not free after the smoke ($used MiB)"; exit 1; fi''',
     '''used=$(gpu0_used)
if [ "$used" -gt 2048 ]; then echo "STOP: GPU0 not free after the smoke ($used MiB)"; exit 1; fi'''),
    ('''$PY step0_coverage.py --fold 2''', '''timeout 2h $PY step0_coverage.py --fold 2'''),
    ('''cat $OUT/step0.log | cut -c1-300
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "V16 PRELIM DONE $(date)"''', '''cat $OUT/step0.log | cut -c1-300
cleanup
trap - EXIT
left=$(our_gpu0 | wc -l)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
[ "$left" -eq 0 ] || { echo "WARNING: $left of our processes still on GPU0"; exit 1; }
echo "V16 PRELIM DONE $(date)"'''),
])
patch("sep-sim/smoke_v16.py", [
    ('''    t3 = time.time()
    stats = tr.learner.update(samples, tr.cfg, seed=T.seed_of(a.seed, "update", 1), aux=aux or None,
                              mismatch_abort=a.behav_mismatch_abort)''',
     '''    bad = sorted({s_.get("gen_adapter") for s_ in samples if s_.get("gen_adapter") != tr.gen_name})
    res["task1_groups"]["off_policy_adapters"] = bad          # must be [] (as one_update asserts)
    t3 = time.time()
    try:
        stats = tr.learner.update(samples, tr.cfg, seed=T.seed_of(a.seed, "update", 1), aux=aux or None,
                                  mismatch_abort=a.behav_mismatch_abort)
    except RA.MismatchAbort as e:                             # keep what was measured before the abort
        stats = {"mismatch_abort": e.value}'''),
    ('''                     **{k: stats.get(k) for k in ("rl_grad_norm", "aux_grad_norm", "grad_norm", "kl", "loss", "n_tokens",''',
     '''                     "mismatch_abort": stats.get("mismatch_abort"),
                     **{k: stats.get(k) for k in ("rl_grad_norm", "aux_grad_norm", "grad_norm", "kl", "loss", "n_tokens",'''),
])
print("patched")
