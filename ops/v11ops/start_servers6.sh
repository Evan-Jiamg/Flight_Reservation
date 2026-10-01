# Start OUR two vLLM servers on cfda5 from the incremental, dynamic GPU placeholder (gpu_holder3.py):
#   gpt-oss-120b (R0 + ledger judge + LLM controller)  port 8029, --gpu-memory-utilization 0.78, on GPU role_oss
#   Qwen3-4B Planner generation (runtime LoRA)          port 8031, --gpu-memory-utilization 0.15, on GPU role_planner
# The holder commits a placement (P1: both servers on one GPU, trainer on the other; P2: gpt-oss alone, Planner vLLM
# next to the trainer) once it holds all of it; role_train is the commit marker.
# Hand-over (user 2026-09-30, after the 23:38 lost race): vLLM (0.22.1 on cfda5; CUDA graphs inside the utilization)
# checks free >= util * total at startup. Right before each launch the placeholder releases what makes nvidia-smi free
# >= util * total + slack (gpt-oss 3 GiB, planner 2 GiB), never going below the floor the GPU must keep (the trainer's
# TRAIN_NEED on the training GPU); serve.sh is written beforehand and vLLM is started at once. A hand-over that would
# leave less than util * total + MIN_START_SLACK_GIB (1) free does not launch (-> re-plan). The Planner share stays held
# until gpt-oss is up.  P1: 91 -> 14 (gpt-oss) -> 0 (planner gets everything left, as start_servers5.sh did).
# P2: gpt-oss GPU 77 -> 0; planner/train GPU 61 -> 45 (the trainer then takes the 45 via take_train_gpu).
# A failed start is a lost memory race (NOT an error) when serve.log has vLLM's startup "less than desired GPU memory
# utilization" ValueError, or a CUDA OOM / memory-profiling / no-cache-blocks error AND nvidia-smi shows that memory was
# taken on that GPU since the launch (less free, or a new process of another user). Then our half-started server is
# killed (its own session only), the placeholder target on that GPU goes straight back to its pre-hand-over value (it
# re-takes the freed memory), the placeholder re-plans (`replan`, role files removed; it keeps what it holds), and the
# back-off (60, 120, 300, then 600 s; logged) is waited right before the NEXT launch, after the next commit -- never a stop
# (user decision 2026-10-01). A hand-over that cannot leave enough free memory (ambiguous: nothing launched) is
# re-planned at most MAX_SAME_RACE (3) times per server + GPU. Any other failure (model path, config, a memory error
# with no other process involved) -> exit 1.  Exit 0 = both servers up.
# SS6_LIB=1: only define the functions (test_start_servers6.py sources this file).
# OSS_ONLY=1 (SPEC v18 labelling, audit A): only gpt-oss is needed -- exit 0 as soon as gpt-oss is up; the Planner vLLM is
# not started (its share stays with the placeholder).
G=/tmp2/mzjiang_usersim/grpo_planner; H=${HOLD_DIR:-$G/hold}   # HOLD_DIR: only for a cut-over next to an old placeholder
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin
R=/tmp2/mzjiang_usersim/r0_vllm; Q=/tmp2/mzjiang_usersim/planner_vllm
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # placeholder / nvidia-smi / CUDA_VISIBLE_DEVICES indices agree
SERVER_NEED=${SERVER_NEED_GIB:-91}; OSS_NEED=${OSS_NEED_GIB:-77}; PLANNER_NEED=${PLANNER_NEED_GIB:-16}
TRAIN_NEED=${TRAIN_NEED_GIB:-45}
OSS_SLACK=${OSS_START_SLACK_GIB:-3}; PL_SLACK=${PLANNER_START_SLACK_GIB:-2}; MIN_SLACK=${MIN_START_SLACK_GIB:-1}
MAX_SAME_RACE=${MAX_SAME_RACE:-3}
OSS_UTIL=0.78; PL_UTIL=0.15
ME=${SS6_USER:-mzjiang}
HOLDER_PAT="python.* .*gpu_holder3\.py"
STARTUP_RE="less than desired GPU memory utilization"
MEM_RE="CUDA out of memory|CUDA error: out of memory|OutOfMemoryError|Error in memory profiling|No available memory for the cache blocks"
FATAL_RE="Engine core initialization failed"   # a bare Traceback may be a non-fatal warning path
declare -A RACES BACKOFF
# 2026-10-01: a server counts as up only if OUR vllm process serves the port -- another user's server on the same
# localhost port (seen at 20:03 on 8029) must never be reused
up() { pgrep -u $ME -f "vllm serve .*--port $1" > /dev/null && curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q "$2"; }
# a port that answers HTTP while no vllm process of ours serves it: another user's server -- never reused, never raced
foreign() { ! pgrep -u $ME -f "vllm serve .*--port $1" > /dev/null && curl -s -m 5 http://127.0.0.1:$1/v1/models > /dev/null 2>&1; }
held() { cat $H/status_$1 2>/dev/null || echo 0; }
tgt() { cat $H/target_$1 2>/dev/null || echo 0; }
role() { cat $H/role_$1 2>/dev/null; }
settarget() { echo $2 > $H/target_$1.tmp && mv -f $H/target_$1.tmp $H/target_$1; }
gpu_of() { grep -o 'CUDA_VISIBLE_DEVICES=[0-9]*' $1/serve.sh 2>/dev/null | head -1 | cut -d= -f2; }
holder_ok() {   # the holder process runs AND writes $H/plan (every tick) -- a wrong HOLD_DIR fails here
  pgrep -u $ME -f "$HOLDER_PAT" > /dev/null && [ -f $H/plan ] && [ $(( $(date +%s) - $(stat -c %Y $H/plan) )) -lt 60 ]
}
holder_alive() { holder_ok || { echo "STOP: the GPU holder (gpu_holder3.py) is not running or not writing $H/plan (HOLD_DIR?)"; exit 1; }; }
waitheld_le() { local n=0; until [ "$(held $1)" -le $2 ]; do n=$((n + 1)); [ $((n % 300)) -eq 0 ] && holder_alive; sleep 0.2; done; }
base() { [ "$1" = "$TG" ] && echo $TRAIN_NEED || echo 0; }      # what the placeholder keeps on GPU $1 once both servers run
gpu_mem() { nvidia-smi -i $1 --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' '; }
# handover_target GPU UTIL SLACK FLOOR -> the placeholder target for this launch, or "NO <free GiB after>".
#   SLACK = GiB above util * total wanted free (released from the holder, but it keeps >= FLOOR); SLACK "all": release
#   down to FLOOR. NO when even then free < util * total + MIN_SLACK.
handover_target() {
  local m; m=$(gpu_mem $1) || { echo "NO nvidia-smi"; return; }
  [ -n "$m" ] || { echo "NO nvidia-smi"; return; }
  awk -v u=$2 -v s=$3 -v f=$4 -v h=$(held $1) -v mn=$MIN_SLACK -v fm=${m% *} -v tm=${m#* } 'BEGIN {
    have = fm / 1024; req = u * tm / 1024
    if (s == "all") t = f
    else { r = req + s - have; rel = (r > int(r)) ? int(r) + 1 : int(r); if (rel < 0) rel = 0; t = h - rel }
    if (t < f) t = f
    if (t > h) t = h
    after = have + h - t
    if (after < req + mn) printf "NO %.1f\n", after; else print t }'
}
foreign_pids() {   # compute processes of OTHER users on GPU $1 (sorted, space separated)
  local u p owner
  u=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ' ' | awk -F, -v u="$u" '$2 == u {print $1}'); do
    owner=$(ps -o user= -p $p 2>/dev/null | tr -d ' ')
    [ -n "$owner" ] && [ "$owner" != "$ME" ] && echo $p
  done | sort -n | tr '\n' ' '
}
# classify_failure LOG GPU FREE_MIB_AT_LAUNCH "PIDS_AT_LAUNCH" -> "race <why>" or "error <why>" (our server already killed)
classify_failure() {
  grep -qE "$STARTUP_RE" $1 && { echo "race vLLM startup check: free < util * total"; return; }
  grep -qE "$MEM_RE" $1 || { echo "error not a memory failure"; return; }
  local m now p new=""
  m=$(gpu_mem $2); now=${m% *}
  for p in $(foreign_pids $2); do case " $4 " in *" $p "*) ;; *) new="$new $p";; esac; done
  if [ -n "$new" ]; then echo "race memory error; new process(es) of other users on GPU $2:$new"; return; fi
  if [ -n "$now" ] && [ "$now" -lt $(( $3 - 512 )) ]; then echo "race memory error; GPU $2 free fell from $3 to $now MiB since the launch"; return; fi
  echo "error memory error, but nobody took memory on GPU $2 (free $now MiB, $3 at launch) -- our own sizing?"
}
strike() {   # $1 key -> 1 when the (ambiguous) not-enough-memory hand-over happened MAX_SAME_RACE times
  RACES[$1]=$(( ${RACES[$1]:-0} + 1 ))
  if [ ${RACES[$1]} -ge $MAX_SAME_RACE ]; then echo "STOP: $1 could not be handed enough memory ${RACES[$1]} times on the same placement"; return 1; fi
  return 0
}
backoff() {   # $1 key: a lost race with evidence -- wait longer each time (60, 120, 300, 600, 600 ... s), never stop
  local n d
  n=$(( ${BACKOFF[$1]:-0} + 1 )); BACKOFF[$1]=$n
  case $n in 1) d=60;; 2) d=120;; 3) d=300;; *) d=600;; esac
  echo "BACKOFF: $1 lost race #$n -- waiting $d s before re-planning ($(date +%H:%M))"
  sleep $d
}
hint() {   # tell the placeholder which of our servers already run where (they need nothing from it)
  local o=- p=-
  if up 8029 gpt-oss-120b; then o=$(gpu_of $R); [ -n "$o" ] || { echo "ABORT: gpt-oss is up but $R/serve.sh names no GPU"; exit 1; }; fi
  if up 8031 planner-base; then p=$(gpu_of $Q); [ -n "$p" ] || { echo "ABORT: planner vLLM is up but $Q/serve.sh names no GPU"; exit 1; }; fi
  echo "oss=$o planner=$p" > $H/servers_up.tmp && mv -f $H/servers_up.tmp $H/servers_up
  [ "$o" != - ] && [ "$p" != - ]
}
request_replan() {
  hint
  rm -f $H/role_train $H/role_oss $H/role_planner $H/role_server
  echo 1 > $H/replan.tmp && mv -f $H/replan.tmp $H/replan
  echo "placeholder asked to re-plan ($(date +%H:%M))"
}
wait_commit() {   # 0 = committed (OG PG TG set); 1 = both servers answer meanwhile (the caller starts over)
  local n=0
  until [ -f $H/role_train ]; do
    if [ $((n % 25)) -eq 0 ]; then holder_alive; hint && return 1; fi
    [ $((n % 3000)) -eq 0 ] && echo "WAITING_GPU: placement not committed yet: $(cat $H/plan 2>/dev/null) ($(date +%H:%M))"
    n=$((n + 1)); sleep 0.2
  done
  sleep 0.5                                # role_train is written last; the others are in place
  OG=$(role oss); PG=$(role planner); TG=$(role train)
  return 0
}
wait_up() {   # $1 port  $2 model name  $3 pgrep pattern  $4 log  ($WAIT_UP_TRIES x $WAIT_UP_SLEEP s)
  local i
  for i in $(seq 1 ${WAIT_UP_TRIES:-120}); do
    up $1 $2 && return 0
    # 2026-09-30: grace of 3 checks (30 s) before calling the process dead -- right after `setsid nohup bash serve.sh &`
    # the command line is still "bash serve.sh" until its `exec vllm serve ... --port N` runs, so an immediate pgrep on
    # "port N" raced and aborted a healthy launch (smoke v17, 21:35).
    if [ $i -gt 3 ]; then
      pgrep -u $ME -f "$3" > /dev/null || { echo "server process died:"; grep -E "Error|error" $4 | tail -3 | cut -c1-250; return 1; }
      # alive: a memory line alone (e.g. FlashInfer autotuner "skipping tactic ... CUDA out of memory") is not fatal;
      # only with vLLM's "Engine core initialization failed" (or once the process has exited, above)
      if grep -qE "$FATAL_RE" $4 && grep -qE "$STARTUP_RE|$MEM_RE" $4; then
        echo "server failed (memory):"; grep -E "$STARTUP_RE|$MEM_RE" $4 | tail -2 | cut -c1-250; return 1; fi
    fi
    sleep ${WAIT_UP_SLEEP:-10}
  done
  echo "server not up after $(( ${WAIT_UP_TRIES:-120} * ${WAIT_UP_SLEEP:-10} / 60 )) min"; return 1
}
kill_session() {   # $1 session id of OUR setsid launch, $2 port -- only our processes of that launch are touched
  local s sids="$1" alive p i mine
  for p in $(pgrep -u $ME -f "vllm serve .*--port $2"); do sids="$sids $(ps -o sid= -p $p 2>/dev/null)"; done
  mine=$(ps -o sid= -p $$ | tr -d ' ')
  sids=$(printf "%s\n" $sids | tr -d ' ' | grep -v '^$' | sort -u | grep -vx "$mine")   # never our own (pipeline) session
  for s in $sids; do pkill -u $ME -s $s 2>/dev/null; done
  for i in $(seq 1 60); do
    alive=0; for s in $sids; do pgrep -u $ME -s $s > /dev/null && alive=1; done
    [ $alive = 0 ] && break; sleep 2
  done
  for s in $sids; do pkill -9 -u $ME -s $s 2>/dev/null; done
  sleep 3
}
write_serve() {   # $1 oss|planner  $2 gpu
  if [ $1 = oss ]; then
cat > $R/serve.sh <<EOF
#!/bin/bash
export PATH=$PYDIR:\$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$2 PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared VLLM_CACHE_ROOT=$R/cache
exec $PYDIR/vllm serve $M --served-model-name gpt-oss-120b \\
  --max-model-len ${OSS_MAX_MODEL_LEN:-32768} --host 127.0.0.1 --port 8029 --gpu-memory-utilization $OSS_UTIL
EOF
    chmod +x $R/serve.sh
  else
cat > $Q/serve.sh <<EOF
#!/bin/bash
export PATH=$PYDIR:\$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$2 PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared VLLM_CACHE_ROOT=$Q/cache
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec $PYDIR/vllm serve $Q4 --served-model-name planner-base --dtype bfloat16 \\
  --enable-lora --max-lora-rank 16 --max-loras 2 --enable-prefix-caching \\
  --max-model-len 20480 --host 127.0.0.1 --port 8031 --gpu-memory-utilization $PL_UTIL
EOF
    chmod +x $Q/serve.sh
  fi
}
# start_server oss|planner GPU -> 0 up, 75 lost the memory (re-plan and wait), 1 real error / too many identical races
start_server() {
  local which=$1 g=$2 port name util dir slack floor t sid lf lp c pre
  if [ $which = oss ]; then port=8029; name=gpt-oss-120b; util=$OSS_UTIL; dir=$R; slack=$OSS_SLACK; floor=$(base $g)
  else
    port=8031; name=planner-base; util=$PL_UTIL; dir=$Q
    if [ "$g" = "$TG" ]; then slack=$PL_SLACK; floor=$TRAIN_NEED     # P2: the trainer's 45 stay held
    else slack=all; floor=$(base $g); fi                              # P1: everything left on the server GPU
  fi
  write_serve $which $g
  # the back-off of an earlier lost race happens HERE: after the next commit, while the placeholder holds everything
  if [ -n "${NEXT_BACKOFF:-}" ]; then backoff "$NEXT_BACKOFF"; NEXT_BACKOFF=""; fi
  pre=$(held $g)
  t=$(handover_target $g $util $slack $floor)
  case "$t" in NO*)
    echo "NOT LAUNCHED: $name on GPU $g would see only ${t#NO } GiB free (needs util x total + $MIN_SLACK) -> re-plan"
    strike "$name@g$g" || return 1; sleep 20; return 75;; esac
  settarget $g $t; waitheld_le $g $t
  setsid nohup bash $dir/serve.sh > $dir/serve.log 2>&1 < /dev/null &
  sid=$!
  lf=$(gpu_mem $g); lf=${lf% *}; lp=$(foreign_pids $g)            # vLLM needs >10 s before it allocates anything
  echo "$name launched on GPU $g ($util; placeholder -> $t GiB, free $lf MiB) $(date +%H:%M)"
  wait_up $port $name "port $port" $dir/serve.log && return 0
  kill_session $sid $port
  c=$(classify_failure $dir/serve.log $g "$lf" "$lp")
  if [ "${c%% *}" = race ]; then
    settarget $g $pre                  # at once: the placeholder re-takes what our killed server freed (audit B)
    echo "RETRY: $name lost the memory hand-over on GPU $g (${c#race }) -- our half-started server killed, placeholder target back to $pre GiB, re-plan ($(date +%H:%M))"
    NEXT_BACKOFF="$name@g$g"           # the wait comes before the NEXT launch, not while the memory is released
    return 75
  fi
  echo "ABORT: $name did not come up (${c#error }):"; grep -v "Loading" $dir/serve.log | tail -5 | cut -c1-250
  return 1
}
lower_to_base() {   # never raises a target; the training GPU keeps TRAIN_NEED
  local g
  for g in $(printf "%s\n" $OG $PG $TG | sort -u); do
    [ "$(tgt $g)" -gt "$(base $g)" ] && settarget $g $(base $g)
  done
}
[ "${SS6_LIB:-0}" = 1 ] && return 0

mkdir -p $R $Q $H
M=$(ls -d /tmp2/hf_shared/hub/models--openai--gpt-oss-120b/snapshots/*/ | head -1)
Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
holder_alive
while true; do
  OUP=0; PUP=0; up 8029 gpt-oss-120b && OUP=1; up 8031 planner-base && PUP=1
  for port in 8029 8031; do
    [ "${OSS_ONLY:-0}" = 1 ] && [ $port = 8031 ] && continue
    foreign $port && { echo "STOP: port $port answers but is not served by a vllm process of $ME -- not ours, not reused"; exit 1; }
  done
  [ "${OSS_ONLY:-0}" = 1 ] && [ $OUP = 1 ] && { echo "gpt-oss up (OSS_ONLY: no planner vLLM)"; exit 0; }
  if [ ! -f $H/role_train ]; then
    if hint; then
      echo "servers already up: gpt-oss on GPU $(gpu_of $R), planner vLLM on GPU $(gpu_of $Q) (placeholder plans the training GPU only)"; exit 0; fi
    wait_commit || continue
  fi
  OG=$(role oss); PG=$(role planner); TG=$(role train)
  if { [ $OUP = 1 ] && [ "$(gpu_of $R)" != "$OG" ]; } || { [ $PUP = 1 ] && [ "$(gpu_of $Q)" != "$PG" ]; }; then
    echo "committed placement (oss $OG, planner $PG) does not match the running servers -> re-plan"
    request_replan; continue
  fi
  if [ $OUP = 1 ] && [ $PUP = 1 ]; then
    lower_to_base; echo "servers already up: gpt-oss on GPU $OG, planner vLLM on GPU $PG; training GPU $TG"; exit 0; fi
  echo "placement: gpt-oss GPU $OG, planner vLLM GPU $PG, training GPU $TG $(date +%H:%M)"
  if [ $OUP = 0 ]; then
    start_server oss $OG; rc=$?
    [ $rc = 75 ] && { request_replan; continue; }
    [ $rc = 0 ] || exit 1
  fi
  echo "gpt-oss up"
  [ "${OSS_ONLY:-0}" = 1 ] && { echo "OSS_ONLY: the planner vLLM is not started"; exit 0; }
  if [ $PUP = 0 ]; then
    start_server planner $PG; rc=$?
    [ $rc = 75 ] && { request_replan; continue; }
    [ $rc = 0 ] || exit 1
  fi
  echo "planner vLLM up"
  lower_to_base
  echo "servers up: gpt-oss GPU $OG, planner vLLM GPU $PG; training GPU $TG keeps $(tgt $TG) GiB for the trainer"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  exit 0
done
