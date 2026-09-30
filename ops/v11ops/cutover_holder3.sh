#!/bin/bash
# Cut-over from the old placeholders to gpu_holder3.py on cfda5 (2026-10-01), without giving memory away:
#   old: gpu_holder2.py in $G/hold (status_<g> / target_<g>), gpu_grab.py in $G/grab1 (status / target; GPU 1)
#   new: gpu_holder3.py in $G/hold3 (the two cannot share hold/: same file names)
# 0. pre-checks, nothing touched before they pass: gpu_holder2's hold/role_server and hold/role_train exist; gpu_grab's
#    pid sits on GPU 1's UUID only.
# 1. freeze the old ones (grab1/target first -- without a target gpu_grab.py keeps grabbing --, then hold/target_<g>) at
#    what they hold now,
# 2. start holder3 in hold3/,
# 3. every second: write hold3/extra_<g> = GiB the old holders still hold on GPU g (holder3 counts it as available when
#    it CHOOSES its plan, so a plan that needs this memory can be chosen; min(status, target), rewritten right after each
#    step; holder3 ignores it when older than 10 s); lower ONE old holder by up to STEP (2) GiB on
#    a GPU where holder3's target exceeds what it holds, then wait until holder3's status there rises (absorbed). Not
#    absorbed within 10 s: WARN, all old holders stay frozen; within 120 s: ABORT (old holders frozen at their current
#    size, holder3 left running, extra_<g> still accurate). Once holder3 has COMMITTED, old memory on GPUs where it
#    wants no more is surplus and is released at once,
# 4. when the old holders hold 0 GiB: touch their stop files, wait for them to exit, remove hold3/extra_<g>.
# An item whose process is gone is dropped (gpu_holder2's two items share one pid).
# Afterwards every run_v17 script needs:  export HOLD_DIR=/tmp2/mzjiang_usersim/grpo_planner/hold3
# Usage: nohup bash cutover_holder3.sh > cutover_holder3.log 2>&1 &
set -uo pipefail
G=${CUTOVER_G:-/tmp2/mzjiang_usersim/grpo_planner}; OLD=$G/hold; GRAB=$G/grab1; H3=$G/hold3
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
STEP=${CUTOVER_STEP_GIB:-2}
H2PAT="python.* .*gpu_holder2\.py"; GRPAT="python.* .*gpu_grab\.py"; H3PAT="python.* .*gpu_holder3\.py"
log() { echo "$(date +%H:%M:%S) $*"; }
rdn() { local v; v=$(cat $1 2>/dev/null); v=${v%.*}; [ -n "$v" ] && echo $v || echo 0; }
setf() { echo $2 > $1.tmp && mv -f $1.tmp $1; }
holder_ok() { pgrep -u mzjiang -f "$H3PAT" > /dev/null && [ -f $H3/plan ] && [ $(( $(date +%s) - $(stat -c %Y $H3/plan) )) -lt 60 ]; }
abort() { log "ABORT: $*"; exit 1; }
# ---- 0. pre-checks
H2PID=$(pgrep -u mzjiang -f "$H2PAT" | head -1); GRPID=$(pgrep -u mzjiang -f "$GRPAT" | head -1)
[ -n "$H2PID$GRPID" ] || log "no old placeholder running"
if [ -n "$H2PID" ]; then
  [ -f $OLD/role_server ] && [ -f $OLD/role_train ] || abort "gpu_holder2 (pid $H2PID) has no hold/role_server + hold/role_train yet (still assigning roles) - nothing touched"
fi
if [ -n "$GRPID" ]; then
  U1=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ' | awk -F, '$1 == 1 {print $2}')
  ON=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ' ' | awk -F, -v p=$GRPID '$1 == p {print $2}' | sort -u)
  [ -n "$U1" ] && [ "$ON" = "$U1" ] || abort "gpu_grab.py (pid $GRPID) is on '$ON', not only on GPU 1 ($U1) - nothing touched"
fi
# ---- 1. freeze (items: name target status gpu pid)
ITEMS=()
if [ -n "$GRPID" ]; then setf $GRAB/target $(rdn $GRAB/status); ITEMS+=("grab1 $GRAB/target $GRAB/status 1 $GRPID"); log "froze grab1 at $(rdn $GRAB/status) GiB"; fi
if [ -n "$H2PID" ]; then
  for g in 0 1; do setf $OLD/target_$g $(rdn $OLD/status_$g); ITEMS+=("hold2_g$g $OLD/target_$g $OLD/status_$g $g $H2PID"); log "froze hold2 g$g at $(rdn $OLD/status_$g) GiB"; done
fi
# ---- 2. holder3
if ! pgrep -u mzjiang -f "$H3PAT" > /dev/null; then
  rm -rf $H3; mkdir -p $H3
  CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder3.py $H3 >> $G/gpu_holder3.log 2>&1 < /dev/null &
  log "gpu_holder3.py started in $H3"
fi
for i in $(seq 1 60); do holder_ok && break; sleep 1; done
holder_ok || abort "holder3 not up (see $G/gpu_holder3.log); old holders left frozen"
live() { kill -0 $1 2>/dev/null; }
write_extra() {   # min(status, target): a step just ordered is not counted while the old holder still shows it
  local g it e st tg
  for g in 0 1; do
    e=0
    for it in "${ITEMS[@]}"; do
      set -- $it; [ $4 = $g ] && live $5 || continue
      st=$(rdn $3); tg=$st; [ -f $2 ] && tg=$(rdn $2); [ $tg -lt $st ] && st=$tg
      e=$((e + st))
    done
    setf $H3/extra_$g $e
  done
}
# ---- 3. hand over
declare -A LAST WHEN
PEND=""; WARNED=0; IDLE=0
while true; do
  holder_ok || abort "holder3 died / stopped writing $H3/plan; old holders frozen at their current size ($H3/extra_* left accurate)"
  write_extra
  left=0; for it in "${ITEMS[@]}"; do set -- $it; live $5 && left=$((left + $(rdn $3))); done
  [ $left -eq 0 ] && break
  if [ -n "$PEND" ]; then                       # the last step must be absorbed before anything else moves
    set -- $PEND; name=$1; g=$4
    h3=$(rdn $H3/status_$g); age=$(( $(date +%s) - ${WHEN[$name]} ))
    if ! live $5; then log "$name: process gone, dropped"; PEND=""; WARNED=0
    elif [ $h3 -gt ${LAST[$name]} ]; then log "g$g: holder3 absorbed ($h3 GiB)"; PEND=""; WARNED=0
    else
      [ $age -ge 120 ] && abort "step of $name not absorbed by holder3 within 120 s (others took it?); old holders frozen at their current size, holder3 running, $H3/extra_* accurate -- remove them after stopping the old holders by hand"
      [ $age -ge 10 ] && [ $WARNED = 0 ] && { log "WARN: step of $name not absorbed within 10 s; all old holders frozen until it is (abort at 120 s)"; WARNED=1; }
      sleep 1; continue
    fi
  fi
  committed=0; [ -f $H3/role_train ] && committed=1
  stepped=0
  for it in "${ITEMS[@]}"; do
    set -- $it; name=$1; tf=$2; sf=$3; g=$4; pid=$5
    live $pid || continue
    old=$(rdn $sf); [ $old -gt 0 ] || continue
    h3=$(rdn $H3/status_$g); t3=$(rdn $H3/target_$g)
    if [ $t3 -gt $h3 ]; then
      amt=$STEP; [ $old -lt $amt ] && amt=$old; [ $((t3 - h3)) -lt $amt ] && amt=$((t3 - h3))
      setf $tf $((old - amt)); write_extra; LAST[$name]=$h3; WHEN[$name]=$(date +%s); PEND="$it"; stepped=1
      log "g$g: $name $old -> $((old - amt)) GiB (holder3 $h3/$t3 GiB$([ $committed = 1 ] && echo ', committed'))"
      break
    elif [ $committed = 1 ]; then
      setf $tf 0; write_extra; stepped=1
      log "g$g: $name $old -> 0 GiB (surplus: holder3 committed and holds its target $h3/$t3 GiB)"
    fi
  done
  if [ $stepped = 0 ]; then
    [ $((IDLE % 60)) -eq 0 ] && log "waiting: holder3 wants no more on the GPUs with old memory and has not committed: $(cat $H3/plan)"
    IDLE=$((IDLE + 1))
  fi
  sleep 1
done
# ---- 4. stop the old ones
log "old placeholders hold 0 GiB; stopping them"
[ -n "$H2PID" ] && touch $OLD/stop
[ -n "$GRPID" ] && touch $GRAB/stop
for i in $(seq 1 30); do pgrep -u mzjiang -f "python.* .*(gpu_holder2|gpu_grab)\.py" > /dev/null || break; sleep 1; done
pgrep -u mzjiang -f "python.* .*(gpu_holder2|gpu_grab)\.py" > /dev/null && log "WARN: an old placeholder is still running after its stop file"
rm -f $H3/extra_0 $H3/extra_1
log "cut-over done: $(cat $H3/plan)"
log "now: export HOLD_DIR=$H3   (for run_v17_smoke.sh / run_v17_fold.sh / run_v17_test.sh / start_servers6.sh)"
