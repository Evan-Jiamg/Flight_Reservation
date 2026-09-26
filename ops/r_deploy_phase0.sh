G=/tmp2/mzjiang_usersim/grpo_planner; P=$G/phase0
if pgrep -u mzjiang -f run_phase0.sh > /dev/null; then echo "phase0 already running"; exit 1; fi
mkdir -p $P && cd $P && echo "$PAYLOAD_B64" | base64 -d | tar xzf - || exit 1
ls -la $P | tail -6
[ -f $G/runs/pend_f2_v10/ckpt/u00001/adapter/adapter_config.json ] || { echo "u1 adapter missing"; exit 1; }
setsid nohup bash $P/run_phase0.sh > $P/run_phase0.log 2>&1 < /dev/null &
echo "phase0 launched"
