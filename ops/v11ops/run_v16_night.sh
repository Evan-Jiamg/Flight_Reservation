#!/bin/bash
# Night runner (user 2026-09-28: "完整確認無誤後，開始正式訓練"; GPU0 and GPU1 both usable, dynamic):
#  1. wait until some GPU is completely free (< 2 GiB) and none of our processes run (never take a GPU in use);
#  2. run_v16_prelim.sh (smoke with the real models, then Step 0);
#  3. the formal run starts ONLY if the smoke finished and passes every check below (a Step 0 failure is reported but
#     does not block the formal run: Step 0 is a diagnostic);
#  4. formal run: run_v16_launch.sh (dynamic all-at-once placeholder over both GPUs) + run_v16_guard.sh (OOM recovery).
G=/tmp2/mzjiang_usersim/grpo_planner; OUT=$G/runs/v16_prelim
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $2 < 2048 {print $1; exit}'
}
exec 9>$G/.v16_night.lock
flock -n 9 || { echo "NIGHT: another night runner holds the lock"; exit 1; }
[ -d $G/runs/pend_f2_v16 ] && { echo "NIGHT: runs/pend_f2_v16 already exists - not starting a second formal run"; exit 1; }
n=0
until [ -n "$(free_gpu)" ] && ! pgrep -u mzjiang -f "vllm serve|train_planner_rl.py|gpu_holder2|run_v16_prelim\.sh" > /dev/null; do
  [ $((n % 10)) -eq 0 ] && echo "NIGHT WAITING_GPU: no completely free GPU yet; used: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '{printf "g%s=%dG ", $1, $2/1024}')($(date +%H:%M))"
  n=$((n + 1)); sleep 60
done
echo "NIGHT: GPU $(free_gpu) free -> prelim $(date +%H:%M)"
bash $G/run_v16_prelim.sh > $G/run_v16_prelim.log 2>&1
echo "NIGHT: prelim ended rc=$? $(date +%H:%M): $(grep -E '^(smoke rc|step0 rc|STOP|V16 PRELIM DONE)' $G/run_v16_prelim.log | tr '\n' ' ' | cut -c1-200)"
if ! grep -q "^smoke rc=0" $G/run_v16_prelim.log || [ ! -f $OUT/smoke_run/smoke_v16.json ]; then
  echo "NIGHT STOP: the smoke did not finish - no formal run"; exit 1; fi
$PY - <<'PYEOF' > $G/run_v16_smoke_check.txt 2>&1
import json, math, sys
r = json.load(open("/tmp2/mzjiang_usersim/grpo_planner/runs/v16_prelim/smoke_run/smoke_v16.json"))
v, g, u, a = r["val_task1"], r["task1_groups"], r["update"], r["args"]
aux_only = g.get("n_informative_groups") == 0          # every group zero-std: an aux-only update (no RL gradient)
checks = [
    ("Dr. GRPO (grpo_std_norm false)", r["algo_cfg_grpo_std_norm"] is False),
    ("controller w_dist bounds [1, 5]", r["controller_w_dist_bounds"] == [1.0, 5.0]),
    ("spec task1_G 8 / task1_convs 8 / floor 0.5 / margin 0.10",
     a["task1_G"] == 8 and a["task1_convs"] == 8 and a["stop_sup_floor"] == 0.5 and a["t1_trigger_margin"] == 0.1),
    ("validation Task 1 has decision points", v["n_points"] > 0),
    ("scored end probabilities >= 70% of the points", v["n_points"] > 0 and v["n_valid"] / v["n_points"] >= 0.7),
    ("greedy decision agrees with p_end > 0.5 on >= 80%", v["greedy_vs_p_agree"] is not None and v["greedy_vs_p_agree"] >= 0.8),
    ("bal_p computed", v["metrics"] is not None and 0.0 <= v["metrics"]["bal_p"] <= 1.0),
    ("Task 1 groups of 8 samples", g["samples_per_row"] == [8]),
    ("valid Task 1 decisions >= 70%", g["valid_rate"] >= 0.7),
    ("no off-policy generation", g["off_policy_adapters"] == []),
    ("no mismatch abort", u.get("mismatch_abort") is None),
    ("vLLM / learner mismatch <= 0.1", aux_only or (u.get("behav_mismatch_mean") is not None and u["behav_mismatch_mean"] <= 0.1)),
    ("stop supervision present", u["aux_n"] > 0 and u["aux_weight"] >= 0.5),
    ("finite RL gradient", aux_only or (u.get("rl_grad_norm") is not None and math.isfinite(u["rl_grad_norm"]) and u["rl_grad_norm"] > 0)),
]
if aux_only:
    print("WARN  every Task 1 group had identical rewards: aux-only update, RL gradient / mismatch not measured")
for name, ok in checks:
    print("%s  %s" % ("PASS" if ok else "FAIL", name))
print("step sizes: rl_grad_norm %s aux_grad_norm %s adv_abs_mean %s (with std normalisation %s) kl %s mismatch %s" % (
    u.get("rl_grad_norm"), u.get("aux_grad_norm"), u.get("adv_abs_mean_dr_grpo"), u.get("adv_abs_mean_if_std_norm"),
    u.get("kl"), u.get("behav_mismatch_mean")))
print("validation Task 1: %s" % json.dumps({k: v["metrics"][k] for k in ("bal_p", "auc", "logloss", "n_points", "n_invalid")}))
print("refill: %s" % json.dumps({k: g[k] for k in ("n_base_groups", "n_refill_groups", "n_refill_convs", "n_informative_groups", "refill_stop")}))
print("SMOKE VERDICT: %s" % ("PASS" if all(ok for _, ok in checks) else "FAIL"))
PYEOF
cat $G/run_v16_smoke_check.txt
grep -q "SMOKE VERDICT: PASS" $G/run_v16_smoke_check.txt || { echo "NIGHT STOP: smoke checks failed - no formal run"; exit 1; }
if pgrep -u mzjiang -f "vllm serve|train_planner_rl.py|gpu_holder2" > /dev/null; then
  echo "NIGHT STOP: our processes still running after the prelim - no formal run"; exit 1; fi
echo "NIGHT: smoke passed -> formal run $(date +%H:%M)"
setsid nohup bash $G/run_v16_launch.sh > $G/run_v16_launch.log 2>&1 < /dev/null &
sleep 5
setsid nohup bash $G/run_v16_guard.sh > $G/run_v16_guard.log 2>&1 < /dev/null &
sleep 30
tail -3 $G/run_v16_launch.log
echo "NIGHT: formal run launched $(date +%H:%M)"
