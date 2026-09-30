echo "##### host"; hostname
echo "##### GPUs"; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "##### GPU processes: owner only"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do echo "pid $p user $(ps -o user= -p $p 2>/dev/null)"; done
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo "##### our processes"; pgrep -u mzjiang -af "train_planner_rl|eval_test_rl|vllm serve|gpu_holder2|run_v16" | grep -v pgrep | cut -c1-80; echo "(none above = idle)"
python3 - <<'PYEOF'
import json
R = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/"
print("##### v16 fold 2 per-update training statistics")
for l in open(R + "updates.jsonl"):
    r = json.loads(l)
    ls = r.get("learner_stats") or {}
    t1 = r.get("task1_train") or {}
    t2 = r.get("task2_groups") or r.get("groups") or {}
    cfg = r.get("cfg_used") or {}
    out = {"u": r.get("update"),
           "rl_g": ls.get("rl_grad_norm"), "aux_g": ls.get("aux_grad_norm"), "kl": ls.get("kl"),
           "adv_abs": ls.get("adv_abs_mean"), "n_samp": r.get("n_samples"), "aux_n": ls.get("aux_n"),
           "aux_p_before": ls.get("aux_p_correct_before"),
           "t1_acc": t1.get("acc"), "t1_end_final": t1.get("end_at_final"), "t1_end_nonfinal": t1.get("end_at_nonfinal"),
           "t1_skip0": t1.get("groups_skipped_zero_std"), "t1_base": t1.get("n_base_groups"), "t1_inform": t1.get("n_informative_groups"),
           "w": {k: cfg.get(k) for k in ("w_cov", "w_dist", "w_aux")}}
    print(json.dumps(out))
print("##### keys available (update 1)")
r = json.loads(open(R + "updates.jsonl").readline())
print(sorted(r.keys()))
print("learner_stats keys:", sorted((r.get("learner_stats") or {}).keys()))
for k in r:
    if isinstance(r[k], dict) and k not in ("learner_stats", "cfg_used", "next_cfg"):
        print(k, "->", sorted(r[k].keys())[:25])
PYEOF
