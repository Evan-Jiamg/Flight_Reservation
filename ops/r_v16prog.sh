python3 - <<'PYEOF'
import json, os
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v16/"
for l in open(O + "updates.jsonl"):
    u = json.loads(l); a = u["train_aggregate"]; st = u["learner_stats"]; c = a["components_mean"]; t = a.get("task1_train") or {}
    print("U%d %4.0fs rew %.3f cov %.3f dist %.3f turns %.2f | mism %.4f kl %.4f rl_gn %.4f aux_gn %.4f adv %.3f | aux_w %.2f floor_act %s | T1 fin %.2f early %.2f inf %s/%s refill %s %s | w_dist %.2f" % (
        u["update"], u["update_s"], a["reward_mean"], c.get("coverage", 0), c.get("dist", 0), c.get("turns", 0),
        st.get("behav_mismatch_mean") or 0, st.get("kl") or 0, st.get("rl_grad_norm") or 0, st.get("aux_grad_norm") or 0,
        st.get("adv_abs_mean") or 0, a.get("aux_weight", 0), a.get("aux_floor_active"), t.get("end_at_final") or 0,
        t.get("end_at_nonfinal") or 0, t.get("n_informative_groups"), t.get("n_base_groups"), t.get("n_refill_convs"),
        t.get("refill_stop"), u["cfg_used"]["w_dist"]))
for l in open(O + "validation.jsonl"):
    v = json.loads(l)
    if v.get("kind") == "summary":
        ts, t1 = v["turn_stats"] or {}, v["task1"] or {}
        print("VAL u%d sel=%.3f sim %.2f human %.2f w1 %.3f cov %.3f bal_p %.3f auc %s term_f1 %s d2 %s" % (v["update"], v["selection_score"] or -9,
              ts.get("sim_turns_mean", 0), ts.get("human_turns_mean", 0), ts.get("turn_w1") or 0, ts.get("coverage_mean", 0),
              t1.get("bal_p", -1), t1.get("auc"), t1.get("term_f1"), v.get("d2")))
p = O + "llm_controller.jsonl"
if os.path.exists(p):
    for l in open(p):
        r = json.loads(l)
        print("CTRL u%s ok=%s rollback=%s applied=%s" % (r.get("update"), r.get("ok"), r.get("rollback"),
              {k: (v["factor"], round(v["value"], 3)) for k, v in (r.get("applied") or {}).items()}))
        print("   ", (r.get("rationale") or r.get("error") or "")[:300])
PYEOF
tail -3 /tmp2/mzjiang_usersim/grpo_planner/run_v16_formal.log | cut -c1-200
