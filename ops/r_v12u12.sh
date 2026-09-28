python3 - <<'PYEOF'
import json, collections
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
for l in open(O + "updates.jsonl"):
    u = json.loads(l)
    if u["update"] < 9:
        continue
    a = u["train_aggregate"]; c = a["components_mean"]; st = u["learner_stats"]
    print("U%d rew %.3f shadow %s comps %s unclean %s singletons %s kl %.4f rl_gn %s aux_gn %s aux_w %s cfg %s" % (
        u["update"], a["reward_mean"], a.get("shadow_reward_mean"), {k: round(v, 3) for k, v in c.items() if isinstance(v, (int, float))},
        a.get("n_unclean_episodes"), a.get("n_dropped_singleton_episodes"), st.get("kl") or 0, st.get("rl_grad_norm"),
        st.get("aux_grad_norm"), a.get("aux_weight"),
        {k: round(u["cfg_used"][k], 3) for k in ("w_cov", "w_dist", "w_aux", "lambda_unparsed", "lambda_hit_max_new")}))
rows = [json.loads(l) for l in open(O + "rollouts.jsonl")]
for up in (11, 12):
    rs = [r["episode"] for r in rows if r["update"] == up]
    print("U%d n=%d turns %s human %s end %s cov %s clean %s" % (up, len(rs), [e["emitted_user_turns"] for e in rs],
          sorted(set((e["conversation_id"][:8], e["human_turns"]) for e in rs)),
          dict(collections.Counter(e["end_kind"] for e in rs)), [round(e["coverage"], 2) for e in rs],
          sum(1 for e in rs if e.get("clean"))))
PYEOF
