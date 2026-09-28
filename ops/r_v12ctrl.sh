python3 - <<'PYEOF'
import json
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
for l in open(O + "llm_controller.jsonl"):
    r = json.loads(l)
    print("u%s ok=%s rollback=%s applied=%s" % (r.get("update"), r.get("ok"), r.get("rollback"), r.get("applied")))
    print("   ", (r.get("rationale") or r.get("error") or "")[:500])
for l in open(O + "updates.jsonl"):
    u = json.loads(l)
    print("U%d used w_dist %.3f next w_dist %.3f" % (u["update"], u["cfg_used"]["w_dist"], u["next_cfg"]["w_dist"]))
PYEOF
