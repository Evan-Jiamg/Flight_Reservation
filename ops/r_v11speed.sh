RUN=/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11
echo "--- GPU processes (memory) and whether ours"
ours=" $(pgrep -u mzjiang | tr '\n' ' ') "
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader | while IFS=, read pid mem uuid; do
  case "$ours" in *" $pid "*) o=OURS;; *) o=other;; esac; echo "$pid $mem ${uuid: -6} $o"; done
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader | awk -F', ' '{print $1, substr($2, length($2)-5), $3, $4}'
python3 - <<'PYEOF'
import json, collections
rows = [json.loads(l) for l in open("/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/rollouts.jsonl")]
by = collections.defaultdict(list)
for r in rows:
    by[r["update"]].append(r)
for u in sorted(by):
    rs = by[u]
    tot = collections.Counter(); n = collections.Counter(); turns = []
    for r in rs:
        e = r["episode"]; turns.append(e["emitted_user_turns"])
        for s in e["trace"]:
            for k in ("planner_s", "speaker_s", "r0_s", "ledger_s"):
                if s.get(k) is not None:
                    tot[k] += s[k]; n[k] += 1
    print("U%d episodes %d mean turns %.2f rollout_s mean %.0f per-step %s" % (u, len(rs), sum(turns) / len(turns),
          sum(r["rollout_s"] for r in rs) / len(rs), {k: round(tot[k] / n[k], 1) for k in tot}))
PYEOF
