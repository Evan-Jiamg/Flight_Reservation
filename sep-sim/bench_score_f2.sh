#!/bin/bash
# Score the v17 pend fold-2 teacher-forced generations (bench_tf_generate.py) with the TREC 2026 user-simulation
# benchmark's tools/score_method.py, the benchmark repo used READ-ONLY (nothing is written inside it: --out,
# --judge-cache-dir and --dump-per-turn point under $O; PYTHONDONTWRITEBYTECODE=1).
#
#   bash bench_score_f2.sh 0|5|all            score u0 (SFT) / u5 (final GRPO) / both
#   PREFLIGHT_ONLY=1 bash bench_score_f2.sh all   every check below, no judge call (no generations needed yet)
#
# Settings and why (benchmark CLAUDE.md §1, protocols/cross_validation_3fold_goal_persona.md "怎麼跑一個 fold"):
#   --method-id   built by tools/metrics/results_directory.py: <slot>__teacher_forced__goal_persona_fold_2_test_side__
#                 judge_gpt_oss_120b_effort_low__scorer_v3 (judge_slot / scorer_slot of the DEFAULT scorer version);
#                 parse() must round-trip and the slot must not be a retired method id.
#   --corpus      the official corpus (sha pinned = the fold manifest's source_sha256) SLICED to fold 2's 9 test
#                 sessions (§1.3 pit 1: generations and corpus both sliced, else the human side is 249 turns and
#                 the act judge loses prev_agent_text); the slice is checked: 9 finished records, 40 user turns.
#   --split-manifest / --fold 2   records the split, refuses any generation row off fold 2's test side.
#   --skip requirement_item_overlap,human_utterance_copying   §1.3 pit 2 / §1.5 (both are wrong on a sliced corpus;
#                 the reference invocation run_eval_gp.sh:74-77). --test-goals 5,9,10 is passed as asked but is
#                 only read by requirement_item_overlap, which is skipped.
#   --termination our fold-2 K+1 probe + rollout file (tools/probe_termination.py); checked to be fold 2 (§1.3 pit 3).
#   judge         local gpt-oss-120b (:8029), --reasoning-effort low and 2AFC effort low = the scorer-v3 fold runs
#                 in results/ (judge_gpt_oss_120b_effort_low). For gpt-oss the effort is NOT sent (gpt-5 dialect
#                 only); it is part of the cache key and of the judge slot. 2AFC calibration record:
#                 instruments/twoafc_gates_gpt-oss-120b.json (main domain, gates_pass true; checked here).
#   end policy    default `count` (never passed, CLAUDE.md §1.1); scorer version default (v3-2026-09-23).
#   order         generation (bench_tf_generate.py, ONE GPU, no gpt-oss) first; then start gpt-oss on :8029 (the GPU the
#                 generation freed) and run this script -- it stops with a clear message if :8029 does not answer.
#                 SCORE_JUDGE_URL overrides the endpoint. score_env_provenance.json / judge_servers.txt next to each
#                 results file: package versions, both judge servers (gates calibrated on :8011, scored on :8029).
#   naming        slots ..._v17_fold_2_sft_u0 / ..._v17_fold_2_grpo_u5; the stop_u*.json files written earlier used
#                 "<prefix>__sft_u0" / "<prefix>__grpo_u5" as their score_stop method id (same policies).
set -uo pipefail
WHICH=${1:-all}
B=/tmp2/hchsu/trec2026-usersim-benchmark
G=/tmp2/mzjiang_usersim/grpo_planner
O=$G/bench_eval_f2_v17
FOLD=2
TEST_GOALS=5,9,10
SLOT_PREFIX=sep_sim_pend_qwen3_4b_planner_ditto_8b_speaker_v17_fold_2
CORPUS_SRC=/tmp2/TREC_UserSim_MingZhi/UserLM/04-data/final_dataset_curation_v1.jsonl
CORPUS_SHA=288f0f4ec7404bca0807567ba15a42ca50bae8c3d01b5416c60b6de7d5eac53f
MANIFEST=domains/main_dataset_search/folds3_goal_persona_v1.json
MANIFEST_SHA=b610ba6b3b680e7d28b371f8207383dc4a8cafec0770089cff9533a87bfdec1f
GATES=instruments/twoafc_gates_gpt-oss-120b.json
GATES_SHA=ad5697f8a0c42c6631ce32e5982e7e85c957de1e2953f4ee457ebaf677f1cf8b
BENCH_HEAD=99206550caf72820d3953c98e11d60cb408df46a      # the benchmark commit these checks were written against
CORPUS_F2=$O/corpus_goal_persona_fold_2_test_side.jsonl
JMODEL=gpt-oss-120b
EFFORT=low
BASEPY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
VENV=/tmp2/mzjiang_usersim/venv_bench_score        # consistent-test + scikit-learn + sentence-transformers (never $HOME)
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HOME=/tmp2/mzjiang_usersim/hf_cache HF_HUB_OFFLINE=1 SCORER_EMBED_DEVICE=cpu   # all-MiniLM-L6-v2 is cached there
export PIP_CACHE_DIR=/tmp2/mzjiang_usersim/pip_cache XDG_CACHE_HOME=/tmp2/mzjiang_usersim/xdg_cache
# the benchmark judge refuses the environment variables renamed on 2026-09-27 (our run scripts still export them)
unset JUDGE_BASE_URL JUDGE_MODEL JUDGE_REASONING_EFFORT R0_BASE_URL R0_MODEL R0_REASONING_EFFORT JUDGE_BASE_URLS \
      JUDGE_URLS JUDGE_URL R0_URL
JUDGE_URL=${SCORE_JUDGE_URL:-http://127.0.0.1:8029/v1}   # (after the unset: JUDGE_URL is one of the retired names)
case $WHICH in 0) US="0";; 5) US="5";; all) US="0 5";; *) echo "usage: $0 0|5|all"; exit 1;; esac
slot_of() { [ "$1" = 0 ] && echo ${SLOT_PREFIX}_sft_u0 || echo ${SLOT_PREFIX}_grpo_u$1; }
mkdir -p $O
cd $B || { echo "STOP: no benchmark repo $B"; exit 1; }

# ------------------------------------------------------------------ pins
head_now=$(cat .git/refs/heads/master 2>/dev/null || awk '/refs\/heads\/master/ {print $1}' .git/packed-refs)
[ "$head_now" = "$BENCH_HEAD" ] || echo "NOTE: benchmark HEAD is $head_now (checks written against $BENCH_HEAD) - re-read score_method.py's refusals if the scorer changed"
echo "$MANIFEST_SHA  $MANIFEST" | sha256sum -c --quiet || { echo "STOP: fold manifest sha changed"; exit 1; }
echo "$GATES_SHA  $GATES" | sha256sum -c --quiet || { echo "STOP: 2AFC calibration record changed"; exit 1; }
echo "$CORPUS_SHA  $CORPUS_SRC" | sha256sum -c --quiet || { echo "STOP: official corpus sha differs"; exit 1; }

# ------------------------------------------------------------------ fold-2 corpus slice (+ its checks)
$BASEPY - "$CORPUS_SRC" "$MANIFEST" "$FOLD" "$CORPUS_F2" <<'PY' || { echo "STOP: corpus slice check failed"; exit 1; }
import json, os, sys
src, man_p, fold, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
man = json.load(open(man_p, encoding="utf-8"))
f = [x for x in man["folds"] if int(x["fold"]) == fold][0]
ids = set(f["session_ids"])
lines = [l for l in open(src, encoding="utf-8") if l.strip()]
keep = [l if l.endswith("\n") else l + "\n" for l in lines if json.loads(l)["conversation_id"] in ids]
recs = [json.loads(l) for l in keep]
assert len(recs) == len(ids) == f["n_sessions"], ("records", len(recs), len(ids))
assert all(any(m.get("is_final") is True for m in r["chat_messages"]) for r in recs), "an unfinished session"
n_user = sum(1 for r in recs for m in r["chat_messages"] if m["participant_name"] == "User")
assert n_user == f["n_user_turns"], ("user turns", n_user, f["n_user_turns"])
body = "".join(keep)
if os.path.exists(out):
    assert open(out, encoding="utf-8").read() == body, "existing slice %s differs from a fresh slice" % out
else:
    open(out + ".tmp", "w", encoding="utf-8").write(body); os.replace(out + ".tmp", out)
print("corpus slice OK: %d sessions / %d user turns -> %s" % (len(recs), n_user, out))
PY

# ------------------------------------------------------------------ per update
for U in $US; do
  SLOT=$(slot_of $U)
  GEN=${GEN_OVERRIDE:-$O/generations_u$U.jsonl}
  TERM=$O/termination_u$U.json
  MID=$(cd tools && $BASEPY - "$SLOT" <<'PY'
import sys
from metrics import results_directory as RD
from metrics.versions import DEFAULT_SCORER_VERSION
slot = sys.argv[1]
hint = RD.retired_name_hint(slot, "method_id")
assert not hint, "retired method id: " + hint
name = RD.build(slot, "teacher_forced", "goal_persona_fold_2_test_side", RD.judge_slot("gpt-oss-120b", "low"),
                RD.scorer_slot(DEFAULT_SCORER_VERSION))
assert RD.parse(name).method_id == slot and RD.parse(name).name == name
print(name)
PY
) || { echo "STOP: method id does not build / parse"; exit 1; }
  echo "=== u$U: --method-id $MID"
  # termination file: current metric ids only, and fold 2's (K+1 over the 9 test sessions, rollouts over test x 8 seeds)
  $BASEPY - "$TERM" "$O" "$CORPUS_F2" <<'PY' || { echo "STOP: termination file check failed"; exit 1; }
import json, sys
sys.path.insert(0, "tools")
from metrics import termination
p, O, corpus = sys.argv[1], sys.argv[2], sys.argv[3]
keys = termination.load(p)                 # the scorer's own reader: a retired id stops here
d = json.load(open(p, encoding="utf-8"))
assert d.get("_num_sessions_in_extra_turn_after_complete_human_session_probe") == 9, "K+1 probe is not 9 sessions"
prov = d.get("_provenance") or {}
assert all(str(v).startswith(O + "/") for v in prov.values()), "termination inputs outside %s: %r" % (O, prov)
rids = {json.loads(l)["record_id"] for l in open(corpus, encoding="utf-8") if l.strip()}
k1 = [json.loads(l) for l in open(prov["extra_turn_probe_path"], encoding="utf-8") if l.strip()]
assert {r["record_id"] for r in k1} == rids, "K+1 probe sessions != fold-2 test sessions"
print("termination OK:", {k: v for k, v in keys.items()})
PY
  # the 2AFC calibration record, read by the scorer's own reader
  (cd tools && $BASEPY -c "
from pathlib import Path
from metrics import two_alternative_forced_choice as F
g = F.read_gates(Path('../$GATES'))
assert g.get('judge') == '$JMODEL' and g.get('gates_pass') is True, g.get('problems')
assert (F.gates_record_domain(Path('../$GATES'), g) or 'main_dataset_search') == 'main_dataset_search'
print('2AFC gates OK: positive control', g['positive_control_human_vs_known_far_from_human_method']['rate'])
") || { echo "STOP: 2AFC calibration record check failed"; exit 1; }
  if [ ! -f "$GEN" ]; then
    echo "u$U: no generations yet ($GEN)"; [ -n "${PREFLIGHT_ONLY:-}" ] && continue; echo "STOP"; exit 1
  fi
  # generations: the scorer's reader, the retired-flag refusal, one row per gold user turn of fold 2, END invariants
  (cd tools && $BASEPY - "$GEN" "$CORPUS_F2" <<'PY') || { echo "STOP: generations check failed"; exit 1; }
import json, sys
sys.path.insert(0, ".")
from metrics import termination, corpus as C
gen, corpus = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(gen, encoding="utf-8") if l.strip()]
termination.refuse_retired_generation_row_flags(rows, gen)
recs, gold, _ = C.load_gold("main_dataset_search", corpus)
keys = {(r["record_id"], int(r["turn_index"])) for r in rows}
assert len(keys) == len(rows), "duplicate (record_id, turn_index)"
assert keys == set(gold), "rows != gold user turns: missing %s extra %s" % (sorted(set(gold) - keys)[:3], sorted(keys - set(gold))[:3])
for r in rows:
    for k in ("record_id", "turn_index", "is_first_turn", "discipline", "stage", "intent_variant", "greedy",
              "greedy_end_decision", "samples", "sample_end_decisions"):
        assert k in r, (k, r["record_id"], r["turn_index"])
    assert r["greedy_end_decision"] == (not r["greedy"].strip()), (r["record_id"], r["turn_index"])
    assert len(r["samples"]) == len(r["sample_end_decisions"]) == 3
    assert "⟨redacted" not in r["greedy"]
n_end = sum(r["greedy_end_decision"] for r in rows)
print("generations OK: %d rows = %d gold user turns, %d END rows" % (len(rows), len(gold), n_end))
PY
  [ -n "${PREFLIGHT_ONLY:-}" ] && { echo "u$U: preflight OK (PREFLIGHT_ONLY: not scored)"; continue; }
  if ! curl -s -m 10 $JUDGE_URL/models | grep -q "\"$JMODEL\""; then
    echo "STOP: the judge $JMODEL does not answer at $JUDGE_URL/models."
    echo "      Generation does not need it, scoring does: start gpt-oss-120b on :8029 (e.g. on the GPU the generation"
    echo "      just freed: start_servers6.sh / the r0_vllm serve.sh) and re-run: bash $0 $WHICH"
    exit 1
  fi
  if ! $VENV/bin/python -c "import yaml, numpy, scipy, sklearn, sentence_transformers, torch" 2>/dev/null; then
    echo "creating $VENV (consistent-test + scikit-learn + sentence-transformers)"
    $BASEPY -m venv --system-site-packages $VENV && $VENV/bin/pip install -q "scikit-learn" "sentence-transformers" \
      || { echo "STOP: venv setup failed"; exit 1; }
  fi
  $VENV/bin/python -c "import numpy, scipy, sklearn, sentence_transformers, torch; print('scorer env:', 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'sklearn', sklearn.__version__, 'sentence-transformers', sentence_transformers.__version__, 'torch', torch.__version__)"
  OUT=$O/results/$MID/main_dataset_search.json
  mkdir -p $O/results/$MID $O/judge_cache $O/dump
  # score provenance: the scoring env's package versions and both judge servers (the 2AFC gates were calibrated on
  # :8011 -- instruments/twoafc_gates_gpt-oss-120b.json "endpoint" -- we score on :8029; same model name and snapshot
  # is what we can check, the launch flags of :8011 only if that server still runs)
  SP=$O/results/$MID/score_env_provenance.json
  { for port in 8029 8011; do
      echo "port $port models: $(curl -s -m 5 http://127.0.0.1:$port/v1/models | tr -d '\n' | cut -c1-600)"
      echo "port $port launch: $(ps -eo user,args | grep -E 'vllm serve' | grep -E -- "--port $port( |$)" | grep -v grep | cut -c1-600)"
    done; } > $O/results/$MID/judge_servers.txt 2>&1
  $VENV/bin/python - "$SP" "$O/results/$MID/judge_servers.txt" "$GATES" "$JUDGE_URL" <<'PY' || { echo "STOP: score provenance failed"; exit 1; }
import json, sys
from importlib import metadata
sp, servers, gates, url = sys.argv[1:5]
pk = {}
for n in ("numpy", "scipy", "scikit-learn", "sentence-transformers", "torch", "transformers", "PyYAML"):
    try:
        pk[n] = metadata.version(n)
    except metadata.PackageNotFoundError:
        pk[n] = None
g = json.load(open(gates, encoding="utf-8"))
out = {"python": sys.executable, "packages": pk,
       "benchmark_requirements_pins": {"numpy": "2.2.6", "scipy": "1.18.0", "scikit-learn": "1.9.0",
                                       "sentence-transformers": "5.6.1", "torch": "2.10.0"},
       "judge_endpoint_scored": url, "twoafc_gates_endpoint_calibrated": g.get("endpoint"),
       "twoafc_gates_note": "the 2AFC calibration record was measured on %s; this run scores on %s. Same model name "
                            "(gpt-oss-120b); identity of weights / launch flags is only as good as judge_servers.txt"
                            % (g.get("endpoint"), url),
       "judge_servers": open(servers, encoding="utf-8").read().splitlines()}
json.dump(out, open(sp, "w", encoding="utf-8"), indent=1)
print("score env:", pk, "| 2AFC gates calibrated on", g.get("endpoint"), "scored on", url)
PY
  echo "=== scoring u$U $(date)"
  $VENV/bin/python tools/score_method.py \
    --method-id $MID --domain main_dataset_search \
    --generations $GEN --corpus $CORPUS_F2 \
    --split-manifest $MANIFEST --fold $FOLD --test-goals $TEST_GOALS \
    --skip requirement_item_overlap,human_utterance_copying \
    --termination $TERM \
    --judge-model $JMODEL --judge-base-url $JUDGE_URL \
    --reasoning-effort $EFFORT --two-alternative-forced-choice-reasoning-effort $EFFORT \
    --two-alternative-forced-choice-gates $GATES \
    --workers 6 --judge-cache-dir $O/judge_cache \
    --out $OUT --dump-per-turn $O/dump > $O/score_u$U.log 2>&1
  rc=$?
  grep -vE "it/s|Loading|warnings.warn" $O/score_u$U.log | tail -15
  [ $rc -eq 0 ] || { echo "STOP: score_method.py rc=$rc (log $O/score_u$U.log)"; exit 1; }
  grep -q "TWOAFC-BLOCKED" $O/score_u$U.log && echo "WARNING: the 2AFC was BLOCKED (>5% unparsed); see the log"
  $BASEPY - "$OUT" "$GEN" <<'PY' || { echo "STOP: results file check failed"; exit 1; }
import hashlib, json, sys
d = json.load(open(sys.argv[1], encoding="utf-8")); p = d["_provenance"]
assert p["split"]["fold"] == 2 and p["split"]["num_generation_rows"] == 40 == p["split"]["num_test_user_turns"], p["split"]
assert p["generations_sha256"] == hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
assert p["judge_model"] == "gpt-oss-120b" and p["judge_reasoning_effort"] == "low" and p["end_policy"] == "count"
print("results OK:", sys.argv[1], "| scorer", p["scorer_version"], "| families", sorted(p.get("scored_turns_by_family", {})))
PY
done
echo "bench_score_f2 done $(date)"
