#!/bin/bash
# SPEC v18 §8 benchmark suite of the FINAL v18 policy on fold 2's test side, comparable with bench_eval_f2_v17 (same
# scorer v3, local gpt-oss-120b judge at effort low, the same slicing / skips / termination protocol):
#  1. termination inputs from runs/pend_f2_v18/test.jsonl (the final policy's Task 1 end probabilities / K+1 / Task 2
#     rollouts) -> tools/probe_termination.py -> termination_u<k>.json; tools/score_stop.py with the paired conversation
#     bootstrap against v17 SFT (u0) and v17 GRPO (u5) (bench_eval_f2_v17/decisions_u{0,5}.jsonl);
#  2. bench_tf_generate.py --update <final> --reranker <the run's json when its gate passed> (Planner vLLM + Ditto on the
#     training GPU from the placeholder; OOM retried) -> generations_u<k>.jsonl (provenance: reranker sha + gate);
#  3. bench_score_f2_v18.sh <k> (scorer, gpt-oss on :8029) -> results/<slot ..._v18_fold_2_grpo_rerank_u<k>...>;
#  4. bench_boot_v18.py: paired conversation bootstrap of the per-turn dumps vs v17 u0 / u5.
# The benchmark repo is read only; everything is written under bench_eval_f2_v18. Releases every GPU we hold at the end
# unless KEEP_SERVERS=1. Usage: run_v18_bench.sh [F]   (F = 2)
set -uo pipefail
F=${1:-2}
[ "$F" = "2" ] || { echo "STOP: SPEC v18 is a fold-2 trial"; exit 1; }
B=/tmp2/hchsu/trec2026-usersim-benchmark
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v18; RUN=$G/runs/pend_f${F}_v18; H=${HOLD_DIR:-$G/hold}
O=$G/bench_eval_f2_v18; O17=$G/bench_eval_f2_v17; L=$G/labels_v18; RERANK=$L/reranker_v18_f${F}.json
PYDIR=/home/mzjiang/miniconda3/envs/consistent-test/bin; PY=$PYDIR/python
export PATH=$PYDIR:$PATH PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID OPENAI_API_KEY=local-vllm-unused PYTHONDONTWRITEBYTECODE=1
export R0_CONTEXT=${OSS_MAX_MODEL_LEN:-32768}   # 2026-10-02: = gpt-oss max_model_len (start_servers6.sh); 12288 overflowed
export Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
PREFIX=sep_sim_pend_qwen3_4b_planner_ditto_8b_speaker_v18_fold_2
held() { cat $H/status_$1 2>/dev/null || echo 0; }
settarget() { echo $2 > $H/target_$1.tmp && mv -f $H/target_$1.tmp $H/target_$1; }
TN=${TRAIN_NEED_GIB:-45}
TG=""
holder_ok() { pgrep -u mzjiang -f "python.* .*gpu_holder3\.py" > /dev/null && [ -f $H/plan ] && [ $(( $(date +%s) - $(stat -c %Y $H/plan) )) -lt 60 ]; }
holder_check() { holder_ok || { echo "STOP: the GPU holder (gpu_holder3.py) is not running or not writing $H/plan (HOLD_DIR?)"; exit 1; }; }
take_train_gpu() {
  n=0; until [ -f $H/role_train ]; do [ $((n % 300)) -eq 0 ] && { holder_check; echo "WAITING_GPU: no committed placement yet ($(date +%H:%M))"; }; n=$((n + 1)); sleep 1; done
  TG=$(cat $H/role_train); n=0
  settarget $TG $TN
  until [ "$(held $TG)" -ge $TN ]; do [ $((n % 300)) -eq 0 ] && { holder_check; echo "WAITING_GPU: GPU $TG holds $(held $TG)/$TN GiB ($(date +%H:%M))"; }; n=$((n + 1)); sleep 1; done
  settarget $TG 0; n=0
  until [ "$(held $TG)" -le 0 ]; do n=$((n + 1)); [ $((n % 300)) -eq 0 ] && { holder_check; echo "WAITING_GPU: GPU $TG still holds $(held $TG) GiB ($(date +%H:%M))"; }; sleep 1; done
  GPU=$TG
}
release() {
  [ -n "$TG" ] && settarget $TG $TN
  if [ "${KEEP_SERVERS:-0}" = "1" ]; then echo "BENCH: placeholder / servers kept $(date)"; return; fi
  touch $H/stop; sleep 3; pkill -u mzjiang -f "python.* .*gpu_holder3\.py"
  for port in 8029 8031; do
    pkill -u mzjiang -f "vllm serve .*--port $port"
    for i in $(seq 1 60); do pgrep -u mzjiang -f "vllm serve .*--port $port" > /dev/null || break; sleep 2; done
    pkill -9 -u mzjiang -f "vllm serve .*--port $port"
  done
  sleep 5
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "mzjiang" ] || continue
    ps -o args= -p $p 2>/dev/null | grep -qE "vllm|VLLM|EngineCore" && kill -9 $p 2>/dev/null
  done
  echo "BENCH: GPUs released $(date)"
}
if pgrep -u mzjiang -f "train_planner_rl.py|eval_test_rl.py|smoke_v1[78].py|run_v1[78]_fold|bench_tf_generate.py|train_reranker.py" > /dev/null; then
  echo "BENCH: our training / evaluation is running - not starting"; exit 1; fi
cd $C || { echo "STOP: no snapshot $C"; exit 1; }
sha256sum -c --quiet local_sha_v18.txt || { echo "STOP: pend_v18 snapshot sha mismatch"; exit 1; }
pgrep -u mzjiang -f "python.* .*(gpu_holder2|gpu_grab)\.py" > /dev/null && { echo "STOP: an old placeholder is running"; exit 1; }
[ -f $RUN/test_meta.jsonl ] || { echo "STOP: no v18 test evaluation (run_v18_test.sh first)"; exit 1; }
FU=$($PY -c "import json; print(json.load(open('$RUN/final.json'))['final_update'])") || exit 1
SEL=$($PY -c "import json; print(json.load(open('$RUN/ckpt/u%05d/rl_manifest.json' % $FU))['selector'])") || exit 1
RER=""; [ "$SEL" = "borda_rerank" ] && RER="--reranker $RERANK"
mkdir -p $O
echo "=== BENCH v18 final u$FU (selector $SEL) $(date)"
# ---- 1. termination inputs (r_benchA5.sh for v17, here the final v18 policy)
$PY - "$RUN" "$O" "$FU" "$O17" <<'EOF' || { echo "STOP: termination inputs failed"; exit 1; }
import json, sys
sys.path.insert(0, "/tmp2/hchsu/trec2026-usersim-benchmark/tools")
from metrics import metric_ids as M
R, O, fu, O17 = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rows = [json.loads(l) for l in open(R + "/test.jsonl") if l.strip()]
psha = [r for r in rows if r.get("kind") == "summary" and r["update"] == fu][-1]["policy_sha"]
dec, k1, roll, pos = [], [], [], {}
for r in rows:
    if r.get("policy_sha") != psha:
        continue
    if r.get("kind") == "task1":
        t = r["task1"]; n = t["n_real"]
        for e in r["end_probs"]:
            if not e.get("valid"):
                continue
            pos[(r["conversation_id"], e["t"])] = {"conversation_id": r["conversation_id"], "user_turn_index": e["t"],
                                                   "num_user_turns_in_session": n, "human_last_user_turn_label": int(e["t"] == n)}
            dec.append({"conversation_id": r["conversation_id"], "user_turn_index": e["t"], "p_stop": e["p_end"],
                        "stopped": bool(e["greedy_end"])})
        k1.append({"record_id": t["record_id"], "ended": bool(t["k1_ended"])})
    if r.get("kind") == "episode":
        e = r["episode"]
        roll.append({M.ROLLOUT_EPISODE_USER_TURNS: e["turns"], M.ENDED_BY_END_DECISION_OR_EMPTY_OUTPUT: e["end_kind"] == "planner_end"})
for name, data in (("decisions", dec), ("k1", k1), ("rollout", roll)):
    with open(O + "/%s_u%d.jsonl" % (name, fu), "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
# the positions of v17 u0 / u5 and of v18 (a position is (conversation, t), its label t == n): their union, so every
# method's decisions are a subset (score_stop --subset-ok); a disagreeing label would be a data error
for l in open(O17 + "/positions_goal_persona_fold_2_test_side_turn2plus.jsonl"):
    if l.strip():
        p = json.loads(l)
        k = (p["conversation_id"], p["user_turn_index"])
        assert k not in pos or pos[k] == p, ("position label differs", k)
        pos[k] = p
with open(O + "/positions_goal_persona_fold_2_test_side_turn2plus_union.jsonl", "w") as f:
    for k in sorted(pos):
        f.write(json.dumps(pos[k]) + "\n")
print("u%d: decisions %d k1 %d rollout %d" % (fu, len(dec), len(k1), len(roll)))
EOF
POS=$O/positions_goal_persona_fold_2_test_side_turn2plus_union.jsonl
(cd $B && $PY tools/probe_termination.py --extra-turn-probe $O/k1_u$FU.jsonl --rollout $O/rollout_u$FU.jsonl --out $O/termination_u$FU.json) \
  || { echo "STOP: probe_termination failed"; exit 1; }
cat $O/termination_u$FU.json
for BU in 0 5; do
  BID=$([ $BU = 0 ] && echo sep_sim_pend_qwen3_4b_planner_ditto_8b_speaker_v17_fold_2__sft_u0 || echo sep_sim_pend_qwen3_4b_planner_ditto_8b_speaker_v17_fold_2__grpo_u5)
  (cd $B && $PY tools/score_stop.py --method-id ${PREFIX}__grpo_rerank_u$FU --domain main_dataset_search --decisions $O/decisions_u$FU.jsonl \
      --positions $POS --baseline-decisions $O17/decisions_u$BU.jsonl --baseline-id $BID --subset-ok \
      --out $O/stop_u${FU}_vs_v17_u$BU.json 2>&1 | tail -5) || { echo "STOP: score_stop failed"; exit 1; }
done
# ---- 2. generation (the placeholder hands the training GPU over to Ditto; the Planner vLLM serves the final adapter)
trap release EXIT
if ! pgrep -u mzjiang -f "python.* .*gpu_holder3\.py" > /dev/null; then
  rm -rf $H; mkdir -p $H
  PYTHONNOUSERSITE=1 setsid nohup $PY $G/gpu_holder3.py $H >> $G/gpu_holder3.log 2>&1 < /dev/null &
  sleep 10
fi
GEN=$O/generations_u$FU.jsonl
if [ -f $GEN ] && [ -f $GEN.provenance.json ] && grep -q '"finished"' $GEN.provenance.json; then
  echo "generations exist - kept ($GEN)"
else
  rc=1
  for attempt in $(seq 1 10); do
    bash $G/start_servers6.sh > $G/servers_check_bench_v18.log 2>&1 || { echo "STOP: a server failed to start"; tail -8 $G/servers_check_bench_v18.log; exit 1; }
    take_train_gpu
    SZ=$(stat -c %s $O/gen_u$FU.log 2>/dev/null || echo 0)
    $PY bench_tf_generate.py --final --run-dir $RUN --update $FU --fold $F --planner-path $Q4 --gpu $GPU --workers 4 \
        $RER --k1-probe $O/k1_u$FU.jsonl --force-unload --out $GEN >> $O/gen_u$FU.log 2>&1
    rc=$?; settarget $TG $TN; TG=""
    [ $rc -eq 0 ] && break
    if tail -c +$((SZ + 1)) $O/gen_u$FU.log | grep -cE "CUDA out of memory|CUDA error: out of memory|OutOfMemoryError" > /dev/null; then
      echo "OOM -> retry (finished conversations are reused)"; sleep 60; continue; fi
    echo "STOP: generation failed"; tail -20 $O/gen_u$FU.log | cut -c1-300; exit 1
  done
  [ $rc -eq 0 ] || { echo "STOP: 10 OOM retries used"; exit 1; }
fi
grep -A3 -i "agreement" $O/gen_u$FU.log | head -8
# ---- 3. scoring (gpt-oss on :8029, started by start_servers6.sh)
bash $G/start_servers6.sh > $G/servers_check_bench_v18.log 2>&1 || { echo "STOP: a server failed to start"; exit 1; }
bash $C/bench_score_f2_v18.sh $FU > $O/score_all.log 2>&1 || { echo "STOP: scoring failed"; tail -20 $O/score_all.log; exit 1; }
tail -8 $O/score_all.log
# ---- 4. paired conversation bootstrap of the per-turn dumps vs v17 u0 / u5
D18=$(ls $O/dump/${PREFIX}_grpo_rerank_u${FU}__*__main_dataset_search.json | head -1)
D0=$(ls $O17/dump/*v17_fold_2_sft_u0__*__main_dataset_search.json | head -1)
D5=$(ls $O17/dump/*v17_fold_2_grpo_u5__*__main_dataset_search.json | head -1)
$PY bench_boot_v18.py --dump-v18 $D18 --dump-u0 $D0 --dump-u5 $D5 --corpus $O/corpus_goal_persona_fold_2_test_side.jsonl \
    --out $O/bench_boot_v18_u$FU.json > $O/bench_boot_v18_u$FU.txt 2>&1 || { echo "STOP: bootstrap failed"; tail -5 $O/bench_boot_v18_u$FU.txt; exit 1; }
grep -E "length_w1|empty_output|repeating" $O/bench_boot_v18_u$FU.txt | head -12
echo "BENCH v18 DONE $(date): results under $O/results, stop_u${FU}_vs_v17_u{0,5}.json, termination_u$FU.json, bench_boot_v18_u$FU.json"
echo "NOTE (SPEC v18 §6.5): the 2AFC reads the first non-empty SAMPLE -- a candidate we did not output, chosen by the reranker's pick; never read a 2AFC change as 'the output is more human-like'."
