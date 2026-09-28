G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v10
[ -d $C ] || { echo "pend_v10 missing: run part a first"; exit 1; }
cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - || exit 1
cat > $C/LOCAL_SHA256 <<'SHAEOF'
fde22d3be9fbe64c84d9943e734e4152297c923041814f7b0b26737125a9f8f5  implicit_profile.py
bee81bbf1b3936a126d9db4344061e747e307817622308b1e91a558198f03fb9  planner_prompt_v3.py
31493a88a1bc1bc3d6e5c5c32aefda01d79cd278ce5d549a700d39694eb34697  rl_algos.py
30201b2954b7d576cef580b53e1ecd281faf601a1ef647541548ceda92289ef9  rl_controllers.py
67b8f5a719b729b5bac82ee3dbd1334005bbf87d30fa8895827851f3c0a633ee  rl_reward.py
ba3d597ae9185a5af7e48909a85fccbf5ba544eecbd403a461f4a3e345f55cb9  rollout_v4.py
156e2f75684967652adaaf5762eb48b546b3f2ff4f805672ff8bf3720811d9d5  task1_stop.py
f1f7a7bc3c853411b9e4db0a8fb5f0938e723d34cc9a98bdc263c8c09ef49821  task1_v4.py
8acbcd995e27d7c45b4ba0a4de3bf1c761ea73176a925d76bec7637638a9c50b  task2_env.py
86ce6abbe9b0375d91a95b172786daee28daf26574052aee09c6e197339040cf  test_batching_server.py
f3ce373f3909ebd1d965350302941a0e8e465cb36e326e92188e0a97adcba7c6  test_llm4_controller.py
373cde1510c6e6e30d9cdc0682e346487579a95107c96b76755d708c917fcc41  test_pend.py
13247bd2096ddf00b7a4d986881c93a3a942804a1f4a339057899b9a83e0b1b7  test_rl_advantages.py
d3b14e181d0269fc3fef220e1525974d2c296d9850c33a40da2635be21d55bc7  test_verify_pipeline.py
5b5c3de7b816de4170e99303ba5db7c42a596d60e04c7796d9b22c4f20de4ca0  train_planner_rl.py
e32d4832d91b92af7acb5edc6808adcb2d52e82b05b7430ebc6175967727b532  verify_pipeline.py
828b97b0e1a82aa4da5d4b4daca5908e096c1cb2e5e856010a8ba661600fb801  verify_task1.py
83067f72ddbdeb4306b5058cdad920eb65be61bd91b5f5302a4201fd9241ddfc  style_select.py
c18fb651c6ad534d939b8c966357f033f9ff688b482134d57cd84f93edf18ceb  ditto_e16.py
3f97f0b20dc102cc4b3bb291008b49cc1f2fdf1350ee9783991ee5e0d3a7d365  fit_prompts.py
f7ca29f24ae62c6c2606ba48207bfff9685e3335349b2009907d988b32d0f555  batching.py
8688ee60a4c29456f080a78efd3405c42623df6155c2721b98d42d0196e10ed8  task2_episode.py
6d70c0c11dc789c6a762f8a67acc604dab4c7d256068763240b9c69cd21a2b46  diag_t1.py
4cb6fbf8ce67fe09533ad2de87bd622986e8221d79b662be2b0142b13733c236  test_rl_algos_server.py
ce8b38e01ec2276ab81bbd449b71a6b541a3b28b54dc6c802d145384e3b26a20  test_audit4.py
933ebc635b65481ac3ef1b945408e0dc8f29f02d61825cb1ca83108ac10f2dee  test_judge_wrapper.py
SHAEOF
if ! sha256sum -c --quiet LOCAL_SHA256; then echo "SHA MISMATCH vs local worktree: not launching"; exit 1; fi
sha256sum *.py > SHA256SUMS && chmod a-w *.py
echo "part b ok; local sha check passed"
mkdir -p $G/runs
cat > $G/run_v10_formal.sh <<'RUNEOF'
#!/bin/bash
# FORMAL GRPO run, fold 2, pend v10 -- gate stage: --updates 1 (validation at u0 + update 1), then STOP.
# Continued later with --resume --updates N only after the gate checks pass and the user agrees.
set -uo pipefail
G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v10; RUN=$G/runs/pend_f2_v10
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
export PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export R0_BASE_URL=http://127.0.0.1:8029/v1 R0_MODEL=gpt-oss-120b R0_REASONING_EFFORT=low
export JUDGE_BASE_URL=http://127.0.0.1:8029/v1 JUDGE_MODEL=gpt-oss-120b JUDGE_REASONING_EFFORT=low
export CONTROLLER_BASE_URL=http://127.0.0.1:8029/v1 OPENAI_API_KEY=local-vllm-unused
Q4=$(ls -d /tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ | head -1)
pickgpu() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \
            awk -F', ' '{f=int(($2-$3)/1024); if (f>=42) print f" "$1}' | sort -nr | head -1 | cut -d" " -f2; }
cd $C
echo "=== 0 preflight $(date)"
if ! curl -s -m 15 http://127.0.0.1:8029/v1/models | grep -q gpt-oss-120b; then
  echo "ABORT: the gpt-oss-120b server on port 8029 is not answering (not restarting it without the user)"; exit 1; fi
T=""; for f in test_pend_generate.py test_implicit_profile.py test_batching.py test_e16_port.py test_pend.py test_rl_reward.py \
    test_rl_controllers.py test_llm4_controller.py test_rl_advantages.py test_fit_prompts.py test_task2_episode.py \
    test_verify_pipeline.py test_planner_prompt_v3.py test_audit4.py test_judge_wrapper.py; do
    if [ -f $f ]; then T="$T $f"; else echo "missing in snapshot: $f"; fi; done
E1R_TREE=$G/trees/e1r_cf19400 V2FIX_TREE=/home/mzjiang/Sep-Simulator $PY -m pytest -q -p no:cacheprovider $T > $G/v10_unit.txt 2>&1
rc=$?; tail -4 $G/v10_unit.txt
if [ $rc -ne 0 ]; then echo "ABORT: unit tests failed (see $G/v10_unit.txt)"; exit 1; fi
echo "=== 1 waiting for the v9 smoke to finish and for a free GPU $(date)"
while pgrep -u mzjiang -f run_v9_smoke.sh > /dev/null; do sleep 60; done
GPU=""; while [ -z "$GPU" ]; do GPU=$(pickgpu); [ -z "$GPU" ] && sleep 60; done
echo "=== 2 FORMAL GRPO gate run on GPU $GPU $(date)"
if [ -e $RUN/ckpt/LATEST.json ]; then echo "ABORT: $RUN already has checkpoints"; exit 1; fi
mkdir -p $RUN
$PY train_planner_rl.py --fold 2 --planner-path $Q4 --gpu $GPU --G 4 --scenarios-per-update 4 --task1-convs 4 \
    --updates 1 --val-every 5 --rollout-workers 4 --out $RUN > $RUN/train.log 2>&1
echo "train rc=$? $(date)"; grep -v "Loading weights" $RUN/train.log | tail -4 | cut -c1-300
echo "=== 3 verify"
$PY verify_pipeline.py --rl-dir $RUN --splits $G/splits_v1.json --fold 2 --split train --arm pend \
    --sepsim-path $G/trees/e1r_cf19400 > $RUN/verify.txt 2>&1
echo "verify rc=$?"; grep -E "FAIL|WARN|PASSED|FAILED" $RUN/verify.txt | cut -c1-260 | head -30
echo "=== 4 gate metrics"
$PY - <<'PYEOF'
import json, collections
O = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v10/"
for l in open(O + "updates.jsonl"):
    u = json.loads(l); st = u["learner_stats"]; a = u["train_aggregate"]
    print("UPDATE %d update_s %d timing %s" % (u["update"], u["update_s"], u.get("timing")))
    print("  rl_grad_norm %s aux_grad_norm %s grad_norm(total) %s aux_loss %s aux_p_end %s optimizer_steps %s" % (
          st.get("rl_grad_norm"), st.get("aux_grad_norm"), st.get("grad_norm"), st.get("aux_loss"),
          st.get("aux_p_correct_end"), st.get("optimizer_steps")))
    print("  ratio_init_maxdev %s kl %s n_samples %s n_tokens %s" % (st.get("ratio_init_maxdev"), st.get("kl"), u["n_samples"], st.get("n_tokens")))
    print("  reward %.3f shadow %.3f comps %s" % (a["reward_mean"], a["shadow_reward_mean"], a["components_mean"]))
    print("  turn_hist %s unclean %s singletons %s aux_w %s task1 %s" % (a["turn_hist"], a["n_unclean_episodes"],
          a["n_dropped_singleton_episodes"], a["aux_weight"], a["task1_train"]))
    print("  q %s p_h %s" % ([round(x, 3) for x in u["reward_ctx"]["q"]], [round(x, 3) for x in u["reward_ctx"]["p_h"]]))
tot = collections.Counter(); n = collections.Counter(); inc = collections.Counter(); eps = 0
for l in open(O + "rollouts.jsonl"):
    e = json.loads(l)["episode"]; eps += 1
    for k, v in (e.get("episode_counters") or {}).items():
        inc[k] += v
    for s in e["trace"]:
        for k in ("planner_s", "speaker_s", "r0_s", "ledger_s"):
            if s.get(k) is not None:
                tot[k] += s[k]; n[k] += 1
print("EPISODES %d incidents %s" % (eps, dict(inc)))
print("PER-STEP MEAN SECONDS (4 workers, wall incl. waiting):", {k: round(tot[k] / n[k], 1) for k in tot}, "steps", dict(n))
for l in open(O + "validation.jsonl"):
    v = json.loads(l)
    if v["kind"] == "summary":
        print("VAL u%d validation_s %s n=%d unclean=%s withheld=%s sel=%s turn_stats=%s task1=%s" % (v["update"], v.get("validation_s"),
              v["n_episodes"], v["n_unclean_episodes"], v.get("selection_withheld"), v["selection_score"], v["turn_stats"],
              {k: v["task1"][k] for k in ("term_f1", "premature", "k1_end_rate", "n_emitted_capped_turns")} if v["task1"] else None))
PYEOF
ls $RUN/ckpt/
echo "V10 GATE DONE $(date)"
RUNEOF
setsid nohup bash $G/run_v10_formal.sh > $G/run_v10_formal.log 2>&1 < /dev/null &
echo "formal gate run launched"
