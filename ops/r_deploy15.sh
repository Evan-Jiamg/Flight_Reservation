G=/tmp2/mzjiang_usersim/grpo_planner; S=$G/code_snapshots; RUN=$G/runs/pend_f2_v11
echo "$PAYLOAD_B64" | base64 -d > /tmp2/mzjiang_usersim/v15deploy.tgz || exit 1
PY=/home/mzjiang/miniconda3/envs/consistent-test/bin/python
[ -d $S/pend_v15 ] && { echo "pend_v15 already exists"; exit 1; }
# 1. stop the all-candidate re-selection (user: only u5 and u25) - our own evaluation job, no training runs
echo "before:"; pgrep -u mzjiang -af "run_v14_reselect|train_planner_rl.py --fold 2" | cut -c1-120
pkill -u mzjiang -f run_v14_reselect.sh
sleep 2
RP=$(pgrep -u mzjiang -f "train_planner_rl.py --fold 2.*--reselect-seeds")
[ -n "$RP" ] && kill $RP
for i in $(seq 1 60); do pgrep -u mzjiang -f "train_planner_rl.py --fold 2" > /dev/null || break; sleep 2; done
echo "after:"; pgrep -u mzjiang -af "run_v14_reselect|train_planner_rl.py --fold 2" | cut -c1-120; echo "(empty = stopped)"
echo "partial v14 re-selection rows kept in reselect.jsonl: $(wc -l < $RUN/reselect.jsonl) (no summary -> unused)"
# 2. archive of the training record
df -h /tmp2 | tail -1; du -sh $RUN
A=$G/archive/pend_f2_v11_u25_20260927
mkdir -p $A/code_snapshots $A/ops
cp -a $RUN $A/run
for v in pend_v11 pend_v12 pend_v13 pend_v14; do cp -a $S/$v $A/code_snapshots/; done
cp -p $G/splits_v1.json $A/
cp -p $G/run_v12_continue.sh $G/run_v12_guard.sh $G/run_v13_continue.sh $G/run_v13_guard.sh $G/start_servers5.sh $G/gpu_holder2.py $G/watch.sh $G/intervention_w_dist_v13.json $A/ops/
# 3. pend_v15 (also archived)
cp -a $S/pend_v14 $S/pend_v15 && chmod -R u+w $S/pend_v15 && tar xzf /tmp2/mzjiang_usersim/v15deploy.tgz -C $S/pend_v15 code
mv -f $S/pend_v15/code/* $S/pend_v15/ && rmdir $S/pend_v15/code
cd $S/pend_v15 && sha256sum -c --quiet local_sha_v15.txt && echo "pend_v15 sha OK ($(wc -l < local_sha_v15.txt) files)" || { echo "SHA MISMATCH"; exit 1; }
CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 nice -n 19 $PY -m pytest -q -p no:cacheprovider test_reselect.py test_intervention.py test_verify_pipeline.py test_rl_advantages.py 2>&1 | tail -1
cp -a $S/pend_v15 $A/code_snapshots/
tar xzf /tmp2/mzjiang_usersim/v15deploy.tgz -C $G --strip-components=1 ops/run_v15_reselect.sh ops/reselect_boot.py ops/watch.sh
cp -p $G/run_v15_reselect.sh $G/reselect_boot.py $A/ops/
tar xzf /tmp2/mzjiang_usersim/v15deploy.tgz -C $A --strip-components=1 ops/RESUME.md
chmod +x $G/run_v15_reselect.sh
# resume readiness of the latest checkpoint
python3 - <<'PYEOF'
import json, os
R = "/tmp2/mzjiang_usersim/grpo_planner/runs/pend_f2_v11/"
L = json.load(open(R + "ckpt/LATEST.json")); d = R + "ckpt/" + L["dir"]
st = json.load(open(d + "/state.json"))
print("LATEST", L, "files", sorted(os.listdir(d)))
print("state: update", st["update"], "intervention", st.get("intervention", {}).get("at_update"), "best", st["best"]["update"],
      "w_dist", st["cfg"]["w_dist"], "history", len(st["history"]))
PYEOF
cd $A && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
echo "archive: $(wc -l < $A/MANIFEST.sha256) files, $(du -sh $A | cut -f1) at $A"
chmod -R a-w $A/run $A/code_snapshots $A/ops $A/splits_v1.json $A/RESUME.md $A/MANIFEST.sha256
sha256sum -c --quiet $A/MANIFEST.sha256 && echo "archive manifest check OK"
# 4. the narrowed re-selection
setsid nohup bash $G/run_v15_reselect.sh > $G/run_v15_reselect.log 2>&1 < /dev/null &
sleep 30
echo "--- v15 log"; cat $G/run_v15_reselect.log | cut -c1-200
bash $G/watch.sh
