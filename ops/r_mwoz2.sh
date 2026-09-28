B=/tmp2/hchsu/trec2026-usersim-benchmark
grep -n -i "multiwoz\|MWOZ\|data" $B/run_eg1_score_multiwoz.sh | head -15
ls /tmp2/mzjiang_usersim/xdom/data 2>/dev/null; ls -la /tmp2/mzjiang_usersim/xdom/data/multiwoz* 2>/dev/null | head
find /tmp2/mzjiang_usersim /tmp2/hchsu -maxdepth 5 -iname "*multiwoz*" -type d 2>/dev/null | head -10
python3 - <<'EOF'
import json
m=json.load(open("/tmp2/hchsu/trec2026-usersim-benchmark/domains/multiwoz/manifest.json"))
print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for k,v in m.items()})
r=m.get("records") or m.get("sessions")
if r: print(json.dumps(r[0])[:400])
EOF
