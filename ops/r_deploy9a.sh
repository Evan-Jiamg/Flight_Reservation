G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v9
if [ -e $C ]; then echo "pend_v9 exists already"; exit 1; fi
mkdir -p $C && cp $G/code_snapshots/pend_v8/*.py $C/ && chmod u+w $C/*.py && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && echo "part a ok: $(ls $C/*.py | wc -l) py files"
