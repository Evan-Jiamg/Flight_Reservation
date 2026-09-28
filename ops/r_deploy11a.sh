G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/pend_v11
if [ -e $C ]; then echo "pend_v11 exists already"; exit 1; fi
mkdir -p $C && cp $G/code_snapshots/pend_v10/*.py $C/ && chmod u+w $C/*.py && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && echo "part a ok: $(ls $C/*.py | wc -l) py files"
