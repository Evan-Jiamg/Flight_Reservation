D=/tmp2/mzjiang_usersim/xdom/data/multiwoz
python3 - <<'EOF'
import json
D="/tmp2/mzjiang_usersim/xdom/data/multiwoz"
r=json.loads(open(D+"/subset_v1.jsonl").readline())
def shape(x,d=0):
    if isinstance(x,dict): return {k:shape(v,d+1) for k,v in list(x.items())[:12]} if d<3 else "dict"
    if isinstance(x,list): return [shape(x[0],d+1)] if x else []
    return (str(x)[:60])
print(json.dumps(shape(r),indent=1)[:2500])
g=json.load(open(D+"/goals_test.json"))
k=list(g)[:1] if isinstance(g,dict) else None
print(type(g).__name__, len(g), k, json.dumps(g[k[0]] if k else g[0])[:600])
EOF
