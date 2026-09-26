G=/tmp2/mzjiang_usersim/grpo_planner
du -sh $G/stageC_v1/rep0/nogate.jsonl $G/nested $G/trec_inner_fold*_7b_v1/epoch2 $G/prism_sft_7b_v1/best
timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=15 140.109.21.221 'echo direct-ok; hostname' 2>&1 | tail -2
