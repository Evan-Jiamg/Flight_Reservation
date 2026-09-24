G=/tmp2/mzjiang_usersim/grpo_planner
cd $G && rm -f /tmp2/mzjiang_usersim/xfer_221.tgz
tar czf /tmp2/mzjiang_usersim/xfer_221.tgz stageC_v1/rep0/nogate.jsonl nested trec_inner_fold0_7b_v1/epoch2 trec_inner_fold1_7b_v1/epoch2 trec_inner_fold2_7b_v1/epoch2 prism_sft_7b_v1/best stop_prompt.py
sha256sum /tmp2/mzjiang_usersim/xfer_221.tgz; ls -la /tmp2/mzjiang_usersim/xfer_221.tgz
