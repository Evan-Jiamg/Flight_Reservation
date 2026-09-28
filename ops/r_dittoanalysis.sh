G=/tmp2/mzjiang_usersim/grpo_planner; C=$G/code_snapshots/ditto_analysis_v1
mkdir -p $C && cd $C && echo "$PAYLOAD_B64" | base64 -d | tar xzf - && sha256sum *.py > SHA256SUMS && chmod -w *.py
PYTHONNOUSERSITE=1 HF_HOME=/tmp2/hf_shared setsid nohup /home/mzjiang/miniconda3/envs/blackwell-kto-test/bin/python stageD_ditto_analysis.py --replicate 0 --gpu 1 > $G/ditto_analysis_rep0.log 2>&1 < /dev/null &
sleep 60; tail -3 $G/ditto_analysis_rep0.log | cut -c1-200; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
