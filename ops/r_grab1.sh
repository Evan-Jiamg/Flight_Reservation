G=/tmp2/mzjiang_usersim/grpo_planner; D=$G/grab1
echo "$PAYLOAD_B64" | base64 -d | tar xz -C $G gpu_grab.py && echo "gpu_grab.py installed"
pgrep -u mzjiang -f "gpu_grab.py" > /dev/null && { echo "grabber already running"; exit 0; }
rm -rf $D; mkdir -p $D
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 setsid nohup /home/mzjiang/miniconda3/envs/consistent-test/bin/python $G/gpu_grab.py $D >> $G/gpu_grab1.log 2>&1 < /dev/null &
sleep 25
tail -5 $G/gpu_grab1.log
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep 3797edfd
