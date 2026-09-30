nvidia-smi -L
nvidia-smi --query-gpu=index,name,pci.bus_id,uuid --format=csv,noheader
